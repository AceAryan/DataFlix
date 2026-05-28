"""
DataFlix — Training Pipeline
scripts/train.py

Usage:
  python scripts/train.py                  # train all three
  python scripts/train.py --model als
  python scripts/train.py --model bpr
  python scripts/train.py --model hybrid
  python scripts/train.py --skip-als       # load existing ALS, train BPR+Hybrid
"""

import argparse, json, logging, pickle, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.config import (
    PROCESSED_DIR, RESULTS_DIR, DEVICE,
    TRAIN_CSV, VAL_CSV,
    CSR_MATRIX_PATH, BPR_DATA_PATH, USER_POSITIVES_PATH,
    SBERT_EMBEDDINGS_PATH, IMDB_FEATURES_PATH,
    POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH,
    ALS_PATH, BPR_FACTORS_PATH, HYBRID_CKPT_PATH,
    LATENT_DIM_K, ALS_ITERATIONS, ALS_REG,
    LR_BPR, BPR_REG, BPR_EPOCHS, BPR_BATCH_SIZE, BPR_SAMPLES_PER_EPOCH,
    LR_HYBRID, HYBRID_WEIGHT_DECAY, HYBRID_EPOCHS,
    HYBRID_BATCH_SIZE, HYBRID_SAMPLES_PER_EPOCH,
    EARLY_STOP_PATIENCE, FREEZE_EPOCHS,
    EMBED_DIM_D, NUM_HEADS,
    set_seed,
)
from src.models.als    import ALS
from src.models.bpr    import BPR
from src.models.hybrid import HybridModel, HybridTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_csr_cache = None


def _elapsed(t): s = time.time()-t; return f"{int(s//60)}m {s%60:.1f}s"

def _banner(title):
    log.info("")
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)

def _load_csr():
    global _csr_cache
    if _csr_cache is None:
        log.info(f"Loading CSR matrix...")
        _csr_cache = sparse.load_npz(CSR_MATRIX_PATH)
    return _csr_cache

def _get_dims():
    csr = _load_csr()
    return csr.shape  # (n_users, n_items)

def _load_splits():
    log.info("Loading train/val splits...")
    tr = pd.read_csv(TRAIN_CSV)
    va = pd.read_csv(VAL_CSV)
    log.info(f"  Train={len(tr):,} | Val={len(va):,}")
    return tr, va

def _load_bpr_data():
    log.info("Loading BPR data...")
    d         = np.load(BPR_DATA_PATH)
    all_items = d["all_items"]
    item_pop  = d["item_pop_values"]
    with open(USER_POSITIVES_PATH, "rb") as f:
        user_pos = pickle.load(f)
    log.info(f"  {len(user_pos):,} users | {len(all_items):,} items")
    return user_pos, all_items, item_pop

def _load_feature_tensors():
    log.info("Loading feature tensors...")
    t = {
        "sbert":   torch.load(SBERT_EMBEDDINGS_PATH,   weights_only=False),
        "imdb":    torch.load(IMDB_FEATURES_PATH,      weights_only=False),
        "pop":     torch.load(POPULARITY_PATH,          weights_only=False),
        "history": torch.load(HISTORY_EMBEDDINGS_PATH, weights_only=False),
    }
    for k, v in t.items():
        log.info(f"  {k:<10}: {tuple(v.shape)}")
    return t


# ── ALS ───────────────────────────────────────────────────────────────────────

def train_als(skip_if_exists: bool = False) -> ALS:
    _banner("ALS — Matrix Factorisation Baseline")
    if skip_if_exists and ALS_PATH.exists():
        log.info(f"Loading existing ALS factors...")
        return ALS.load(ALS_PATH)
    csr   = _load_csr()
    model = ALS(n_factors=LATENT_DIM_K, n_iterations=ALS_ITERATIONS, reg=ALS_REG)
    model.fit(csr)
    model.save(ALS_PATH)
    log.info(f"ALS done — final RMSE: {model.train_rmse_history[-1]:.5f}")
    return model


# ── BPR ───────────────────────────────────────────────────────────────────────

def train_bpr(als_model: ALS) -> BPR:
    _banner("BPR — Bayesian Personalized Ranking")
    n_users, n_items = _get_dims()
    user_pos, all_items, item_pop = _load_bpr_data()

    model = BPR(
        n_users=n_users, n_items=n_items,
        n_factors=LATENT_DIM_K, lr=LR_BPR, reg=BPR_REG,
        n_epochs=BPR_EPOCHS, batch_size=BPR_BATCH_SIZE,
        samples_per_epoch=BPR_SAMPLES_PER_EPOCH, device=DEVICE,
    )
    model.init_from_als(als_model)
    model.fit(user_pos, all_items, item_pop)
    model.save(BPR_FACTORS_PATH)
    log.info(f"BPR done — final loss: {model.train_loss_history[-1]:.5f}")
    return model


# ── Hybrid ────────────────────────────────────────────────────────────────────

def train_hybrid(als_model: ALS, bpr_model: BPR) -> HybridTrainer:
    _banner("Hybrid — CF + Content (BPR ranking loss)")
    n_users, n_items = _get_dims()
    tensors = _load_feature_tensors()
    _, _, item_pop = _load_bpr_data()  # need item_pop for negative sampling

    # Reload all_items too
    d         = np.load(BPR_DATA_PATH)
    all_items = d["all_items"]

    hybrid_model = HybridModel(
        n_users=n_users, n_items=n_items,
        embed_dim=EMBED_DIM_D, n_heads=NUM_HEADS,
    )
    log.info(repr(hybrid_model))

    hybrid_model.load_cf_weights(
        uf   = als_model.get_user_factors_tensor(),
        if_  = als_model.get_item_factors_tensor(),
        ubpr = bpr_model.get_user_embeddings_tensor(),
        ibpr = bpr_model.get_item_embeddings_tensor(),
    )

    trainer = HybridTrainer(
        model             = hybrid_model,
        sbert_emb         = tensors["sbert"],
        imdb_feats        = tensors["imdb"],
        popularity        = tensors["pop"],
        history_emb       = tensors["history"],
        all_items         = all_items,
        item_pop          = item_pop,
        device            = DEVICE,
        lr                = LR_HYBRID,
        weight_decay      = HYBRID_WEIGHT_DECAY,
        n_epochs          = HYBRID_EPOCHS,
        batch_size        = HYBRID_BATCH_SIZE,
        samples_per_epoch = HYBRID_SAMPLES_PER_EPOCH,
        patience          = EARLY_STOP_PATIENCE,
        freeze_epochs     = FREEZE_EPOCHS,
    )
    trainer.fit()
    log.info(f"Hybrid done — best val loss: {trainer.best_val_loss:.5f}")
    return trainer


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    t0 = time.time()
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_all    = args.model is None
    als = bpr  = hybrid = None

    # ALS
    if run_all or args.model == "als":
        als = train_als(skip_if_exists=args.skip_als)
    elif ALS_PATH.exists():
        log.info("Loading existing ALS...")
        als = ALS.load(ALS_PATH)
    else:
        if args.model in ("bpr","hybrid"):
            raise FileNotFoundError(f"ALS factors not found. Run --model als first.")

    # BPR
    if run_all or args.model == "bpr":
        bpr = train_bpr(als)
    elif BPR_FACTORS_PATH.exists():
        log.info("Loading existing BPR...")
        bpr = BPR.load(BPR_FACTORS_PATH, device=DEVICE)

    # Hybrid
    if run_all or args.model == "hybrid":
        if als is None or bpr is None:
            raise RuntimeError("Hybrid needs ALS and BPR. Run both first.")
        hybrid = train_hybrid(als, bpr)

    # Summary
    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  TRAINING COMPLETE                                    ║")
    log.info(f"║  Total: {_elapsed(t0):<47}║")
    log.info("╠══════════════════════════════════════════════════════╣")
    if als:
        log.info(f"║  ALS    RMSE : {als.train_rmse_history[-1]:.5f}{'':>38}║")
    if bpr:
        log.info(f"║  BPR    loss : {bpr.train_loss_history[-1]:.5f}{'':>38}║")
    if hybrid:
        log.info(f"║  Hybrid loss : {hybrid.best_val_loss:.5f}{'':>38}║")
    log.info("╠══════════════════════════════════════════════════════╣")
    for label, path in [("ALS", ALS_PATH), ("BPR", BPR_FACTORS_PATH), ("Hybrid", HYBRID_CKPT_PATH)]:
        ok = "✓" if path.exists() else "✗"
        log.info(f"║  [{ok}] {label:<20} {path.name:<30}║")
    log.info("╚══════════════════════════════════════════════════════╝")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["als","bpr","hybrid"], default=None)
    p.add_argument("--skip-als", action="store_true",
                   help="Load existing ALS instead of retraining")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())