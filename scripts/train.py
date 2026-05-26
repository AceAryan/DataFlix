"""
DataFlix — Master Training Script
scripts/train.py

Runs all three models in sequence:
  1. ALS  — matrix factorisation baseline (explicit feedback)
  2. BPR  — ranking-optimised model (warm-started from ALS)
  3. Hybrid — CF + content end-to-end model (warm-started from ALS + BPR)

Each model is checkpointed independently so you can re-run individual
models without retraining from scratch.

Usage:
  python scripts/train.py                      # Train all three
  python scripts/train.py --model als          # Train ALS only
  python scripts/train.py --model bpr          # Train BPR only (requires ALS)
  python scripts/train.py --model hybrid       # Train hybrid only (requires ALS + BPR)
  python scripts/train.py --skip-als           # Skip ALS, load existing factors
  python scripts/train.py --hpo                # Run Optuna HPO before training
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pickle
import torch
from scipy import sparse

from src.config import (
    PROCESSED_DIR, RESULTS_DIR, DEVICE,
    TRAIN_CSV, VAL_CSV,
    CSR_MATRIX_PATH, BPR_DATA_PATH, USER_POSITIVES_PATH, BPR_SAMPLES_PER_EPOCH,
    SBERT_EMBEDDINGS_PATH, IMDB_FEATURES_PATH,
    POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH,
    ALS_PATH,
    LATENT_DIM_K, ALS_ITERATIONS, ALS_REG,
    LR_PATH_B, BPR_REG, BPR_EPOCHS, BPR_BATCH_SIZE, BPR_SAMPLES_PER_EPOCH,
    LR_PATH_A, WEIGHT_DECAY, MAX_EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE,
    OPTUNA_N_TRIALS, OPTUNA_LATENT_DIMS, OPTUNA_REG_RANGE,
    OPTUNA_LR_RANGE, OPTUNA_EMBED_DIMS, OPTUNA_HEADS,
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

BPR_FACTORS_PATH   = RESULTS_DIR / "bpr_factors.npz"
HYBRID_CKPT_PATH   = RESULTS_DIR / "hybrid_best.pt"
HPO_RESULTS_PATH   = RESULTS_DIR / "hpo_results.json"


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading train/val splits (MovieLens 25M)...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    log.info(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")
    return train_df, val_df


_csr_cache: sparse.csr_matrix | None = None

def _load_csr() -> sparse.csr_matrix:
    global _csr_cache
    if _csr_cache is None:
        log.info(f"Loading CSR matrix from {CSR_MATRIX_PATH}...")
        _csr_cache = sparse.load_npz(CSR_MATRIX_PATH)
    return _csr_cache


def _load_bpr_data() -> tuple[dict, np.ndarray, np.ndarray]:
    log.info("Loading BPR training data...")
    bpr_data = np.load(BPR_DATA_PATH)
    all_items      = bpr_data["all_items"]
    item_pop_index = bpr_data["item_pop_index"]
    item_pop_vals  = bpr_data["item_pop_values"]

    # Reconstruct popularity array aligned with all_items
    item_pop = np.zeros(len(all_items), dtype=np.float32)
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_items)}
    for idx, val in zip(item_pop_index, item_pop_vals):
        if idx in idx_to_pos:
            item_pop[idx_to_pos[idx]] = val

    with open(USER_POSITIVES_PATH, "rb") as f:
        user_positives = pickle.load(f)

    log.info(f"  {len(user_positives):,} users | {len(all_items):,} items")
    return user_positives, all_items, item_pop


def _load_feature_tensors() -> dict[str, torch.Tensor]:
    log.info("Loading feature tensors...")
    tensors = {
        "sbert":   torch.load(SBERT_EMBEDDINGS_PATH,   weights_only=True),
        "imdb":    torch.load(IMDB_FEATURES_PATH,      weights_only=True),
        "pop":     torch.load(POPULARITY_PATH,          weights_only=True),
        "history": torch.load(HISTORY_EMBEDDINGS_PATH, weights_only=True),
    }
    for name, t in tensors.items():
        log.info(f"  {name:<10}: {tuple(t.shape)}")
    return tensors


def _get_dims(train_df: pd.DataFrame) -> tuple[int, int]:
    """
    Derive n_users and n_items from the CSR matrix shape — the single
    source of truth. Using train_df.max()+1 can give a different count
    if the highest movie_idx in the DataFrame doesn't match the matrix
    dimensions (off-by-one between BPR data and ALS).
    """
    csr = _load_csr()
    n_users, n_items = csr.shape
    return n_users, n_items


# ── Model 1: ALS ──────────────────────────────────────────────────────────────

def train_als(
    skip_if_exists: bool = False,
    n_factors:  int   = LATENT_DIM_K,
    n_iter:     int   = ALS_ITERATIONS,
    reg:        float = ALS_REG,
) -> ALS:
    _banner("ALS — Matrix Factorisation Baseline")

    if skip_if_exists and ALS_PATH.exists():
        log.info(f"ALS checkpoint found at {ALS_PATH} — loading.")
        return ALS.load(ALS_PATH)

    csr = _load_csr()
    model = ALS(n_factors=n_factors, n_iterations=n_iter, reg=reg)
    model.fit(csr)
    model.save(ALS_PATH)

    log.info(f"\nALS complete — final train RMSE: "
             f"{model.train_rmse_history[-1]:.5f}")
    return model


# ── Model 2: BPR ──────────────────────────────────────────────────────────────

def train_bpr(
    als_model:      ALS,
    skip_if_exists: bool  = False,
    n_factors:      int   = LATENT_DIM_K,
    lr:             float = LR_PATH_B,
    reg:            float = BPR_REG,
    n_epochs:       int   = BPR_EPOCHS,
) -> BPR:
    _banner("BPR — Bayesian Personalized Ranking")

    if skip_if_exists and BPR_FACTORS_PATH.exists():
        log.info(f"BPR checkpoint found at {BPR_FACTORS_PATH} — loading.")
        return BPR.load(BPR_FACTORS_PATH, device=DEVICE)

    train_df, _ = _load_splits()
    n_users, n_items = _get_dims(train_df)

    user_positives, all_items, item_pop = _load_bpr_data()

    model = BPR(
        n_users           = n_users,
        n_items           = n_items,
        n_factors         = n_factors,
        lr                = lr,
        reg               = reg,
        n_epochs          = n_epochs,
        batch_size        = BPR_BATCH_SIZE,
        samples_per_epoch = BPR_SAMPLES_PER_EPOCH,
        device            = DEVICE,
    )

    # Warm-start from ALS factors
    model.init_from_als(als_model)
    model.fit(user_positives, all_items, item_pop)
    model.save(BPR_FACTORS_PATH)

    log.info(f"\nBPR complete — final loss: "
             f"{model.train_loss_history[-1]:.5f}")
    return model


# ── Model 3: Hybrid ───────────────────────────────────────────────────────────

def train_hybrid(
    als_model:      ALS,
    bpr_model:      BPR,
    skip_if_exists: bool  = False,
    embed_dim:      int   = None,
    n_heads:        int   = None,
    lr:             float = LR_PATH_B,
    freeze_epochs:  int   = 5,
) -> HybridTrainer:
    _banner("Hybrid — CF + Content Model (BPR ranking loss)")

    if skip_if_exists and HYBRID_CKPT_PATH.exists():
        log.info(f"Hybrid checkpoint found — skipping.")
        return None

    from src.config import EMBED_DIM_D, NUM_HEADS
    embed_dim = embed_dim or EMBED_DIM_D
    n_heads   = n_heads   or NUM_HEADS

    train_df, _  = _load_splits()
    n_users, n_items = _get_dims(train_df)
    tensors = _load_feature_tensors()

    # Load BPR training data for hybrid triplet sampling
    log.info("Loading BPR triplet data for Hybrid...")
    bpr_data = np.load(BPR_DATA_PATH)
    all_items = bpr_data["all_items"]
    item_pop  = bpr_data["item_pop_values"]
    with open(USER_POSITIVES_PATH, "rb") as f:
        user_positives = pickle.load(f)

    hybrid_model = HybridModel(
        n_users   = n_users,
        n_items   = n_items,
        embed_dim = embed_dim,
        n_heads   = n_heads,
    )
    log.info(repr(hybrid_model))

    hybrid_model.load_cf_weights(
        user_factors = als_model.get_user_factors_tensor(),
        item_factors = als_model.get_item_factors_tensor(),
        user_bpr     = bpr_model.get_user_embeddings_tensor(),
        item_bpr     = bpr_model.get_item_embeddings_tensor(),
    )

    trainer = HybridTrainer(
        model             = hybrid_model,
        sbert_emb         = tensors["sbert"],
        imdb_feats        = tensors["imdb"],
        popularity        = tensors["pop"],
        history_emb       = tensors["history"],
        user_positives    = user_positives,
        all_items         = all_items,
        item_pop          = item_pop,
        device            = DEVICE,
        lr                = lr,
        weight_decay      = WEIGHT_DECAY,
        n_epochs          = MAX_EPOCHS,
        batch_size        = BATCH_SIZE,
        samples_per_epoch = BPR_SAMPLES_PER_EPOCH,
        patience          = EARLY_STOP_PATIENCE,
        freeze_epochs     = freeze_epochs,
    )
    trainer.fit()

    log.info(f"\nHybrid complete — best val loss: {trainer.best_val_loss:.5f}")
    return trainer


# ── Optuna HPO ────────────────────────────────────────────────────────────────

def run_hpo(target: str = "hybrid") -> dict:
    """
    Optuna hyperparameter search.
    Searches over latent dim, regularisation, learning rate, embed dim, heads.
    Optimises validation RMSE for hybrid; validation loss for BPR.

    Saves best params to results/hpo_results.json.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Run: pip install optuna")

    _banner(f"Optuna HPO — target: {target}")

    train_df, val_df = _load_splits()
    n_users, n_items = _get_dims(train_df)

    def objective(trial: optuna.Trial) -> float:
        set_seed()

        n_factors = trial.suggest_categorical("n_factors", OPTUNA_LATENT_DIMS)
        reg       = trial.suggest_float("reg", *OPTUNA_REG_RANGE, log=True)
        lr        = trial.suggest_float("lr",  *OPTUNA_LR_RANGE,  log=True)

        if target == "hybrid":
            embed_dim = trial.suggest_categorical("embed_dim", OPTUNA_EMBED_DIMS)
            n_heads   = trial.suggest_categorical("n_heads",   OPTUNA_HEADS)

            # Quick ALS for warm-start (reduced iterations for speed)
            csr   = _load_csr()
            als   = ALS(n_factors=n_factors, n_iterations=5, reg=reg)
            als.fit(csr)

            tensors = _load_feature_tensors()
            model = HybridModel(n_users=n_users, n_items=n_items,
                                n_factors=n_factors, embed_dim=embed_dim,
                                n_heads=n_heads)
            model.load_cf_weights(
                user_factors = als.get_user_factors_tensor(),
                item_factors = als.get_item_factors_tensor(),
                user_bpr     = torch.zeros(n_users, n_factors),
                item_bpr     = torch.zeros(n_items, n_factors),
            )
            trainer = HybridTrainer(
                model       = model,
                sbert_emb   = tensors["sbert"],
                imdb_feats  = tensors["imdb"],
                popularity  = tensors["pop"],
                history_emb = tensors["history"],
                device      = DEVICE,
                lr          = lr,
                n_epochs    = 10,   # Short run for HPO
                patience    = 3,
                freeze_epochs = 2,
            )
            trainer.fit(train_df, val_df)
            return trainer.best_val_rmse

        elif target == "als":
            csr   = _load_csr()
            model = ALS(n_factors=n_factors, n_iterations=10, reg=reg)
            model.fit(csr)
            return model.train_rmse_history[-1]

        elif target == "bpr":
            user_pos, all_items, item_pop = _load_bpr_data()
            model = BPR(n_users=n_users, n_items=n_items,
                        n_factors=n_factors, lr=lr, reg=reg,
                        n_epochs=10, device=DEVICE)
            model.fit(user_pos, all_items, item_pop)
            return model.train_loss_history[-1]

        return float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)

    best = study.best_params
    log.info(f"\nHPO complete — best params: {best}")
    log.info(f"Best value: {study.best_value:.5f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HPO_RESULTS_PATH, "w") as f:
        json.dump({"target": target, "best_params": best,
                   "best_value": study.best_value}, f, indent=2)
    log.info(f"HPO results saved → {HPO_RESULTS_PATH}")

    return best


# ── Utilities ─────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    log.info("")
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


def _elapsed(start: float) -> str:
    s = time.time() - start
    return f"{int(s // 60)}m {s % 60:.1f}s"


def _summary(
    als:    ALS    | None,
    bpr:    BPR    | None,
    hybrid: HybridTrainer | None,
    t_start: float,
) -> None:
    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  TRAINING COMPLETE                                    ║")
    log.info(f"║  Total time: {_elapsed(t_start):<42}║")
    log.info("╠══════════════════════════════════════════════════════╣")
    if als:
        log.info(f"║  ALS    final train RMSE : "
                 f"{als.train_rmse_history[-1]:.5f}{'':>27}║")
    if bpr:
        log.info(f"║  BPR    final loss       : "
                 f"{bpr.train_loss_history[-1]:.5f}{'':>27}║")
    if hybrid:
        log.info(f"║  Hybrid best val loss    : "
                 f"{hybrid.best_val_loss:.5f}{'':>27}║")
    log.info("╠══════════════════════════════════════════════════════╣")
    log.info("║  Saved checkpoints:                                   ║")
    for label, path in [
        ("ALS factors",    ALS_PATH),
        ("BPR factors",    BPR_FACTORS_PATH),
        ("Hybrid model",   HYBRID_CKPT_PATH),
    ]:
        status = "✓" if path.exists() else "✗"
        log.info(f"║  [{status}] {label:<20} {path.name:<28}║")
    log.info("╚══════════════════════════════════════════════════════╝")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    t_start = time.time()
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Optional HPO pass ──
    hpo_params: dict = {}
    if args.hpo:
        hpo_params = run_hpo(target=args.model or "hybrid")

    # Extract HPO-tuned params if available
    n_factors  = hpo_params.get("n_factors",  LATENT_DIM_K)
    reg_als    = hpo_params.get("reg",         ALS_REG)
    lr_bpr     = hpo_params.get("lr",          LR_PATH_B)
    embed_dim  = hpo_params.get("embed_dim",   None)
    n_heads    = hpo_params.get("n_heads",     None)
    lr_hybrid  = hpo_params.get("lr",          LR_PATH_A)

    run_all = args.model is None
    als_model = bpr_model = hybrid_trainer = None

    # ── ALS ──
    if run_all or args.model == "als":
        als_model = train_als(
            skip_if_exists = args.skip_als,
            n_factors      = n_factors,
            reg            = reg_als,
        )
    elif ALS_PATH.exists():
        log.info("Loading existing ALS factors...")
        als_model = ALS.load(ALS_PATH)
    else:
        if args.model in ("bpr", "hybrid"):
            raise FileNotFoundError(
                f"ALS factors not found at {ALS_PATH}. "
                "Run ALS first: python scripts/train.py --model als"
            )

    # ── BPR ──
    if run_all or args.model == "bpr":
        bpr_model = train_bpr(
            als_model      = als_model,
            skip_if_exists = False,
            n_factors      = n_factors,
            lr             = lr_bpr,
        )
    elif BPR_FACTORS_PATH.exists():
        log.info("Loading existing BPR factors...")
        bpr_model = BPR.load(BPR_FACTORS_PATH, device=DEVICE)

    # ── Hybrid ──
    if run_all or args.model == "hybrid":
        if als_model is None or bpr_model is None:
            raise RuntimeError(
                "Hybrid model requires ALS and BPR factors. "
                "Run: python scripts/train.py --model als && python scripts/train.py --model bpr"
            )
        hybrid_trainer = train_hybrid(
            als_model      = als_model,
            bpr_model      = bpr_model,
            skip_if_exists = False,
            embed_dim      = embed_dim,
            n_heads        = n_heads,
            lr             = lr_hybrid,
        )

    _summary(als_model, bpr_model, hybrid_trainer, t_start)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DataFlix model training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py                      Train all three models
  python scripts/train.py --model als          Train ALS only
  python scripts/train.py --model bpr          Train BPR (loads ALS from disk)
  python scripts/train.py --model hybrid       Train hybrid (loads ALS+BPR)
  python scripts/train.py --skip-als           Load existing ALS, train BPR+hybrid
  python scripts/train.py --hpo --model hybrid Run HPO then train hybrid
        """,
    )
    parser.add_argument(
        "--model", choices=["als", "bpr", "hybrid"], default=None,
        help="Train a specific model only (default: train all three)",
    )
    parser.add_argument(
        "--skip-als", action="store_true",
        help="Load existing ALS factors instead of retraining",
    )
    parser.add_argument(
        "--hpo", action="store_true",
        help="Run Optuna HPO before training to find best hyperparameters",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())