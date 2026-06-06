"""
DataFlix — Central Training Pipeline
scripts/train.py

Usage:
  python scripts/train.py --models all
  python scripts/train.py --models lightgcn sasrec twotower
  python scripts/train.py --models hybrid
  python scripts/train.py --models easr
"""

import argparse, logging, pickle, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.config import (
    PROCESSED_DIR, RESULTS_DIR, DEVICE,
    TRAIN_CSV, VAL_CSV, CSR_MATRIX_PATH, BPR_DATA_PATH, USER_POSITIVES_PATH,
    SBERT_EMBEDDINGS_PATH, IMDB_FEATURES_PATH, POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH,
    ALS_PATH, BPR_FACTORS_PATH, HYBRID_CKPT_PATH,
    LATENT_DIM_K, ALS_ITERATIONS, ALS_REG, LR_BPR, BPR_REG, BPR_EPOCHS, BPR_BATCH_SIZE, 
    BPR_SAMPLES_PER_EPOCH, LR_HYBRID, HYBRID_WEIGHT_DECAY, HYBRID_EPOCHS,
    HYBRID_BATCH_SIZE, HYBRID_SAMPLES_PER_EPOCH, EARLY_STOP_PATIENCE, FREEZE_EPOCHS, EMBED_DIM_D,
    set_seed,
)

from src.models.als      import ALS
from src.models.bpr      import BPR
from src.models.easr     import EASR                                    # ← EASE^R
from src.models.hybrid   import HybridModel, HybridTrainer
from src.models.twotower import TwoTowerModel, TwoTowerTrainer
from src.models.lightgcn import LightGCN, LightGCNTrainer, LIGHTGCN_CKPT_PATH
# from src.models.sasrec   import SASRec, SASRecTrainer, SASREC_CKPT_PATH

TWOTOWER_CKPT_PATH  = RESULTS_DIR / "twotower_best.pt"
LIGHTGCN_GRAPH_PATH = PROCESSED_DIR / "lightgcn_graph.npz"
SASREC_SEQS_PATH    = PROCESSED_DIR / "sasrec_seqs.pkl"
EASR_PATH           = RESULTS_DIR / "easr.npz"                         # ← EASE^R checkpoint

# ── EASE^R hyper-parameter ────────────────────────────────────────────────────
# λ=350 is a strong default for ML-scale dense CF matrices.
# Sweep [200, 350, 500, 750] if you want to tune — each run is cheap.
EASR_REG = 350.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

def _elapsed(t): s = time.time()-t; return f"{int(s//60)}m {s%60:.1f}s"
def _banner(title): log.info(f"\n{'='*60}\n  {title}\n{'='*60}")
def _get_dims(): return sparse.load_npz(CSR_MATRIX_PATH).shape

def _load_bpr_data():
    d = np.load(BPR_DATA_PATH)
    with open(USER_POSITIVES_PATH, "rb") as f: pos = pickle.load(f)
    return pos, d["all_items"], d["item_pop_values"]

def _load_feature_tensors():
    return {
        "sbert":   torch.load(SBERT_EMBEDDINGS_PATH, weights_only=False),
        "imdb":    torch.load(IMDB_FEATURES_PATH, weights_only=False),
        "pop":     torch.load(POPULARITY_PATH, weights_only=False),
        "history": torch.load(HISTORY_EMBEDDINGS_PATH, weights_only=False),
    }

def main(args):
    t0 = time.time()
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    to_run = set(args.models)
    if "all" in to_run:
        to_run = {"als", "bpr", "easr", "hybrid", "twotower", "lightgcn", "sasrec"}

    als = bpr = hybrid = twotower = lightgcn = sasrec = None

    # --- 1. CF Baselines ---
    if "als" in to_run:
        _banner("ALS — Matrix Factorisation")
        als = ALS(n_factors=LATENT_DIM_K, n_iterations=ALS_ITERATIONS, reg=ALS_REG)
        als.fit(sparse.load_npz(CSR_MATRIX_PATH))
        als.save(ALS_PATH)
    elif "bpr" in to_run or "hybrid" in to_run:
        als = ALS.load(ALS_PATH)

    if "bpr" in to_run:
        _banner("BPR — Bayesian Personalized Ranking")
        n_u, n_i = _get_dims()
        user_pos, all_items, item_pop = _load_bpr_data()
        bpr = BPR(n_users=n_u, n_items=n_i, n_factors=LATENT_DIM_K, lr=LR_BPR, reg=BPR_REG, n_epochs=BPR_EPOCHS, batch_size=BPR_BATCH_SIZE, samples_per_epoch=BPR_SAMPLES_PER_EPOCH, device=DEVICE)
        bpr.init_from_als(als)
        bpr.fit(user_pos, all_items, item_pop)
        bpr.save(BPR_FACTORS_PATH)
    elif "hybrid" in to_run:
        bpr = BPR.load(BPR_FACTORS_PATH, device=DEVICE)

    # --- 1b. EASE^R — Embarrassingly Shallow AutoEncoder ----------------
    if "easr" in to_run:
        _banner("EASE^R — Embarrassingly Shallow AutoEncoder")
        # Loads the same CSR matrix used by ALS — no extra preprocessing needed.
        easr = EASR(reg=EASR_REG)
        easr.fit(sparse.load_npz(CSR_MATRIX_PATH))
        easr.save(EASR_PATH)

    # --- 2. Neural Models ---
    if "hybrid" in to_run or "twotower" in to_run:
        tensors = _load_feature_tensors()

    if "hybrid" in to_run:
        _banner("Hybrid — CF + Content")
        n_u, n_i = _get_dims()
        user_pos, all_items, item_pop = _load_bpr_data()
        m = HybridModel(n_users=n_u, n_items=n_i, embed_dim=EMBED_DIM_D)
        m.load_cf_weights(als.get_user_factors_tensor(), als.get_item_factors_tensor(), bpr.get_user_embeddings_tensor(), bpr.get_item_embeddings_tensor())
        
        hybrid = HybridTrainer(
            model=m, 
            sbert_emb=tensors["sbert"], 
            imdb_feats=tensors["imdb"], 
            popularity=tensors["pop"], 
            history_emb=tensors["history"], 
            all_items=all_items, 
            item_pop=item_pop, 
            device=DEVICE, 
            lr=LR_HYBRID, 
            weight_decay=HYBRID_WEIGHT_DECAY, 
            n_epochs=HYBRID_EPOCHS, 
            batch_size=HYBRID_BATCH_SIZE, 
            samples_per_epoch=HYBRID_SAMPLES_PER_EPOCH, 
            patience=EARLY_STOP_PATIENCE, 
            freeze_epochs=FREEZE_EPOCHS
        )
        hybrid.fit()

    if "twotower" in to_run:
        _banner("Two-Tower — Dual Encoder Model")
        n_u, n_i = _get_dims()
        _, all_items, item_pop = _load_bpr_data()
        m = TwoTowerModel(n_users=n_u, n_items=n_i, embed_dim=EMBED_DIM_D)
        
        twotower = TwoTowerTrainer(
            model=m, 
            sbert_emb=tensors["sbert"], 
            imdb_feats=tensors["imdb"], 
            popularity=tensors["pop"], 
            history_emb=tensors["history"], 
            all_items=all_items, 
            item_pop=item_pop, 
            device=DEVICE, 
            lr=LR_HYBRID, 
            weight_decay=HYBRID_WEIGHT_DECAY, 
            n_epochs=HYBRID_EPOCHS, 
            batch_size=HYBRID_BATCH_SIZE, 
            samples_per_epoch=HYBRID_SAMPLES_PER_EPOCH, 
            patience=EARLY_STOP_PATIENCE
        )
        twotower.fit()

    # --- 3. Advanced Baselines ---
    if "lightgcn" in to_run:
        _banner("LightGCN — Graph Neural Network")
        if not LIGHTGCN_GRAPH_PATH.exists(): raise FileNotFoundError("Graph missing. Run preprocess.py --tasks lightgcn")
        n_u, n_i = _get_dims()
        user_pos, all_items, item_pop = _load_bpr_data()
        shrunk_model = LightGCN(n_users=n_u, n_items=n_i, embed_dim=64, n_layers=2)
        lightgcn = LightGCNTrainer(shrunk_model, LIGHTGCN_GRAPH_PATH, all_items, item_pop)
        lightgcn.fit(user_pos)

    # if "sasrec" in to_run:
    #     _banner("SASRec — Sequential Transformer")
    #     if not SASREC_SEQS_PATH.exists(): raise FileNotFoundError("Seqs missing. Run preprocess.py --tasks sasrec")
    #     _, n_i = _get_dims()
    #     _, all_items, _ = _load_bpr_data()
    #     sasrec = SASRecTrainer(SASRec(n_i), SASREC_SEQS_PATH, all_items)
    #     sasrec.fit()

    log.info(f"\n{'='*60}\n  TRAINING COMPLETE — {_elapsed(t0)}\n{'='*60}")

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["all"],
                   choices=["all", "als", "bpr", "easr", "hybrid", "twotower", "lightgcn", "sasrec"])
    return p.parse_args()

if __name__ == "__main__": main(_parse_args())