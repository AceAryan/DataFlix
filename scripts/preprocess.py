"""
DataFlix — Preprocessing Pipeline
scripts/run_preprocessing.py

Usage:
  python scripts/run_preprocessing.py --tasks all
  python scripts/run_preprocessing.py --tasks core features
  python scripts/run_preprocessing.py --tasks lightgcn sasrec
"""

import argparse, json, logging, sys, time, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.config import (
    PROCESSED_DIR, STATS_JSON,
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    MOVIE_MAP_CSV, USER_MAP_CSV,
    CSR_MATRIX_PATH, SBERT_EMBEDDINGS_PATH, IMDB_FEATURES_PATH,
    set_seed,
)
from src.data.parse      import load_ratings
from src.data.preprocess import run_preprocessing
from src.data.features   import run_feature_engineering

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# New Paths for advanced models
LIGHTGCN_GRAPH_PATH = PROCESSED_DIR / "lightgcn_graph.npz"
SASREC_SEQS_PATH    = PROCESSED_DIR / "sasrec_seqs.pkl"

def _elapsed(t): s = time.time()-t; return f"{int(s//60)}m {s%60:.1f}s"
def _banner(title):
    log.info("")
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)

def build_lightgcn_graph(n_users, n_items):
    _banner("Task: LightGCN Graph Generation")
    t = time.time()
    train_df = pd.read_csv(TRAIN_CSV)
    
    users = train_df["user_idx"].values
    items = train_df["movie_idx"].values + n_users
    
    row  = np.concatenate([users, items])
    col  = np.concatenate([items, users])
    data = np.ones_like(row, dtype=np.float32)
    
    adj = sp.coo_matrix((data, (row, col)), shape=(n_users + n_items, n_users + n_items))
    rowsum = np.array(adj.sum(1)).flatten()
    
    d_inv = np.power(rowsum, -0.5)
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    
    norm_adj = d_mat.dot(adj).dot(d_mat).tocoo()
    sp.save_npz(LIGHTGCN_GRAPH_PATH, norm_adj)
    log.info(f"  Saved → {LIGHTGCN_GRAPH_PATH.name} ({_elapsed(t)})")

def build_sasrec_sequences():
    _banner("Task: SASRec Sequence Generation")
    t = time.time()
    train_df = pd.read_csv(TRAIN_CSV)
    
    if "timestamp" not in train_df.columns:
        log.warning("  No 'timestamp' found. Grouping by raw occurrence (fallback).")
    else:
        train_df = train_df.sort_values(["user_idx", "timestamp"])
        
    user_sequences = train_df.groupby("user_idx")["movie_idx"].apply(list).to_dict()
    valid_sequences = {u: seq for u, seq in user_sequences.items() if len(seq) >= 3}
    
    with open(SASREC_SEQS_PATH, "wb") as f:
        pickle.dump(valid_sequences, f)
    log.info(f"  Generated {len(valid_sequences):,} sequences. Saved → {SASREC_SEQS_PATH.name} ({_elapsed(t)})")

def run_pipeline(tasks: list, skip_sbert: bool = False) -> None:
    t0 = time.time()
    set_seed()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    run_all = "all" in tasks

    # 1. Core Preprocessing (Parse + CSR + BPR generation)
    stats = {}
    if run_all or "core" in tasks:
        _banner("STEP 1 & 2 — Parse & Preprocess Core Data")
        t = time.time()
        ratings = load_ratings()
        stats = run_preprocessing(ratings)
        log.info(f"  Done ({_elapsed(t)})")
    else:
        if STATS_JSON.exists():
            with open(STATS_JSON) as f: stats = json.load(f)

    # 2. Features (SBERT, IMDB)
    if run_all or "features" in tasks:
        _banner("STEP 3 — Feature Engineering")
        t = time.time()
        movie_map_df = pd.read_csv(MOVIE_MAP_CSV)
        user_map_df  = pd.read_csv(USER_MAP_CSV)
        train_df     = pd.read_csv(TRAIN_CSV)
        run_feature_engineering(
            movie_map=dict(zip(movie_map_df["movie_id"], movie_map_df["movie_idx"])), 
            user_map=dict(zip(user_map_df["user_id"], user_map_df["user_idx"])),
            train_df=train_df, skip_sbert=skip_sbert,
        )
        log.info(f"  Done ({_elapsed(t)})")

    # 3. LightGCN
    if run_all or "lightgcn" in tasks:
        if not stats: raise RuntimeError("Missing stats.json to get dimensions. Run 'core' first.")
        build_lightgcn_graph(stats['n_users'], stats['n_movies'])

    # 4. SASRec
    if run_all or "sasrec" in tasks:
        build_sasrec_sequences()

    log.info(f"\n  Pipeline Complete! Total time: {_elapsed(t0)}")

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["all"], choices=["all", "core", "features", "lightgcn", "sasrec"])
    p.add_argument("--skip-sbert", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    run_pipeline(**vars(_parse_args()))