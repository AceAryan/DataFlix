"""
DataFlix — Preprocessing
src/data/preprocess.py

Steps:
  1. k-core filtering (iterative until convergence)
  2. Per-user temporal train/val/test split (80/10/10)
  3. Compact integer ID mappings
  4. Mean-centred ratings (user mean from train only)
  5. CSR sparse matrix for ALS
  6. BPR positive sets and popularity data
  7. Cold-start user identification
"""

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from src.config import (
        PROCESSED_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV,
        USER_MAP_CSV, MOVIE_MAP_CSV, COLD_START_CSV, STATS_JSON,
        CSR_MATRIX_PATH, BPR_DATA_PATH, USER_POSITIVES_PATH,
        MIN_USER_RATINGS, MIN_MOVIE_RATINGS,
        TRAIN_RATIO, VAL_RATIO, COLD_START_THRESHOLD,
    )
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR        = _ROOT / "data/processed"
    TRAIN_CSV            = PROCESSED_DIR / "train.csv"
    VAL_CSV              = PROCESSED_DIR / "val.csv"
    TEST_CSV             = PROCESSED_DIR / "test.csv"
    USER_MAP_CSV         = PROCESSED_DIR / "user_map.csv"
    MOVIE_MAP_CSV        = PROCESSED_DIR / "movie_map.csv"
    COLD_START_CSV       = PROCESSED_DIR / "cold_start_users.csv"
    STATS_JSON           = PROCESSED_DIR / "stats.json"
    CSR_MATRIX_PATH      = PROCESSED_DIR / "train_csr.npz"
    BPR_DATA_PATH        = PROCESSED_DIR / "bpr_data.npz"
    USER_POSITIVES_PATH  = PROCESSED_DIR / "user_positives.pkl"
    MIN_USER_RATINGS     = 10
    MIN_MOVIE_RATINGS    = 10
    TRAIN_RATIO          = 0.8
    VAL_RATIO            = 0.1
    COLD_START_THRESHOLD = 20

log = logging.getLogger(__name__)


def kcore_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Iterative k-core until convergence."""
    before = len(df)
    it = 0
    while True:
        it += 1
        n0 = len(df)
        mc = df["movie_id"].value_counts()
        df = df[df["movie_id"].isin(mc[mc >= MIN_MOVIE_RATINGS].index)]
        uc = df["user_id"].value_counts()
        df = df[df["user_id"].isin(uc[uc >= MIN_USER_RATINGS].index)]
        log.info(f"  k-core iter {it}: {n0:,} → {len(df):,}")
        if len(df) == n0:
            break
    log.info(f"  k-core done: {before-len(df):,} rows dropped")
    return df.reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user temporal 80/10/10 split."""
    log.info("  Per-user temporal split...")
    df = df.copy()
    df["_row"] = np.arange(len(df))
    df = df.sort_values(["user_id", "timestamp", "_row"]).reset_index(drop=True)

    train_rows, val_rows, test_rows = [], [], []
    for _, grp in df.groupby("user_id", sort=False):
        n       = len(grp)
        n_train = max(1, int(np.floor(n * TRAIN_RATIO)))
        n_val   = max(1, int(np.floor(n * VAL_RATIO)))
        n_test  = n - n_train - n_val
        if n_test < 1:
            n_train = max(1, n - 2)
            n_val   = 1
            n_test  = max(0, n - n_train - n_val)
        idx = grp.index.tolist()
        train_rows.extend(idx[:n_train])
        val_rows.extend(idx[n_train:n_train+n_val])
        test_rows.extend(idx[n_train+n_val:])

    drop = ["_row"]
    tr = df.loc[train_rows].drop(columns=drop).reset_index(drop=True)
    va = df.loc[val_rows].drop(columns=drop).reset_index(drop=True)
    te = df.loc[test_rows].drop(columns=drop).reset_index(drop=True)
    log.info(f"  train={len(tr):,} | val={len(va):,} | test={len(te):,}")
    return tr, va, te


def create_id_mappings(tr, va, te):
    """Build compact integer ID maps from union of all splits."""
    all_users  = pd.concat([tr["user_id"], va["user_id"], te["user_id"]]).unique()
    all_movies = pd.concat([tr["movie_id"], va["movie_id"], te["movie_id"]]).unique()
    user_map  = {u: i for i, u in enumerate(sorted(all_users))}
    movie_map = {int(m): i for i, m in enumerate(sorted(all_movies.astype(int)))}

    def _apply(df):
        df = df.copy()
        df["user_idx"]  = df["user_id"].map(user_map).astype("int32")
        df["movie_idx"] = df["movie_id"].map(movie_map).astype("int32")
        return df

    log.info(f"  {len(user_map):,} users | {len(movie_map):,} movies")
    return user_map, movie_map, _apply(tr), _apply(va), _apply(te)


def mean_center(tr, va, te):
    """Subtract per-user mean (from train) from all splits."""
    means = tr.groupby("user_id")["rating"].mean().rename("um")
    global_mean = float(tr["rating"].mean())

    def _center(df):
        df = df.copy()
        df = df.join(means, on="user_id")
        df["um"] = df["um"].fillna(global_mean)
        df["rating_centered"] = (df["rating"] - df["um"]).astype("float32")
        return df.drop(columns=["um"])

    tr, va, te = _center(tr), _center(va), _center(te)
    log.info(f"  Mean centering done | train centered mean: {tr['rating_centered'].mean():.4f}")
    return tr, va, te


def build_csr(df: pd.DataFrame, n_users: int, n_movies: int) -> sparse.csr_matrix:
    mat = sparse.csr_matrix(
        (df["rating_centered"].values.astype("float32"),
         (df["user_idx"].values, df["movie_idx"].values)),
        shape=(n_users, n_movies),
    )
    log.info(f"  CSR: {n_users:,}×{n_movies:,} | nnz={mat.nnz:,} | "
             f"density={mat.nnz/(n_users*n_movies)*100:.4f}%")
    return mat


def build_bpr_data(df: pd.DataFrame):
    """
    user_positives : {user_idx → set of movie_idx} — ALL rated items
    all_items      : array of all movie_idx
    item_pop       : rating count per item (for popularity-weighted sampling)
    """
    user_pos = defaultdict(set)
    for row in df.itertuples(index=False):
        user_pos[int(row.user_idx)].add(int(row.movie_idx))
    item_pop  = df["movie_idx"].value_counts().sort_index()
    all_items = item_pop.index.values.astype("int32")
    log.info(f"  BPR: {len(user_pos):,} users | {len(all_items):,} items")
    return dict(user_pos), all_items, item_pop


def identify_cold_users(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["user_idx","user_id"]).size().reset_index(name="n_train")
    cold   = counts[counts["n_train"] < COLD_START_THRESHOLD].copy()
    log.info(f"  Cold-start: {len(cold):,} users (< {COLD_START_THRESHOLD} train ratings)")
    return cold


def run_preprocessing(ratings: pd.DataFrame) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log.info("\n=== Preprocessing ===")

    log.info("\n[1] k-core filtering")
    ratings = kcore_filter(ratings)

    log.info("\n[2] Temporal split")
    tr, va, te = temporal_split(ratings)

    log.info("\n[3] ID mappings")
    user_map, movie_map, tr, va, te = create_id_mappings(tr, va, te)
    n_users, n_movies = len(user_map), len(movie_map)
    pd.DataFrame(user_map.items(),  columns=["user_id", "user_idx"]).to_csv(USER_MAP_CSV,  index=False)
    pd.DataFrame(movie_map.items(), columns=["movie_id","movie_idx"]).to_csv(MOVIE_MAP_CSV, index=False)

    log.info("\n[4] Mean centering")
    tr, va, te = mean_center(tr, va, te)
    tr.to_csv(TRAIN_CSV, index=False)
    va.to_csv(VAL_CSV,   index=False)
    te.to_csv(TEST_CSV,  index=False)

    log.info("\n[5] CSR matrix")
    csr = build_csr(tr, n_users, n_movies)
    sparse.save_npz(CSR_MATRIX_PATH, csr)

    log.info("\n[6] BPR data")
    user_pos, all_items, item_pop = build_bpr_data(tr)
    np.savez(BPR_DATA_PATH,
             all_items=all_items,
             item_pop_index=item_pop.index.values,
             item_pop_values=item_pop.values)
    with open(USER_POSITIVES_PATH, "wb") as f:
        pickle.dump(user_pos, f)

    log.info("\n[7] Cold-start users")
    cold = identify_cold_users(tr)
    cold.to_csv(COLD_START_CSV, index=False)

    stats = {
        "n_users": n_users, "n_movies": n_movies,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "n_cold":  len(cold),
        "density": float(csr.nnz / (n_users * n_movies) * 100),
    }
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"\n=== Done: {n_users:,} users | {n_movies:,} movies | "
             f"train={len(tr):,} | density={stats['density']:.4f}% ===")
    return stats