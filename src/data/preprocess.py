"""
DataFlix — Preprocessing
src/data/preprocess.py

Takes the parsed MovieLens DataFrame from parse.py and produces:
  1. k-core filtered ratings
  2. Per-user temporal train / val / test split
  3. Compact integer ID mappings
  4. Mean-centred ratings
  5. CSR sparse matrix (for ALS / MF training)
  6. BPR binary data (for ranking model training)
  7. Cold-start user identification

Note on Netflix Prize:
  Netflix was dropped as a cross-domain validation source.
  Title/year alignment achieved only 5.9% movie coverage (1,045 / 17,770
  movies matched), making cross-domain evaluation statistically unreliable.
  All splits are from MovieLens 25M only.
"""

import logging
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from src.config import (
        PROCESSED_DIR,
        TRAIN_CSV, VAL_CSV, TEST_CSV,
        USER_MAP_CSV, MOVIE_MAP_CSV,
        COLD_START_CSV, STATS_JSON,
        CSR_MATRIX_PATH, BPR_DATA_PATH, USER_POSITIVES_PATH,
        MIN_USER_RATINGS, MIN_MOVIE_RATINGS,
        TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
        COLD_START_THRESHOLD,
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
    TEST_RATIO           = 0.1
    COLD_START_THRESHOLD = 15

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── 1. k-core filtering ───────────────────────────────────────────────────────

def kcore_filter(
    df: pd.DataFrame,
    min_user: int = MIN_USER_RATINGS,
    min_movie: int = MIN_MOVIE_RATINGS,
) -> pd.DataFrame:
    """
    Iteratively remove users with < min_user ratings and movies with
    < min_movie ratings until convergence.
    """
    before = len(df)
    iteration = 0
    while True:
        iteration += 1
        n_before = len(df)
        movie_counts = df["movie_id"].value_counts()
        df = df[df["movie_id"].isin(movie_counts[movie_counts >= min_movie].index)]
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_user].index)]
        n_after = len(df)
        log.info(f"  k-core iter {iteration}: {n_before:,} → {n_after:,} ratings")
        if n_after == n_before:
            break
    log.info(f"  k-core complete: dropped {before - len(df):,} ratings "
             f"({(before - len(df))/before*100:.1f}%)")
    return df.reset_index(drop=True)


# ── 2. Per-user temporal split ────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float   = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each user independently, sort by timestamp then assign:
      first train_ratio      → train
      next  val_ratio        → val
      remainder              → test
    """
    log.info("  Running per-user temporal split...")
    df = df.copy()
    df["_row"] = np.arange(len(df))
    df = df.sort_values(["user_id", "timestamp", "_row"]).reset_index(drop=True)

    train_rows, val_rows, test_rows = [], [], []
    for _, group in df.groupby("user_id", sort=False):
        n       = len(group)
        n_train = max(1, int(np.floor(n * train_ratio)))
        n_val   = max(1, int(np.floor(n * val_ratio)))
        n_test  = n - n_train - n_val
        if n_test < 1:
            n_train = max(1, n - 2)
            n_val   = 1
            n_test  = max(0, n - n_train - n_val)
        idxs = group.index.tolist()
        train_rows.extend(idxs[:n_train])
        val_rows.extend(idxs[n_train:n_train + n_val])
        test_rows.extend(idxs[n_train + n_val:])

    train_df = df.loc[train_rows].drop(columns=["_row"]).reset_index(drop=True)
    val_df   = df.loc[val_rows].drop(columns=["_row"]).reset_index(drop=True)
    test_df  = df.loc[test_rows].drop(columns=["_row"]).reset_index(drop=True)

    log.info(f"  Split → train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")
    return train_df, val_df, test_df


# ── 3. ID mappings ────────────────────────────────────────────────────────────

def create_id_mappings(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build compact integer mappings from the union of all three splits.
    """
    all_users  = pd.concat([train_df["user_id"], val_df["user_id"], test_df["user_id"]]).unique()
    all_movies = pd.concat([train_df["movie_id"], val_df["movie_id"], test_df["movie_id"]]).unique()

    user_map  = {uid: idx for idx, uid in enumerate(sorted(all_users))}
    movie_map = {mid: idx for idx, mid in enumerate(sorted(all_movies.astype(int)))}

    def _apply(df):
        df = df.copy()
        df["user_idx"]  = df["user_id"].map(user_map).astype("int32")
        df["movie_idx"] = df["movie_id"].map(movie_map).astype("int32")
        return df

    log.info(f"  ID mapping: {len(user_map):,} users | {len(movie_map):,} movies")
    return user_map, movie_map, _apply(train_df), _apply(val_df), _apply(test_df)


def save_id_mappings(user_map: dict, movie_map: dict) -> None:
    pd.DataFrame(user_map.items(),  columns=["user_id",  "user_idx"]).to_csv(USER_MAP_CSV,  index=False)
    pd.DataFrame(movie_map.items(), columns=["movie_id", "movie_idx"]).to_csv(MOVIE_MAP_CSV, index=False)
    log.info(f"  Saved user_map → {USER_MAP_CSV}")
    log.info(f"  Saved movie_map → {MOVIE_MAP_CSV}")


# ── 4. Mean centering ─────────────────────────────────────────────────────────

def mean_center_ratings(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Subtract each user's mean rating (computed on train only) from all splits.
    Adds column rating_centered. Original rating column is preserved.
    """
    user_means = train_df.groupby("user_id")["rating"].mean().rename("user_mean")

    def _center(df):
        df = df.copy()
        df = df.join(user_means, on="user_id")
        df["user_mean"] = df["user_mean"].fillna(df["rating"].mean())
        df["rating_centered"] = (df["rating"] - df["user_mean"]).astype("float32")
        return df.drop(columns=["user_mean"])

    train_df = _center(train_df)
    val_df   = _center(val_df)
    test_df  = _center(test_df)
    log.info(f"  Mean centering complete | train centered mean: "
             f"{train_df['rating_centered'].mean():.4f} (expect ≈0)")
    return train_df, val_df, test_df


# ── 5. CSR matrix ─────────────────────────────────────────────────────────────

def build_csr_matrix(
    train_df: pd.DataFrame,
    n_users:  int,
    n_movies: int,
) -> sparse.csr_matrix:
    mat = sparse.csr_matrix(
        (train_df["rating_centered"].values.astype("float32"),
         (train_df["user_idx"].values, train_df["movie_idx"].values)),
        shape=(n_users, n_movies),
    )
    log.info(f"  CSR matrix: {n_users:,} × {n_movies:,} | "
             f"nnz={mat.nnz:,} | density={mat.nnz/(n_users*n_movies)*100:.4f}%")
    return mat


# ── 6. BPR data ───────────────────────────────────────────────────────────────

def build_bpr_data(
    train_df: pd.DataFrame,
) -> tuple[dict, np.ndarray, pd.Series]:
    user_positives = defaultdict(set)
    for row in train_df.itertuples(index=False):
        user_positives[int(row.user_idx)].add(int(row.movie_idx))
    item_popularity = train_df["movie_idx"].value_counts().sort_index()
    all_items = item_popularity.index.values.astype("int32")
    log.info(f"  BPR data: {len(user_positives):,} users | {len(all_items):,} items")
    return dict(user_positives), all_items, item_popularity


# ── 7. Cold-start identification ──────────────────────────────────────────────

def identify_cold_start_users(
    train_df:  pd.DataFrame,
    threshold: int = COLD_START_THRESHOLD,
) -> pd.DataFrame:
    counts = train_df.groupby(["user_idx", "user_id"]).size().reset_index(name="n_train_ratings")
    cold   = counts[counts["n_train_ratings"] < threshold].copy()
    log.info(f"  Cold-start users (< {threshold} train ratings): {len(cold):,}")
    return cold


# ── Main pipeline function ────────────────────────────────────────────────────

def run_preprocessing(ratings: pd.DataFrame) -> dict:
    """
    Full preprocessing pipeline from parsed ML ratings to saved artifacts.

    Parameters
    ----------
    ratings : output of parse.load_all_ratings()
    """
    import json
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("\n" + "=" * 55)
    log.info("Step 2 — Preprocessing")
    log.info("=" * 55)

    log.info("\n[2a] k-core filtering...")
    ratings = kcore_filter(ratings)

    log.info("\n[2b] Temporal splitting...")
    train_df, val_df, test_df = temporal_split(ratings)

    log.info("\n[2c] Building ID mappings...")
    user_map, movie_map, train_df, val_df, test_df = create_id_mappings(
        train_df, val_df, test_df
    )
    save_id_mappings(user_map, movie_map)
    n_users  = len(user_map)
    n_movies = len(movie_map)

    log.info("\n[2d] Mean centering...")
    train_df, val_df, test_df = mean_center_ratings(train_df, val_df, test_df)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV,     index=False)
    test_df.to_csv(TEST_CSV,   index=False)
    log.info(f"  Saved train/val/test CSVs → {PROCESSED_DIR}")

    log.info("\n[2e] Building CSR matrix...")
    csr_mat = build_csr_matrix(train_df, n_users, n_movies)
    sparse.save_npz(CSR_MATRIX_PATH, csr_mat)

    log.info("\n[2f] Building BPR data...")
    user_positives, all_items, item_pop = build_bpr_data(train_df)
    np.savez(BPR_DATA_PATH,
             all_items=all_items,
             item_pop_index=item_pop.index.values,
             item_pop_values=item_pop.values)
    with open(USER_POSITIVES_PATH, "wb") as f:
        pickle.dump(user_positives, f)

    log.info("\n[2g] Identifying cold-start users...")
    cold_df = identify_cold_start_users(train_df)
    cold_df.to_csv(COLD_START_CSV, index=False)

    stats = {
        "n_users":       n_users,
        "n_movies":      n_movies,
        "n_train":       len(train_df),
        "n_val":         len(val_df),
        "n_test":        len(test_df),
        "n_cold_start":  len(cold_df),
        "density_pct":   float(csr_mat.nnz / (n_users * n_movies) * 100),
    }
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("\n" + "=" * 55)
    log.info("Preprocessing complete")
    log.info(f"  Users   : {n_users:,}")
    log.info(f"  Movies  : {n_movies:,}")
    log.info(f"  Train   : {len(train_df):,}")
    log.info(f"  Val     : {len(val_df):,}")
    log.info(f"  Test    : {len(test_df):,}")
    log.info(f"  Density : {stats['density_pct']:.4f}%")
    log.info("=" * 55)
    return stats