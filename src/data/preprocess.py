"""
DataFlix — Step 2: Preprocessing
src/data/preprocess.py

Takes the parsed ML and Netflix DataFrames from parse.py and produces:
  1. k-core filtered ratings (iterative, applied per-source then combined)
  2. Per-user temporal train / val / test split
  3. Compact integer ID mappings (user_id str → user_idx int, movie_id int → movie_idx int)
  4. Mean-centred ratings
  5. CSR sparse matrix (for ALS / MF training)
  6. BPR binary data (for ranking model training)
  7. Cold-start user identification

All outputs written to data/processed/.
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
    PROCESSED_DIR       = _ROOT / "data/processed"
    TRAIN_CSV           = PROCESSED_DIR / "train.csv"
    VAL_CSV             = PROCESSED_DIR / "val.csv"
    TEST_CSV            = PROCESSED_DIR / "test.csv"
    USER_MAP_CSV        = PROCESSED_DIR / "user_map.csv"
    MOVIE_MAP_CSV       = PROCESSED_DIR / "movie_map.csv"
    COLD_START_CSV      = PROCESSED_DIR / "cold_start_users.csv"
    STATS_JSON          = PROCESSED_DIR / "stats.json"
    CSR_MATRIX_PATH     = PROCESSED_DIR / "train_csr.npz"
    BPR_DATA_PATH       = PROCESSED_DIR / "bpr_data.npz"
    USER_POSITIVES_PATH = PROCESSED_DIR / "user_positives.pkl"
    MIN_USER_RATINGS    = 20
    MIN_MOVIE_RATINGS   = 10
    TRAIN_RATIO         = 0.8
    VAL_RATIO           = 0.1
    TEST_RATIO          = 0.1
    COLD_START_THRESHOLD = 5

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── 1. k-core filtering ───────────────────────────────────────────────────────

def kcore_filter(
    df: pd.DataFrame,
    min_user: int = MIN_USER_RATINGS,
    min_movie: int = MIN_MOVIE_RATINGS,
    source_label: str = "",
) -> pd.DataFrame:
    """
    Iteratively remove users with < min_user ratings and movies with
    < min_movie ratings until convergence.

    Iterative is important — removing sparse movies can make some users
    drop below the threshold, and vice versa. A single pass misses this.

    Parameters
    ----------
    df          : DataFrame with user_id, movie_id columns
    min_user    : minimum ratings a user must have to be retained
    min_movie   : minimum ratings a movie must have to be retained
    source_label: label for logging (e.g. "MovieLens", "Netflix")
    """
    label  = f"[{source_label}] " if source_label else ""
    before = len(df)
    iteration = 0

    while True:
        iteration += 1
        n_before = len(df)

        # Drop sparse movies
        movie_counts = df["movie_id"].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie].index
        df = df[df["movie_id"].isin(valid_movies)]

        # Drop sparse users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        df = df[df["user_id"].isin(valid_users)]

        n_after = len(df)
        log.info(f"  {label}k-core iter {iteration}: "
                 f"{n_before:,} → {n_after:,} ratings")

        if n_after == n_before:
            break  # Converged

    dropped = before - len(df)
    log.info(f"  {label}k-core complete: dropped {dropped:,} ratings "
             f"({dropped/before*100:.1f}%)")
    return df.reset_index(drop=True)


# ── 2. Per-user temporal split ────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    source_label: str  = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each user independently, sort their ratings by timestamp (then by
    row index as tiebreaker for zero-timestamp Netflix data), then assign:
      - first train_ratio      → train
      - next  val_ratio        → val
      - remainder              → test

    This is correct for recommendation systems — we train on past behaviour
    and evaluate on future behaviour. A global random split would leak future
    ratings into training.

    For Netflix users with timestamp=0, row order (as loaded) is used as
    the time proxy — this is documented and acceptable given the data limitation.
    """
    label = f"[{source_label}] " if source_label else ""
    log.info(f"  {label}Running per-user temporal split...")

    # Sort by user, then timestamp, then original row index as tiebreaker
    df = df.copy()
    df["_row"] = np.arange(len(df))
    df = df.sort_values(["user_id", "timestamp", "_row"]).reset_index(drop=True)

    train_rows, val_rows, test_rows = [], [], []

    for user_id, group in df.groupby("user_id", sort=False):
        n = len(group)
        n_train = max(1, int(np.floor(n * train_ratio)))
        n_val   = max(1, int(np.floor(n * val_ratio)))
        # Test gets the remainder — at least 1 if possible
        n_test  = n - n_train - n_val

        if n_test < 1:
            # Edge case: very few ratings — put last 1 in test, prev 1 in val
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

    log.info(f"  {label}Split → train {len(train_df):,} | "
             f"val {len(val_df):,} | test {len(test_df):,}")

    return train_df, val_df, test_df


# ── 3. ID mappings ────────────────────────────────────────────────────────────

def create_id_mappings(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create compact integer mappings from string user IDs and int movie IDs.
    Mappings are built from the UNION of all three splits so that val/test
    users and movies have valid indices (needed for evaluation).

    Returns
    -------
    user_map  : {user_id_str  → user_idx int}
    movie_map : {movie_id_int → movie_idx int}
    train_df, val_df, test_df with added user_idx and movie_idx columns
    """
    all_users  = pd.concat([train_df["user_id"], val_df["user_id"], test_df["user_id"]]).unique()
    all_movies = pd.concat([train_df["movie_id"], val_df["movie_id"], test_df["movie_id"]]).unique()

    user_map  = {uid: idx for idx, uid in enumerate(sorted(all_users))}
    movie_map = {mid: idx for idx, mid in enumerate(sorted(all_movies.astype(int)))}

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["user_idx"]  = df["user_id"].map(user_map).astype("int32")
        df["movie_idx"] = df["movie_id"].map(movie_map).astype("int32")
        return df

    train_df = _apply(train_df)
    val_df   = _apply(val_df)
    test_df  = _apply(test_df)

    log.info(f"  ID mapping: {len(user_map):,} users | {len(movie_map):,} movies")
    return user_map, movie_map, train_df, val_df, test_df


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
    This removes per-user rating scale bias — someone who always rates 4–5
    and someone who rates 2–3 both have their scale normalised.

    IMPORTANT: mean is computed on train only. Applying the train mean to val
    and test prevents data leakage.

    Adds column `rating_centered` to all three DataFrames.
    Original `rating` column is preserved for interpretability.
    """
    user_means = train_df.groupby("user_id")["rating"].mean().rename("user_mean")

    def _center(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.join(user_means, on="user_id")
        # Users in val/test not in train (shouldn't happen after k-core, but safe)
        df["user_mean"] = df["user_mean"].fillna(df["rating"].mean())
        df["rating_centered"] = (df["rating"] - df["user_mean"]).astype("float32")
        df = df.drop(columns=["user_mean"])
        return df

    train_df = _center(train_df)
    val_df   = _center(val_df)
    test_df  = _center(test_df)

    log.info(f"  Mean centering complete | "
             f"train centered mean: {train_df['rating_centered'].mean():.4f} (expect ≈0)")
    return train_df, val_df, test_df


# ── 5. CSR matrix ─────────────────────────────────────────────────────────────

def build_csr_matrix(
    train_df: pd.DataFrame,
    n_users:  int,
    n_movies: int,
) -> sparse.csr_matrix:
    """
    Build a user × movie CSR sparse matrix from training ratings.
    Values are mean-centred ratings.
    Used directly by ALS and as the interaction matrix for MF models.
    """
    rows = train_df["user_idx"].values
    cols = train_df["movie_idx"].values
    data = train_df["rating_centered"].values.astype("float32")

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_movies))
    log.info(f"  CSR matrix: {n_users:,} × {n_movies:,} | "
             f"nnz={mat.nnz:,} | density={mat.nnz/(n_users*n_movies)*100:.4f}%")
    return mat


# ── 6. BPR data ───────────────────────────────────────────────────────────────

def build_bpr_data(
    train_df: pd.DataFrame,
) -> tuple[dict, np.ndarray, pd.Series]:
    """
    Prepare data structures for Bayesian Personalized Ranking.

    BPR trains on (user, positive_item, negative_item) triples where:
      - positive_item: a movie the user has rated (any rating)
      - negative_item: a movie the user has NOT rated, sampled by popularity

    Returns
    -------
    user_positives : dict {user_idx → set of movie_idx}
        Used during negative sampling to avoid sampling positives as negatives.
    all_items : np.ndarray
        Array of all movie_idx values (population to sample negatives from).
    item_popularity : pd.Series
        movie_idx → count, used for popularity-weighted negative sampling.
        Popular items are more informative negatives.
    """
    user_positives: dict[int, set] = defaultdict(set)
    for row in train_df.itertuples(index=False):
        user_positives[int(row.user_idx)].add(int(row.movie_idx))

    item_popularity = train_df["movie_idx"].value_counts().sort_index()
    all_items = item_popularity.index.values.astype("int32")

    log.info(f"  BPR data: {len(user_positives):,} users with positives | "
             f"{len(all_items):,} unique items")
    return dict(user_positives), all_items, item_popularity


# ── 7. Cold-start identification ──────────────────────────────────────────────

def identify_cold_start_users(
    train_df:  pd.DataFrame,
    threshold: int = COLD_START_THRESHOLD,
) -> pd.DataFrame:
    """
    Identify users with fewer than `threshold` ratings in the training set.
    These are the users for whom collaborative filtering has the least signal.

    Returns DataFrame with columns: user_idx, user_id, n_train_ratings, source
    where source is "ML" or "NF" based on the user_id prefix.
    """
    counts = train_df.groupby(["user_idx", "user_id"]).size().reset_index(name="n_train_ratings")
    cold   = counts[counts["n_train_ratings"] < threshold].copy()
    cold["source"] = cold["user_id"].str[:2]  # "ML" or "NF"

    ml_cold = (cold["source"] == "ML").sum()
    nf_cold = (cold["source"] == "NF").sum()
    log.info(f"  Cold-start users (< {threshold} train ratings): "
             f"{len(cold):,} total | ML: {ml_cold:,} | NF: {nf_cold:,}")
    return cold


# ── Main pipeline function ────────────────────────────────────────────────────

def run_preprocessing(
    ml_ratings: pd.DataFrame,
    nf_ratings: pd.DataFrame,
) -> dict:
    """
    Full preprocessing pipeline from raw parsed ratings to saved artifacts.

    Parameters
    ----------
    ml_ratings : output of parse.load_ml_ratings()
    nf_ratings : output of parse.load_nf_ratings()

    Returns
    -------
    stats : dict  — dataset statistics for stats.json
    """
    import json
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("\n" + "=" * 55)
    log.info("Step 2 — Preprocessing")
    log.info("=" * 55)

    # ── 2a. k-core filter each source independently ──
    log.info("\n[2a] k-core filtering...")
    ml_ratings = kcore_filter(ml_ratings, source_label="MovieLens")
    nf_ratings = kcore_filter(nf_ratings, source_label="Netflix")

    # ── 2b. Per-user temporal split (independently per source) ──
    log.info("\n[2b] Temporal splitting...")
    ml_train, ml_val, ml_test = temporal_split(ml_ratings, source_label="MovieLens")
    nf_train, nf_val, nf_test = temporal_split(nf_ratings, source_label="Netflix")

    # Merge splits across sources
    train_df = pd.concat([ml_train, nf_train], ignore_index=True)
    val_df   = pd.concat([ml_val,   nf_val],   ignore_index=True)
    test_df  = pd.concat([ml_test,  nf_test],  ignore_index=True)

    # ── 2c. ID mappings ──
    log.info("\n[2c] Building ID mappings...")
    user_map, movie_map, train_df, val_df, test_df = create_id_mappings(
        train_df, val_df, test_df
    )
    save_id_mappings(user_map, movie_map)

    n_users  = len(user_map)
    n_movies = len(movie_map)

    # ── 2d. Mean centering ──
    log.info("\n[2d] Mean centering...")
    train_df, val_df, test_df = mean_center_ratings(train_df, val_df, test_df)

    # Save splits
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV,     index=False)
    test_df.to_csv(TEST_CSV,   index=False)
    log.info(f"  Saved train/val/test CSVs → {PROCESSED_DIR}")

    # ── 2e. CSR matrix ──
    log.info("\n[2e] Building CSR matrix...")
    csr_mat = build_csr_matrix(train_df, n_users, n_movies)
    sparse.save_npz(CSR_MATRIX_PATH, csr_mat)
    log.info(f"  Saved CSR → {CSR_MATRIX_PATH}")

    # ── 2f. BPR data ──
    log.info("\n[2f] Building BPR data...")
    user_positives, all_items, item_pop = build_bpr_data(train_df)
    np.savez(
        BPR_DATA_PATH,
        all_items        = all_items,
        item_pop_index   = item_pop.index.values,
        item_pop_values  = item_pop.values,
    )
    with open(USER_POSITIVES_PATH, "wb") as f:
        pickle.dump(user_positives, f)
    log.info(f"  Saved BPR data → {BPR_DATA_PATH}")

    # ── 2g. Cold-start users ──
    log.info("\n[2g] Identifying cold-start users...")
    cold_df = identify_cold_start_users(train_df)
    cold_df.to_csv(COLD_START_CSV, index=False)

    # ── Stats ──
    stats = {
        "n_users":        n_users,
        "n_movies":       n_movies,
        "n_train":        len(train_df),
        "n_val":          len(val_df),
        "n_test":         len(test_df),
        "n_cold_start":   len(cold_df),
        "density_pct":    float(csr_mat.nnz / (n_users * n_movies) * 100),
        "ml_train":       int((train_df["user_id"].str.startswith("ML")).sum()),
        "nf_train":       int((train_df["user_id"].str.startswith("NF")).sum()),
        "ml_val":         int((val_df["user_id"].str.startswith("ML")).sum()),
        "nf_val":         int((val_df["user_id"].str.startswith("NF")).sum()),
        "ml_test":        int((test_df["user_id"].str.startswith("ML")).sum()),
        "nf_test":        int((test_df["user_id"].str.startswith("NF")).sum()),
    }
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("\n" + "=" * 55)
    log.info("Preprocessing complete")
    log.info(f"  Users:   {n_users:,}")
    log.info(f"  Movies:  {n_movies:,}")
    log.info(f"  Train:   {len(train_df):,}")
    log.info(f"  Val:     {len(val_df):,}")
    log.info(f"  Test:    {len(test_df):,}")
    log.info(f"  Density: {stats['density_pct']:.4f}%")
    log.info("=" * 55)

    return stats