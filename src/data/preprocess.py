"""
DataFlix — Data Preprocessing Utilities
Filtering, ID mapping, mean-centering, CSR matrix, BPR data, cold-start users.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    MIN_USER_RATINGS, MIN_MOVIE_RATINGS, COLD_START_THRESHOLD
)


def filter_sparse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iteratively remove users with < MIN_USER_RATINGS and movies with
    < MIN_MOVIE_RATINGS until convergence.
    """
    df = df.copy()
    prev_len = -1
    rounds = 0
    while len(df) != prev_len:
        prev_len = len(df)
        # Filter movies
        movie_counts = df["movie_id"].value_counts()
        valid_movies = movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index
        df = df[df["movie_id"].isin(valid_movies)]
        # Filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index
        df = df[df["user_id"].isin(valid_users)]
        rounds += 1

    print(f"  filter_sparse: {rounds} rounds → "
          f"{len(df):,} ratings, {df['user_id'].nunique():,} users, "
          f"{df['movie_id'].nunique():,} movies")
    return df.reset_index(drop=True)


def create_id_mappings(df: pd.DataFrame):
    """
    Assign contiguous integer indices to user_id and movie_id.

    Returns
    -------
    user_map : dict  str/int → int
    movie_map : dict str/int → int
    df_indexed : pd.DataFrame with added user_idx, movie_idx columns
    """
    unique_users = sorted(df["user_id"].unique())
    unique_movies = sorted(df["movie_id"].unique())

    user_map = {uid: idx for idx, uid in enumerate(unique_users)}
    movie_map = {mid: idx for idx, mid in enumerate(unique_movies)}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_map)
    df["movie_idx"] = df["movie_id"].map(movie_map)

    # Drop rows where mapping failed (shouldn't happen, but safety net)
    df = df.dropna(subset=["user_idx", "movie_idx"])
    df["user_idx"] = df["user_idx"].astype(int)
    df["movie_idx"] = df["movie_idx"].astype(int)

    print(f"  create_id_mappings: {len(user_map):,} users, {len(movie_map):,} movies")
    return user_map, movie_map, df


def mean_center_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract per-user mean rating (computed from this split only).
    Adds 'rating_centered' column and keeps original 'rating' column.
    """
    df = df.copy()
    if "user_idx" in df.columns:
        user_means = df.groupby("user_idx")["rating"].transform("mean")
    else:
        user_means = df.groupby("user_id")["rating"].transform("mean")
    df["rating_centered"] = df["rating"] - user_means
    return df


def build_csr_matrix(df: pd.DataFrame, n_users: int, n_movies: int) -> sparse.csr_matrix:
    """
    Build a sparse CSR rating matrix from a DataFrame with user_idx, movie_idx, rating.
    Uses rating_centered if available, else rating.
    """
    rating_col = "rating_centered" if "rating_centered" in df.columns else "rating"
    row = df["user_idx"].values
    col = df["movie_idx"].values
    data = df[rating_col].values.astype(np.float32)

    mat = sparse.csr_matrix((data, (row, col)), shape=(n_users, n_movies))
    print(f"  CSR matrix: {mat.shape}, {mat.nnz:,} non-zeros, "
          f"density={mat.nnz/(n_users*n_movies)*100:.4f}%")
    return mat


def binarise_for_bpr(df: pd.DataFrame):
    """
    Prepare data for BPR training.

    Returns
    -------
    user_positives : dict  user_idx → set of movie_idx (positive interactions)
    all_items : np.ndarray of all unique movie_idx values
    item_popularity : pd.Series  movie_idx → count (used for popularity-biased neg sampling)
    """
    # Consider rating >= 4 as positive; fall back to all if no positives
    pos_df = df[df["rating"] >= 4]
    if len(pos_df) == 0:
        pos_df = df  # Use all interactions if no explicit positives

    user_positives = (
        pos_df.groupby("user_idx")["movie_idx"]
        .apply(set)
        .to_dict()
    )

    all_items = df["movie_idx"].unique()
    item_popularity = df["movie_idx"].value_counts()

    print(f"  BPR: {len(user_positives):,} users with positives, "
          f"{len(all_items):,} unique items")
    return user_positives, all_items, item_popularity


def identify_cold_start_users(df: pd.DataFrame) -> list:
    """
    Return list of user_idx with fewer than COLD_START_THRESHOLD ratings.
    """
    counts = df.groupby("user_idx")["rating"].count()
    cold = counts[counts < COLD_START_THRESHOLD].index.tolist()
    print(f"  Cold-start users: {len(cold):,} (< {COLD_START_THRESHOLD} ratings)")
    return cold
