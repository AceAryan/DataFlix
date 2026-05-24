"""
DataFlix — Parse & Clean Ratings
src/data/parse.py

Loads MovieLens 25M ratings only.

Netflix Prize was evaluated as a cross-domain validation set but was dropped
due to insufficient movie coverage: title/year alignment between Netflix Prize
(17,770 movies) and MovieLens 25M yielded only 1,045 matched movies — 5.9%
coverage. With 94.1% of Netflix ratings mapping to movies outside the shared
item space, Netflix users had too little signal for meaningful evaluation.
All training, validation, and testing is performed on MovieLens 25M.

Output columns:
  user_id   : str   "ML_<userId>"
  movie_id  : int   MovieLens movieId
  rating    : float 0.5–5.0 in 0.5 increments
  timestamp : int   Unix seconds
"""

import logging
from pathlib import Path

import pandas as pd

try:
    from src.config import (
        ML_RATINGS_PATH,
        MIN_USER_RATINGS, MIN_MOVIE_RATINGS,
    )
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parent.parent.parent
    ML_RATINGS_PATH  = _ROOT / "data/raw/ml-25m/ratings.csv"
    MIN_USER_RATINGS = 10
    MIN_MOVIE_RATINGS = 10

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_ml_ratings() -> pd.DataFrame:
    """
    Load MovieLens 25M ratings.csv and standardise columns.

    Raw columns : userId, movieId, rating, timestamp
    Output      : user_id (str), movie_id (int), rating (float), timestamp (int)
    """
    log.info("Loading MovieLens 25M ratings...")
    df = pd.read_csv(
        ML_RATINGS_PATH,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )
    df = df.rename(columns={
        "userId":    "user_id",
        "movieId":   "movie_id",
        "rating":    "rating",
        "timestamp": "timestamp",
    })

    df["user_id"]   = "ML_" + df["user_id"].astype(str)
    df["movie_id"]  = df["movie_id"].astype("int32")
    df["rating"]    = df["rating"].astype("float32")
    df["timestamp"] = df["timestamp"].astype("int64")

    df = _validate(df)
    log.info(f"  {len(df):,} ratings | "
             f"{df['user_id'].nunique():,} users | "
             f"{df['movie_id'].nunique():,} movies")
    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanity checks:
      - Drop nulls in key columns
      - Drop ratings outside [0.5, 5.0]
      - Drop duplicate (user, movie) pairs — keep last (most recent)
    """
    before = len(df)
    df = df.dropna(subset=["user_id", "movie_id", "rating"])
    df = df[(df["rating"] >= 0.5) & (df["rating"] <= 5.0)]
    df = df.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
    dropped = before - len(df)
    if dropped:
        log.warning(f"  Dropped {dropped:,} invalid/duplicate rows")
    return df.reset_index(drop=True)


def load_all_ratings() -> pd.DataFrame:
    """
    Load and return MovieLens 25M ratings.
    Returns a single DataFrame with columns:
      user_id, movie_id, rating, timestamp
    """
    return load_ml_ratings()


if __name__ == "__main__":
    df = load_all_ratings()
    print(df.head())
    print(df.dtypes)