"""
DataFlix — Parse & Clean Ratings
src/data/parse.py

Loads MovieLens 32M ratings, standardises columns, validates data.
"""

import logging
from pathlib import Path
import pandas as pd

try:
    from src.config import ML_RATINGS_PATH, MIN_USER_RATINGS, MIN_MOVIE_RATINGS
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parent.parent.parent
    ML_RATINGS_PATH   = _ROOT / "data/raw/ml-32m/ratings.csv"
    MIN_USER_RATINGS  = 10
    MIN_MOVIE_RATINGS = 10

log = logging.getLogger(__name__)


def load_ratings() -> pd.DataFrame:
    """
    Load ML-32M ratings.csv.
    Output columns: user_id (str), movie_id (int), rating (float), timestamp (int)
    """
    log.info("Loading MovieLens 32M ratings...")
    df = pd.read_csv(
        ML_RATINGS_PATH,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )
    df = df.rename(columns={
        "userId": "user_id", "movieId": "movie_id",
        "rating": "rating",  "timestamp": "timestamp",
    })
    df["user_id"]   = "ML_" + df["user_id"].astype(str)
    df["movie_id"]  = df["movie_id"].astype("int32")
    df["rating"]    = df["rating"].astype("float32")
    df["timestamp"] = df["timestamp"].astype("int64")

    # Basic validation
    before = len(df)
    df = df.dropna(subset=["user_id", "movie_id", "rating"])
    df = df[(df["rating"] >= 0.5) & (df["rating"] <= 5.0)]
    df = df.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
    dropped = before - len(df)
    if dropped:
        log.warning(f"  Dropped {dropped:,} invalid/duplicate rows")

    log.info(f"  {len(df):,} ratings | "
             f"{df['user_id'].nunique():,} users | "
             f"{df['movie_id'].nunique():,} movies")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    df = load_ratings()
    print(df.head())
    print(df.dtypes)