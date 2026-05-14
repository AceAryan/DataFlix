"""
DataFlix — Step 1b: Parse & Clean Ratings
src/data/parse.py

Loads raw ratings from both MovieLens 25M and Netflix Prize CSVs,
standardises column names, types, and prefixes user IDs to prevent
collision between the two user spaces.

Outputs (in-memory only — pipeline assembles final splits downstream):
  ml_ratings  : pd.DataFrame  [user_id, movie_id, rating, timestamp]
  nf_ratings  : pd.DataFrame  [user_id, movie_id, rating, timestamp]

Both DataFrames use:
  user_id   : str   prefixed  "ML_<id>"  or  "NF_<id>"
  movie_id  : int   MovieLens movieId space (Netflix IDs remapped by align.py)
  rating    : float raw 1–5 star rating (mean-centering happens in preprocess.py)
  timestamp : int   Unix seconds (ML already has this; Netflix Date parsed)
"""

import logging
from pathlib import Path

import pandas as pd

try:
    from src.config import (
        ML_RATINGS_PATH, ML_MOVIES_PATH,
        NETFLIX_RATINGS_PATH, NETFLIX_MOVIES_PATH,
        MIN_USER_RATINGS, MIN_MOVIE_RATINGS,
    )
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parent.parent.parent
    ML_RATINGS_PATH      = _ROOT / "data/raw/ml-25m/ml-25m/ratings.csv"
    ML_MOVIES_PATH       = _ROOT / "data/raw/ml-25m/ml-25m/movies.csv"
    NETFLIX_RATINGS_PATH = _ROOT / "data/raw/netflix/Netflix_Dataset_Rating.csv"
    NETFLIX_MOVIES_PATH  = _ROOT / "data/raw/netflix/Netflix_Dataset_Movie.csv"
    MIN_USER_RATINGS     = 20
    MIN_MOVIE_RATINGS    = 10

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── MovieLens ─────────────────────────────────────────────────────────────────

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

    # Prefix to prevent collision with Netflix user IDs
    df["user_id"] = "ML_" + df["user_id"].astype(str)

    # Coerce types
    df["movie_id"]  = df["movie_id"].astype("int32")
    df["rating"]    = df["rating"].astype("float32")
    df["timestamp"] = df["timestamp"].astype("int64")

    df = _validate(df, source="MovieLens")
    log.info(f"  {len(df):,} ML ratings | "
             f"{df['user_id'].nunique():,} users | "
             f"{df['movie_id'].nunique():,} movies")
    return df


# ── Netflix ───────────────────────────────────────────────────────────────────

def load_nf_ratings(nf_to_ml_map: dict) -> pd.DataFrame:
    """
    Load Netflix_Dataset_Rating.csv, remap movie IDs to ML space,
    and drop movies not in the alignment map.

    Parameters
    ----------
    nf_to_ml_map : dict
        {nf_movie_id (int): {"ml_id": int, ...}}
        Output of align.run_alignment().

    Raw columns : User_ID, Rating, Movie_ID
    Output      : user_id (str), movie_id (int), rating (float), timestamp (int)

    Note: Netflix_Dataset_Rating.csv has no date column in this version,
    so timestamp is set to 0. Temporal splitting falls back to rating order
    for Netflix users (preserving row order as a proxy for time).
    """
    log.info("Loading Netflix ratings...")
    df = pd.read_csv(
        NETFLIX_RATINGS_PATH,
        dtype={"User_ID": "int32", "Movie_ID": "int32", "Rating": "float32"},
    )
    df = df.rename(columns={
        "User_ID":  "user_id",
        "Movie_ID": "movie_id",
        "Rating":   "rating",
    })

    # Prefix Netflix user IDs
    df["user_id"] = "NF_" + df["user_id"].astype(str)
    df["rating"]  = df["rating"].astype("float32")

    # No timestamp in this dataset — use 0 as sentinel
    # Downstream splitter will use row order for Netflix users
    df["timestamp"] = 0

    n_before = len(df)

    # Remap Netflix movie IDs → ML movie IDs (drops unaligned movies)
    ml_id_series = df["movie_id"].map(
        {nf_id: v["ml_id"] for nf_id, v in nf_to_ml_map.items()}
    )
    df["movie_id"] = ml_id_series
    df = df.dropna(subset=["movie_id"]).copy()
    df["movie_id"] = df["movie_id"].astype("int32")

    n_after   = len(df)
    n_dropped = n_before - n_after
    log.info(f"  Dropped {n_dropped:,} ratings on unaligned movies "
             f"({n_dropped/n_before*100:.1f}%)")

    df = _validate(df, source="Netflix")
    log.info(f"  {len(df):,} NF ratings | "
             f"{df['user_id'].nunique():,} users | "
             f"{df['movie_id'].nunique():,} movies")
    return df


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Shared sanity checks applied to both sources:
      - Drop rows with null user_id, movie_id, or rating
      - Drop ratings outside [0.5, 5.0]
      - Drop duplicate (user, movie) pairs — keep the most recent (last row)
    """
    before = len(df)

    # Nulls
    df = df.dropna(subset=["user_id", "movie_id", "rating"])

    # Rating range  (ML uses 0.5–5.0 in 0.5 increments; Netflix 1–5)
    df = df[(df["rating"] >= 0.5) & (df["rating"] <= 5.0)]

    # Duplicate interactions — keep last (most recent behaviour)
    df = df.drop_duplicates(subset=["user_id", "movie_id"], keep="last")

    dropped = before - len(df)
    if dropped:
        log.warning(f"  [{source}] Dropped {dropped:,} invalid/duplicate rows")

    return df.reset_index(drop=True)


# ── Public entry point ────────────────────────────────────────────────────────

def load_all_ratings(nf_to_ml_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate both datasets.

    Returns
    -------
    ml_ratings : pd.DataFrame
    nf_ratings : pd.DataFrame
        Both have columns: user_id, movie_id, rating, timestamp
    """
    ml_ratings = load_ml_ratings()
    nf_ratings = load_nf_ratings(nf_to_ml_map)

    # Sanity check: no user ID collision between the two sets
    ml_users = set(ml_ratings["user_id"].unique())
    nf_users = set(nf_ratings["user_id"].unique())
    overlap  = ml_users & nf_users
    if overlap:
        raise ValueError(
            f"User ID collision between ML and NF datasets: {len(overlap)} overlapping IDs. "
            "Check that ML_ / NF_ prefixes are applied correctly."
        )

    log.info(
        f"\nCombined: {len(ml_ratings)+len(nf_ratings):,} ratings | "
        f"{len(ml_users)+len(nf_users):,} users | "
        f"movie space: ML {ml_ratings['movie_id'].nunique():,} | "
        f"NF (mapped) {nf_ratings['movie_id'].nunique():,}"
    )

    return ml_ratings, nf_ratings


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # For quick smoke-testing without running the full pipeline
    import json
    try:
        from src.config import NF_TO_ML_MAP_JSON
    except ModuleNotFoundError:
        NF_TO_ML_MAP_JSON = Path(__file__).resolve().parent.parent.parent \
                            / "data/processed/netflix_to_ml_movie_map.json"

    if not NF_TO_ML_MAP_JSON.exists():
        raise FileNotFoundError(
            f"Alignment map not found at {NF_TO_ML_MAP_JSON}. "
            "Run align.py first."
        )

    with open(NF_TO_ML_MAP_JSON) as f:
        nf_to_ml_map = {int(k): v for k, v in json.load(f).items()}

    ml_ratings, nf_ratings = load_all_ratings(nf_to_ml_map)

    print("\nMovieLens sample:")
    print(ml_ratings.head())
    print("\nNetflix sample:")
    print(nf_ratings.head())
    print("\nDtypes:")
    print(ml_ratings.dtypes)