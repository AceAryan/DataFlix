"""
DataFlix — Netflix Dataset Parser
Reads the CSV-format Netflix Prize data from archive/.
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import NETFLIX_ARCHIVE_DIR


def parse_netflix() -> pd.DataFrame:
    """
    Load Netflix ratings from archive/Netflix_Dataset_Rating.csv.

    Returns
    -------
    pd.DataFrame with columns: user_id, movie_id, rating
    """
    ratings_path = NETFLIX_ARCHIVE_DIR / "Netflix_Dataset_Rating.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Netflix ratings CSV not found at {ratings_path}. "
            "Expected: archive/Netflix_Dataset_Rating.csv"
        )

    print(f"  Reading Netflix ratings from {ratings_path} ...")
    df = pd.read_csv(ratings_path, dtype={"User_ID": int, "Movie_ID": int, "Rating": float})

    df = df.rename(columns={
        "User_ID": "user_id",
        "Movie_ID": "movie_id",
        "Rating": "rating",
    })

    # Keep only needed columns
    df = df[["user_id", "movie_id", "rating"]].copy()
    df = df.dropna()

    print(f"  Netflix ratings loaded: {len(df):,} rows, "
          f"{df['user_id'].nunique():,} users, {df['movie_id'].nunique():,} movies.")
    return df


def parse_movie_titles() -> pd.DataFrame:
    """
    Load Netflix movie metadata from archive/Netflix_Dataset_Movie.csv.

    Returns
    -------
    pd.DataFrame with columns: movie_id, year, title
    """
    movies_path = NETFLIX_ARCHIVE_DIR / "Netflix_Dataset_Movie.csv"
    if not movies_path.exists():
        raise FileNotFoundError(f"Netflix movies CSV not found at {movies_path}.")

    df = pd.read_csv(movies_path)
    df = df.rename(columns={"Movie_ID": "movie_id", "Year": "year", "Name": "title"})
    df = df[["movie_id", "year", "title"]].copy()
    return df
