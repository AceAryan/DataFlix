"""
DataFlix — Step 3: Feature Engineering
src/data/features.py

Three parallel feature streams, all keyed on movie_idx:

  Stream A — IMDB structured features
    genre OHE (20-dim) + runtime (1) + avg_vote (1) + log_num_votes (1) = 23-dim
    Source: title.basics.tsv + title.ratings.tsv joined via ML links.csv imdbId

  Stream B — SBERT semantic embeddings
    384-dim dense vector per movie from TMDB synopsis
    Model: sentence-transformers/all-MiniLM-L6-v2
    Fallback: zero vector for movies with no synopsis

  Stream C — Popularity & user history features
    Per-movie global popularity (log interaction count)
    Per-user history embedding (mean of SBERT vectors of rated movies)

All tensors saved to data/processed/ as .pt files.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from src.config import (
        PROCESSED_DIR,
        ML_LINKS_PATH, ML_MOVIES_PATH,
        IMDB_BASICS_PATH, IMDB_RATINGS_PATH,
        TMDB_CSV_PATH,
        MOVIE_MAP_CSV, USER_MAP_CSV,
        TRAIN_CSV,
        IMDB_FEATURES_PATH, SBERT_EMBEDDINGS_PATH,
        GENRE_TABLE_PATH, POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH,
        SBERT_DIM, NUM_GENRES, IMDB_FEAT_DIM,
        DEVICE,
    )
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR           = _ROOT / "data/processed"
    ML_LINKS_PATH           = _ROOT / "data/raw/ml-25m/ml-25m/links.csv"
    ML_MOVIES_PATH          = _ROOT / "data/raw/ml-25m/ml-25m/movies.csv"
    IMDB_BASICS_PATH        = _ROOT / "data/raw/imdb/title.basics.tsv"
    IMDB_RATINGS_PATH       = _ROOT / "data/raw/imdb/title.ratings.tsv"
    TMDB_CSV_PATH           = _ROOT / "data/raw/tmdb/TMDB_movie_dataset_v11.csv"
    MOVIE_MAP_CSV           = PROCESSED_DIR / "movie_map.csv"
    USER_MAP_CSV            = PROCESSED_DIR / "user_map.csv"
    TRAIN_CSV               = PROCESSED_DIR / "train.csv"
    IMDB_FEATURES_PATH      = PROCESSED_DIR / "imdb_features.pt"
    SBERT_EMBEDDINGS_PATH   = PROCESSED_DIR / "sbert_embeddings.pt"
    GENRE_TABLE_PATH        = PROCESSED_DIR / "genre_table.pt"
    POPULARITY_PATH         = PROCESSED_DIR / "popularity.pt"
    HISTORY_EMBEDDINGS_PATH = PROCESSED_DIR / "history_embeddings.pt"
    SBERT_DIM               = 384
    NUM_GENRES              = 20
    IMDB_FEAT_DIM           = 23
    DEVICE                  = torch.device("cpu")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stream A — IMDB structured features
# ─────────────────────────────────────────────────────────────────────────────

# Top-20 genres by frequency in IMDB — fixed vocabulary for OHE consistency
GENRE_VOCAB = [
    "Drama", "Comedy", "Thriller", "Action", "Romance",
    "Horror", "Crime", "Documentary", "Adventure", "Sci-Fi",
    "Mystery", "Fantasy", "Biography", "Animation", "Family",
    "History", "Music", "War", "Western", "Sport",
]
assert len(GENRE_VOCAB) == NUM_GENRES, \
    f"GENRE_VOCAB has {len(GENRE_VOCAB)} entries but NUM_GENRES={NUM_GENRES}"

GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_VOCAB)}


def _load_imdb_enrichment() -> pd.DataFrame:
    """
    Join title.basics + title.ratings and return per-tconst features.

    Returns DataFrame with columns:
      tconst, genres_list, runtime_norm, avg_vote_norm, log_num_votes_norm
    All numeric features are normalised to [0, 1] range.
    """
    log.info("  Loading IMDB basics...")
    basics = pd.read_csv(
        IMDB_BASICS_PATH, sep="\t", na_values="\\N", low_memory=False,
        usecols=["tconst", "titleType", "runtimeMinutes", "genres"],
    )
    basics = basics[basics["titleType"] == "movie"].copy()
    basics["runtime"] = pd.to_numeric(basics["runtimeMinutes"], errors="coerce")
    basics = basics.drop(columns=["titleType", "runtimeMinutes"])

    log.info("  Loading IMDB ratings...")
    ratings = pd.read_csv(
        IMDB_RATINGS_PATH, sep="\t", na_values="\\N",
        usecols=["tconst", "averageRating", "numVotes"],
    )

    df = basics.merge(ratings, on="tconst", how="left")

    # Normalise runtime: clip to [60, 240] minutes then scale to [0, 1]
    df["runtime_norm"] = (
        df["runtime"].clip(60, 240).fillna(100) - 60
    ) / 180.0

    # Normalise avg_vote: [1, 10] → [0, 1]
    df["avg_vote_norm"] = (
        df["averageRating"].fillna(df["averageRating"].median()) - 1
    ) / 9.0

    # Log-normalise num_votes: log1p then scale to [0, 1]
    log_votes = np.log1p(df["numVotes"].fillna(0))
    df["log_num_votes_norm"] = log_votes / log_votes.max()

    # Parse genres string "Drama,Crime,Thriller" into list
    df["genres_list"] = df["genres"].fillna("").apply(
        lambda s: [g for g in s.split(",") if g in GENRE_TO_IDX]
    )

    log.info(f"  IMDB enrichment: {len(df):,} movies")
    return df[["tconst", "genres_list", "runtime_norm", "avg_vote_norm", "log_num_votes_norm"]]


def build_imdb_features(movie_map: dict) -> torch.Tensor:
    """
    Build the IMDB feature matrix: shape (n_movies, IMDB_FEAT_DIM=23).

    Layout per row:
      [0:20]  genre OHE  (multi-hot, a movie can belong to multiple genres)
      [20]    runtime_norm
      [21]    avg_vote_norm
      [22]    log_num_votes_norm

    Movies not in IMDB get a zero vector (handled gracefully by the model
    via a learned fallback embedding).

    Parameters
    ----------
    movie_map : {movie_id (int) → movie_idx (int)}
    """
    n_movies = len(movie_map)
    features = np.zeros((n_movies, IMDB_FEAT_DIM), dtype=np.float32)

    # Load ML links to map movieId → tconst
    links = pd.read_csv(ML_LINKS_PATH, usecols=["movieId", "imdbId"])
    links["tconst"] = links["imdbId"].apply(
        lambda x: f"tt{int(x):07d}" if pd.notna(x) else None
    )
    ml_id_to_tconst = dict(zip(links["movieId"], links["tconst"]))

    imdb_df = _load_imdb_enrichment()
    tconst_to_row = dict(zip(imdb_df["tconst"], imdb_df.itertuples(index=False)))

    n_matched = 0
    for movie_id, movie_idx in movie_map.items():
        tconst = ml_id_to_tconst.get(int(movie_id))
        if tconst is None or tconst not in tconst_to_row:
            continue  # Zero vector fallback

        row = tconst_to_row[tconst]
        n_matched += 1

        # Genre multi-hot
        for genre in row.genres_list:
            if genre in GENRE_TO_IDX:
                features[movie_idx, GENRE_TO_IDX[genre]] = 1.0

        features[movie_idx, 20] = row.runtime_norm
        features[movie_idx, 21] = row.avg_vote_norm
        features[movie_idx, 22] = row.log_num_votes_norm

    log.info(f"  IMDB features: {n_matched:,}/{n_movies:,} movies matched "
             f"({n_matched/n_movies*100:.1f}%)")

    tensor = torch.tensor(features, dtype=torch.float32)
    torch.save(tensor, IMDB_FEATURES_PATH)
    log.info(f"  Saved IMDB features → {IMDB_FEATURES_PATH}  shape={tuple(tensor.shape)}")
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Stream B — SBERT embeddings from TMDB synopses
# ─────────────────────────────────────────────────────────────────────────────

def _load_tmdb_synopses(movie_map: dict) -> dict[int, str]:
    """
    Load TMDB CSV and return {movie_idx → synopsis string}.
    Matches via tmdbId from ML links.csv.
    Movies with no synopsis get an empty string (handled by fallback below).
    """
    log.info("  Loading TMDB synopses...")
    links = pd.read_csv(ML_LINKS_PATH, usecols=["movieId", "tmdbId"])
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    ml_id_to_tmdb = {
        int(row.movieId): int(row.tmdbId)
        for row in links.itertuples(index=False)
        if pd.notna(row.tmdbId)
    }

    tmdb = pd.read_csv(TMDB_CSV_PATH, low_memory=False, usecols=["id", "overview"])
    tmdb["id"] = pd.to_numeric(tmdb["id"], errors="coerce")
    tmdb = tmdb.dropna(subset=["id"])
    tmdb_id_to_overview = dict(zip(tmdb["id"].astype(int), tmdb["overview"].fillna("")))

    synopses: dict[int, str] = {}
    for movie_id, movie_idx in movie_map.items():
        tmdb_id = ml_id_to_tmdb.get(int(movie_id))
        if tmdb_id is None:
            synopses[movie_idx] = ""
            continue
        overview = tmdb_id_to_overview.get(tmdb_id, "")
        synopses[movie_idx] = overview if isinstance(overview, str) else ""

    n_with_synopsis = sum(1 for s in synopses.values() if s.strip())
    log.info(f"  TMDB synopses: {n_with_synopsis:,}/{len(movie_map):,} movies "
             f"have non-empty synopsis ({n_with_synopsis/len(movie_map)*100:.1f}%)")
    return synopses


def build_sbert_embeddings(movie_map: dict) -> torch.Tensor:
    """
    Encode TMDB synopses with SBERT → (n_movies, 384) float32 tensor.

    Movies with no synopsis get a zero vector. The hybrid model treats
    zero vectors as "content unknown" and relies more heavily on CF signal
    for those items.

    Uses batch encoding for efficiency — encoding one at a time on 25k movies
    would take hours; batching reduces this to minutes.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )

    n_movies  = len(movie_map)
    synopses  = _load_tmdb_synopses(movie_map)

    # Build ordered lists to preserve index alignment
    movie_indices = sorted(synopses.keys())
    texts = [synopses[idx] for idx in movie_indices]

    # Identify which movies actually have content
    has_content  = [bool(t.strip()) for t in texts]
    content_idxs = [i for i, h in enumerate(has_content) if h]
    content_texts = [texts[i] for i in content_idxs]

    log.info(f"  Loading SBERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    log.info(f"  Encoding {len(content_texts):,} synopses in batches...")
    embeddings_content = model.encode(
        content_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalise → cosine similarity = dot product
    )

    # Assemble full embedding matrix with zeros for missing synopses
    embeddings = np.zeros((n_movies, SBERT_DIM), dtype=np.float32)
    for sparse_i, movie_i in enumerate(content_idxs):
        movie_idx = movie_indices[movie_i]
        embeddings[movie_idx] = embeddings_content[sparse_i]

    n_zero = n_movies - len(content_idxs)
    log.info(f"  {n_zero:,} movies have zero-vector fallback (no synopsis)")

    tensor = torch.tensor(embeddings, dtype=torch.float32)
    torch.save(tensor, SBERT_EMBEDDINGS_PATH)
    log.info(f"  Saved SBERT embeddings → {SBERT_EMBEDDINGS_PATH}  shape={tuple(tensor.shape)}")
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Stream C — Popularity & user history embeddings
# ─────────────────────────────────────────────────────────────────────────────

def build_popularity_features(
    train_df:  pd.DataFrame,
    n_movies:  int,
) -> torch.Tensor:
    """
    Per-movie log-normalised interaction count from training data.
    Shape: (n_movies,) — used as a scalar feature in the hybrid model.

    Log normalisation prevents mega-popular movies from dominating;
    a movie with 100k ratings vs 10k ratings shouldn't be 10× more important.
    """
    counts = train_df["movie_idx"].value_counts()
    pop = np.zeros(n_movies, dtype=np.float32)
    for movie_idx, count in counts.items():
        pop[int(movie_idx)] = count

    log_pop = np.log1p(pop)
    log_pop /= log_pop.max()  # Scale to [0, 1]

    tensor = torch.tensor(log_pop, dtype=torch.float32)
    torch.save(tensor, POPULARITY_PATH)
    log.info(f"  Saved popularity → {POPULARITY_PATH}  shape={tuple(tensor.shape)}")
    return tensor


def build_user_history_embeddings(
    train_df:       pd.DataFrame,
    sbert_embeddings: torch.Tensor,
    n_users:        int,
) -> torch.Tensor:
    """
    Per-user history embedding: mean of SBERT vectors of all movies the user
    rated in training, weighted by rating (higher-rated movies contribute more).

    Shape: (n_users, SBERT_DIM=384)

    This gives the hybrid model a content-aware user representation that
    complements the collaborative latent factors — especially useful for
    cold-start users who have few interactions.

    Users with all-zero movie embeddings (no synopsis coverage) get a zero
    history vector.
    """
    log.info("  Building user history embeddings...")
    emb_np = sbert_embeddings.numpy()  # (n_movies, 384)

    history = np.zeros((n_users, SBERT_DIM), dtype=np.float32)

    # Group by user for efficiency
    grouped = train_df.groupby("user_idx")
    for user_idx, group in grouped:
        movie_indices = group["movie_idx"].values.astype(int)
        # Use raw ratings (not centered) as weights — range [0.5, 5.0]
        weights = group["rating"].values.astype(np.float32)
        weights = weights / weights.sum()  # Normalise weights

        movie_embs = emb_np[movie_indices]  # (n_rated, 384)
        history[int(user_idx)] = (movie_embs * weights[:, None]).sum(axis=0)

    # L2 normalise non-zero rows
    norms = np.linalg.norm(history, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # Avoid div by zero
    history = history / norms

    tensor = torch.tensor(history, dtype=torch.float32)
    torch.save(tensor, HISTORY_EMBEDDINGS_PATH)
    log.info(f"  Saved history embeddings → {HISTORY_EMBEDDINGS_PATH}  shape={tuple(tensor.shape)}")
    return tensor


def build_genre_table(movie_map: dict) -> torch.Tensor:
    """
    Build integer genre index tensor for embedding table lookup.
    Shape: (n_movies,) — primary genre index per movie (0-indexed).

    Multi-genre movies: primary genre = first genre listed in IMDB.
    Movies with no genre: index = NUM_GENRES (out-of-vocab, gets a learned
    fallback embedding in the model).
    """
    n_movies = len(movie_map)
    genre_indices = np.full(n_movies, NUM_GENRES, dtype=np.int64)  # Default = OOV

    links = pd.read_csv(ML_LINKS_PATH, usecols=["movieId", "imdbId"])
    links["tconst"] = links["imdbId"].apply(
        lambda x: f"tt{int(x):07d}" if pd.notna(x) else None
    )
    ml_id_to_tconst = dict(zip(links["movieId"], links["tconst"]))

    basics = pd.read_csv(
        IMDB_BASICS_PATH, sep="\t", na_values="\\N", low_memory=False,
        usecols=["tconst", "titleType", "genres"],
    )
    basics = basics[basics["titleType"] == "movie"]
    tconst_to_genres = dict(zip(basics["tconst"], basics["genres"].fillna("")))

    for movie_id, movie_idx in movie_map.items():
        tconst = ml_id_to_tconst.get(int(movie_id))
        if tconst is None:
            continue
        genres_str = tconst_to_genres.get(tconst, "")
        genres = [g for g in genres_str.split(",") if g in GENRE_TO_IDX]
        if genres:
            genre_indices[movie_idx] = GENRE_TO_IDX[genres[0]]

    tensor = torch.tensor(genre_indices, dtype=torch.long)
    torch.save(tensor, GENRE_TABLE_PATH)
    log.info(f"  Saved genre table → {GENRE_TABLE_PATH}  shape={tuple(tensor.shape)}")
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(
    movie_map: dict,
    user_map:  dict,
    train_df:  pd.DataFrame,
    skip_sbert: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Run all three feature streams and return a dict of tensors.

    Parameters
    ----------
    movie_map   : {movie_id → movie_idx}
    user_map    : {user_id  → user_idx}
    train_df    : training split DataFrame (with user_idx, movie_idx, rating)
    skip_sbert  : if True, skip SBERT encoding (useful for fast iteration/testing)

    Returns
    -------
    {
        "imdb":    (n_movies, 23),
        "sbert":   (n_movies, 384),
        "pop":     (n_movies,),
        "history": (n_users,  384),
        "genre":   (n_movies,),
    }
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    n_movies = len(movie_map)
    n_users  = len(user_map)

    log.info("\n" + "=" * 55)
    log.info("Step 3 — Feature Engineering")
    log.info("=" * 55)

    log.info("\n[3a] Stream A: IMDB structured features")
    imdb_tensor = build_imdb_features(movie_map)

    log.info("\n[3b] Stream B: SBERT semantic embeddings")
    if skip_sbert and SBERT_EMBEDDINGS_PATH.exists():
        log.info("  Skipping SBERT encoding — loading cached embeddings")
        sbert_tensor = torch.load(SBERT_EMBEDDINGS_PATH, weights_only=True)
    else:
        sbert_tensor = build_sbert_embeddings(movie_map)

    log.info("\n[3c] Stream C: Popularity + user history + genre table")
    pop_tensor     = build_popularity_features(train_df, n_movies)
    history_tensor = build_user_history_embeddings(train_df, sbert_tensor, n_users)
    genre_tensor   = build_genre_table(movie_map)

    log.info("\n" + "=" * 55)
    log.info("Feature engineering complete")
    log.info(f"  imdb_features   : {tuple(imdb_tensor.shape)}")
    log.info(f"  sbert_embeddings: {tuple(sbert_tensor.shape)}")
    log.info(f"  popularity      : {tuple(pop_tensor.shape)}")
    log.info(f"  history_embs    : {tuple(history_tensor.shape)}")
    log.info(f"  genre_table     : {tuple(genre_tensor.shape)}")
    log.info("=" * 55)

    return {
        "imdb":    imdb_tensor,
        "sbert":   sbert_tensor,
        "pop":     pop_tensor,
        "history": history_tensor,
        "genre":   genre_tensor,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run feature engineering")
    parser.add_argument("--skip-sbert", action="store_true",
                        help="Skip SBERT encoding and load cached embeddings if available")
    args = parser.parse_args()

    # Load preprocessed artifacts
    movie_map_df = pd.read_csv(MOVIE_MAP_CSV)
    user_map_df  = pd.read_csv(USER_MAP_CSV)
    train_df     = pd.read_csv(TRAIN_CSV)

    movie_map = dict(zip(movie_map_df["movie_id"], movie_map_df["movie_idx"]))
    user_map  = dict(zip(user_map_df["user_id"],   user_map_df["user_idx"]))

    run_feature_engineering(
        movie_map=movie_map,
        user_map=user_map,
        train_df=train_df,
        skip_sbert=args.skip_sbert,
    )