"""
DataFlix — Feature Engineering
src/data/features.py

Three feature streams, all keyed on movie_idx:

  Stream A — IMDB structured features (23-dim)
    20-dim genre multi-hot + runtime_norm + avg_vote_norm + log_votes_norm

  Stream B — SBERT semantic embeddings (384-dim)
    TMDB synopsis → all-MiniLM-L6-v2 → L2-normalised vector
    Movies with no synopsis → zero vector fallback

  Stream C — Popularity + user history
    Per-movie log-normalised interaction count
    Per-user weighted mean of SBERT vectors of rated movies
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from src.config import (
        PROCESSED_DIR, ML_LINKS_PATH, ML_MOVIES_PATH,
        IMDB_BASICS_PATH, IMDB_RATINGS_PATH, TMDB_CSV_PATH,
        MOVIE_MAP_CSV, USER_MAP_CSV, TRAIN_CSV,
        IMDB_FEATURES_PATH, SBERT_EMBEDDINGS_PATH,
        GENRE_TABLE_PATH, POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH,
        SBERT_DIM, NUM_GENRES, IMDB_FEAT_DIM,
    )
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR           = _ROOT / "data/processed"
    ML_LINKS_PATH           = _ROOT / "data/raw/ml-32m/links.csv"
    ML_MOVIES_PATH          = _ROOT / "data/raw/ml-32m/movies.csv"
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

log = logging.getLogger(__name__)

GENRE_VOCAB = [
    "Drama","Comedy","Thriller","Action","Romance",
    "Horror","Crime","Documentary","Adventure","Sci-Fi",
    "Mystery","Fantasy","Biography","Animation","Family",
    "History","Music","War","Western","Sport",
]
assert len(GENRE_VOCAB) == NUM_GENRES
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_VOCAB)}


# ── Stream A: IMDB features ───────────────────────────────────────────────────

def build_imdb_features(movie_map: dict) -> torch.Tensor:
    n_movies = len(movie_map)
    feats    = np.zeros((n_movies, IMDB_FEAT_DIM), dtype=np.float32)

    # Build movieId → tconst map via links.csv
    links = pd.read_csv(ML_LINKS_PATH, usecols=["movieId","imdbId"])
    links["tconst"] = links["imdbId"].apply(
        lambda x: f"tt{int(x):07d}" if pd.notna(x) else None
    )
    mid_to_tconst = dict(zip(links["movieId"], links["tconst"]))

    # Load IMDB
    log.info("  Loading IMDB basics...")
    basics = pd.read_csv(
        IMDB_BASICS_PATH, sep="\t", na_values="\\N", low_memory=False,
        usecols=["tconst","titleType","runtimeMinutes","genres"],
    )
    basics = basics[basics["titleType"]=="movie"].copy()
    basics["runtime"] = pd.to_numeric(basics["runtimeMinutes"], errors="coerce")

    log.info("  Loading IMDB ratings...")
    ratings = pd.read_csv(IMDB_RATINGS_PATH, sep="\t", na_values="\\N")
    imdb = basics.merge(ratings, on="tconst", how="left")

    # Normalise
    imdb["runtime_norm"]       = (imdb["runtime"].clip(60,240).fillna(100) - 60) / 180.0
    imdb["avg_vote_norm"]      = (imdb["averageRating"].fillna(imdb["averageRating"].median()) - 1) / 9.0
    log_votes                  = np.log1p(imdb["numVotes"].fillna(0))
    imdb["log_votes_norm"]     = log_votes / log_votes.max()
    imdb["genres_list"]        = imdb["genres"].fillna("").apply(
        lambda s: [g for g in s.split(",") if g in GENRE_TO_IDX]
    )
    tconst_to_row = {row.tconst: row for row in imdb.itertuples(index=False)}

    n_matched = 0
    for movie_id, movie_idx in movie_map.items():
        tconst = mid_to_tconst.get(int(movie_id))
        if not tconst or tconst not in tconst_to_row:
            continue
        row = tconst_to_row[tconst]
        n_matched += 1
        for g in row.genres_list:
            feats[movie_idx, GENRE_TO_IDX[g]] = 1.0
        feats[movie_idx, 20] = row.runtime_norm
        feats[movie_idx, 21] = row.avg_vote_norm
        feats[movie_idx, 22] = row.log_votes_norm

    log.info(f"  IMDB: {n_matched:,}/{n_movies:,} matched ({n_matched/n_movies*100:.1f}%)")
    t = torch.tensor(feats, dtype=torch.float32)
    torch.save(t, IMDB_FEATURES_PATH)
    log.info(f"  Saved IMDB features → {IMDB_FEATURES_PATH}  shape={tuple(t.shape)}")
    return t


def build_genre_table(movie_map: dict) -> torch.Tensor:
    """Primary genre index per movie for embedding table lookup."""
    n_movies     = len(movie_map)
    genre_idx    = np.full(n_movies, NUM_GENRES, dtype=np.int64)
    links        = pd.read_csv(ML_LINKS_PATH, usecols=["movieId","imdbId"])
    links["tconst"] = links["imdbId"].apply(
        lambda x: f"tt{int(x):07d}" if pd.notna(x) else None
    )
    mid_to_tconst = dict(zip(links["movieId"], links["tconst"]))
    basics = pd.read_csv(
        IMDB_BASICS_PATH, sep="\t", na_values="\\N", low_memory=False,
        usecols=["tconst","titleType","genres"],
    )
    basics = basics[basics["titleType"]=="movie"]
    tconst_to_genres = dict(zip(basics["tconst"], basics["genres"].fillna("")))
    for movie_id, movie_idx in movie_map.items():
        tconst = mid_to_tconst.get(int(movie_id))
        if not tconst: continue
        gs = [g for g in tconst_to_genres.get(tconst,"").split(",") if g in GENRE_TO_IDX]
        if gs:
            genre_idx[movie_idx] = GENRE_TO_IDX[gs[0]]
    t = torch.tensor(genre_idx, dtype=torch.long)
    torch.save(t, GENRE_TABLE_PATH)
    log.info(f"  Saved genre table → {GENRE_TABLE_PATH}")
    return t


# ── Stream B: SBERT embeddings ────────────────────────────────────────────────

def build_sbert_embeddings(movie_map: dict) -> torch.Tensor:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("pip install sentence-transformers")

    n_movies = len(movie_map)

    # Build movie_idx → synopsis
    links = pd.read_csv(ML_LINKS_PATH, usecols=["movieId","tmdbId"])
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    mid_to_tmdb = {int(r.movieId): int(r.tmdbId)
                   for r in links.itertuples(index=False) if pd.notna(r.tmdbId)}

    tmdb = pd.read_csv(TMDB_CSV_PATH, low_memory=False, usecols=["id","overview"])
    tmdb["id"] = pd.to_numeric(tmdb["id"], errors="coerce")
    tmdb = tmdb.dropna(subset=["id"])
    tid_to_overview = dict(zip(tmdb["id"].astype(int), tmdb["overview"].fillna("")))

    # Collect (movie_idx, synopsis) pairs that have content
    idx_with_text, texts = [], []
    for movie_id, movie_idx in movie_map.items():
        tid  = mid_to_tmdb.get(int(movie_id))
        text = tid_to_overview.get(tid, "") if tid else ""
        if isinstance(text, str) and text.strip():
            idx_with_text.append(movie_idx)
            texts.append(text)

    n_with = len(texts)
    log.info(f"  TMDB synopses: {n_with:,}/{n_movies:,} "
             f"({n_with/n_movies*100:.1f}%) — {n_movies-n_with:,} zero-vector fallback")

    log.info("  Encoding with SBERT (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs  = model.encode(
        texts, batch_size=512, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    embeddings = np.zeros((n_movies, SBERT_DIM), dtype=np.float32)
    for i, movie_idx in enumerate(idx_with_text):
        embeddings[movie_idx] = vecs[i]

    t = torch.tensor(embeddings, dtype=torch.float32)
    torch.save(t, SBERT_EMBEDDINGS_PATH)
    log.info(f"  Saved SBERT embeddings → {SBERT_EMBEDDINGS_PATH}  shape={tuple(t.shape)}")
    return t


# ── Stream C: Popularity + user history ──────────────────────────────────────

def build_popularity(train_df: pd.DataFrame, n_movies: int) -> torch.Tensor:
    counts = train_df["movie_idx"].value_counts()
    pop    = np.zeros(n_movies, dtype=np.float32)
    for idx, cnt in counts.items():
        pop[int(idx)] = cnt
    log_pop = np.log1p(pop)
    log_pop /= log_pop.max()
    t = torch.tensor(log_pop, dtype=torch.float32)
    torch.save(t, POPULARITY_PATH)
    log.info(f"  Saved popularity → {POPULARITY_PATH}")
    return t


def build_history_embeddings(
    train_df: pd.DataFrame,
    sbert:    torch.Tensor,
    n_users:  int,
) -> torch.Tensor:
    """
    Per-user history embedding = weighted mean of SBERT vectors of rated movies.
    Weights = raw rating (higher-rated movies contribute more).
    L2-normalised per user.
    """
    log.info("  Building user history embeddings...")
    emb_np  = sbert.numpy()
    history = np.zeros((n_users, SBERT_DIM), dtype=np.float32)

    for user_idx, grp in train_df.groupby("user_idx"):
        midxs   = grp["movie_idx"].values.astype(int)
        weights = grp["rating"].values.astype(np.float32)
        weights = weights / weights.sum()
        history[int(user_idx)] = (emb_np[midxs] * weights[:, None]).sum(axis=0)

    norms   = np.linalg.norm(history, axis=1, keepdims=True)
    norms   = np.where(norms == 0, 1.0, norms)
    history = history / norms

    t = torch.tensor(history, dtype=torch.float32)
    torch.save(t, HISTORY_EMBEDDINGS_PATH)
    log.info(f"  Saved history embeddings → {HISTORY_EMBEDDINGS_PATH}  shape={tuple(t.shape)}")
    return t


# ── Main entry point ──────────────────────────────────────────────────────────

def run_feature_engineering(
    movie_map:  dict,
    user_map:   dict,
    train_df:   pd.DataFrame,
    skip_sbert: bool = False,
) -> dict[str, torch.Tensor]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    n_movies = len(movie_map)
    n_users  = len(user_map)

    log.info("\n=== Feature Engineering ===")

    log.info("\n[A] IMDB structured features")
    imdb_t = build_imdb_features(movie_map)
    build_genre_table(movie_map)

    log.info("\n[B] SBERT embeddings")
    if skip_sbert and SBERT_EMBEDDINGS_PATH.exists():
        log.info("  Loading cached SBERT embeddings")
        sbert_t = torch.load(SBERT_EMBEDDINGS_PATH, weights_only=False)
    else:
        sbert_t = build_sbert_embeddings(movie_map)

    log.info("\n[C] Popularity + history")
    pop_t     = build_popularity(train_df, n_movies)
    history_t = build_history_embeddings(train_df, sbert_t, n_users)

    log.info(f"\n=== Features done ===")
    log.info(f"  imdb    : {tuple(imdb_t.shape)}")
    log.info(f"  sbert   : {tuple(sbert_t.shape)}")
    log.info(f"  pop     : {tuple(pop_t.shape)}")
    log.info(f"  history : {tuple(history_t.shape)}")

    return {
        "imdb": imdb_t, "sbert": sbert_t,
        "pop":  pop_t,  "history": history_t,
    }