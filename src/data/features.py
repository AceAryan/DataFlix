"""
DataFlix — Feature Engineering
Builds all .pt feature tensors needed for model training:
  - sbert_embeddings.pt
  - genre_table.pt
  - popularity.pt
  - user_features.pt
  - history_embeddings.pt

Also builds netflix_to_ml_movie_map.json alignment from Netflix Movie_ID → ML movieId.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    PROCESSED_DIR, ROOT_DIR, SBERT_DIM, NUM_GENRES
)


# ─── Alignment ────────────────────────────────────────────────
def build_netflix_ml_alignment(movie_map: dict) -> dict:
    """
    Match Netflix movies → MovieLens movieId using title+year similarity.
    Saves netflix_to_ml_movie_map.json in PROCESSED_DIR.

    Returns dict: {netflix_movie_id: ml_movie_idx}
    """
    alignment_path = PROCESSED_DIR / "netflix_to_ml_movie_map.json"

    if alignment_path.exists():
        print("  [alignment] Loading existing netflix_to_ml_movie_map.json")
        with open(alignment_path) as f:
            raw = json.load(f)
        return {int(k): int(v) for k, v in raw.items()}

    print("  [alignment] Building Netflix→ML movie alignment ...")

    nf_movies_path = ROOT_DIR / "archive" / "Netflix_Dataset_Movie.csv"
    ml_movies_path = ROOT_DIR / "ml-25m" / "ml-25m" / "movies.csv"

    nf_movies = pd.read_csv(nf_movies_path)
    nf_movies = nf_movies.rename(columns={"Movie_ID": "movie_id", "Year": "year", "Name": "title"})
    nf_movies["title_clean"] = nf_movies["title"].str.lower().str.strip()
    nf_movies["year"] = pd.to_numeric(nf_movies["year"], errors="coerce").fillna(0).astype(int)

    ml_movies = pd.read_csv(ml_movies_path)
    # ML title format: "Title (year)"
    ml_movies["year"] = ml_movies["title"].str.extract(r"\((\d{4})\)$")[0]
    ml_movies["year"] = pd.to_numeric(ml_movies["year"], errors="coerce").fillna(0).astype(int)
    ml_movies["title_clean"] = (
        ml_movies["title"]
        .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        .str.lower()
        .str.strip()
    )
    # Build a lookup: (title_clean, year) → movieId
    ml_lookup = {}
    for _, row in ml_movies.iterrows():
        key = (row["title_clean"], row["year"])
        ml_lookup[key] = row["movieId"]
    # Also lookup by title alone (year=0 fallback)
    ml_title_lookup = {}
    for _, row in ml_movies.iterrows():
        ml_title_lookup[row["title_clean"]] = row["movieId"]

    nf_to_ml = {}
    for _, row in nf_movies.iterrows():
        key = (row["title_clean"], row["year"])
        if key in ml_lookup:
            nf_to_ml[int(row["movie_id"])] = int(ml_lookup[key])
        elif row["title_clean"] in ml_title_lookup:
            nf_to_ml[int(row["movie_id"])] = int(ml_title_lookup[row["title_clean"]])

    print(f"  [alignment] Matched {len(nf_to_ml):,} / {len(nf_movies):,} Netflix movies")

    # Save as JSON (netflix_movie_id → ml_movieId)
    with open(alignment_path, "w") as f:
        json.dump({str(k): v for k, v in nf_to_ml.items()}, f)

    # Convert ml_movieId → movie_idx using movie_map
    aligned = {}
    for nf_id, ml_id in nf_to_ml.items():
        if ml_id in movie_map:
            aligned[nf_id] = movie_map[ml_id]

    return aligned


# ─── Synopses from TMDB ───────────────────────────────────────
def load_synopses_from_tmdb(movie_map: dict) -> dict:
    """
    Load movie synopses from the local TMDB CSV, matched to movie_map.

    Returns dict: {movie_idx: synopsis_string}
    """
    tmdb_path = ROOT_DIR / "tmdb" / "TMDB_movie_dataset_v11.csv"
    ml_movies_path = ROOT_DIR / "ml-25m" / "ml-25m" / "movies.csv"
    links_path = ROOT_DIR / "ml-25m" / "ml-25m" / "links.csv"

    if not tmdb_path.exists():
        print("  [synopses] TMDB CSV not found, using empty synopses.")
        return {}

    print("  [synopses] Loading TMDB synopses ...")
    tmdb = pd.read_csv(tmdb_path, usecols=["id", "title", "overview"])
    tmdb = tmdb.dropna(subset=["overview"])
    tmdb["overview"] = tmdb["overview"].astype(str)
    tmdb["id"] = pd.to_numeric(tmdb["id"], errors="coerce")

    # Try to match via links.csv (movieId → tmdbId)
    synopses = {}
    if links_path.exists():
        links = pd.read_csv(links_path)
        links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
        tmdb_id_map = dict(zip(tmdb["id"], tmdb["overview"]))
        for _, row in links.iterrows():
            ml_id = int(row["movieId"])
            if ml_id in movie_map and not pd.isna(row["tmdbId"]):
                tmdb_id = int(row["tmdbId"])
                if tmdb_id in tmdb_id_map:
                    synopses[movie_map[ml_id]] = tmdb_id_map[tmdb_id]

    if not synopses:
        # Fallback: match by title
        ml_movies = pd.read_csv(ml_movies_path)
        ml_movies["title_clean"] = (
            ml_movies["title"]
            .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
            .str.lower().str.strip()
        )
        tmdb["title_clean"] = tmdb["title"].str.lower().str.strip()
        tmdb_title_map = dict(zip(tmdb["title_clean"], tmdb["overview"]))
        for _, row in ml_movies.iterrows():
            ml_id = int(row["movieId"])
            if ml_id in movie_map:
                synopsis = tmdb_title_map.get(row["title_clean"], "")
                if synopsis:
                    synopses[movie_map[ml_id]] = synopsis

    print(f"  [synopses] Found synopses for {len(synopses):,} / {len(movie_map):,} movies")
    return synopses


# ─── Main Feature Engineering ─────────────────────────────────
def run_feature_engineering(
    train_df: pd.DataFrame,
    ml_movies: pd.DataFrame,
    n_users: int,
    n_movies: int,
    synopses: dict,
):
    """
    Build and save all feature tensors to PROCESSED_DIR.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. SBERT Embeddings ──
    _build_sbert_embeddings(n_movies, synopses)

    # ── 2. Genre Table ──
    _build_genre_table(ml_movies, n_movies)

    # ── 3. Popularity ──
    _build_popularity(train_df, n_movies)

    # ── 4. User Features ──
    _build_user_features(train_df, n_users)

    # ── 5. History Embeddings ──
    sbert_data = torch.load(PROCESSED_DIR / "sbert_embeddings.pt", weights_only=False)
    sbert_emb = sbert_data["embeddings"]  # (n_movies, SBERT_DIM)
    _build_history_embeddings(train_df, n_users, n_movies, sbert_emb)

    print("  [features] All feature tensors saved.")


def _build_sbert_embeddings(n_movies: int, synopses: dict):
    out_path = PROCESSED_DIR / "sbert_embeddings.pt"
    if out_path.exists():
        print("  [sbert] Already exists, skipping.")
        return

    print("  [sbert] Encoding synopses with sentence-transformers ...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        texts = []
        idx_list = []
        for idx in range(n_movies):
            text = synopses.get(idx, "")
            texts.append(text if text else "unknown movie")
            idx_list.append(idx)

        # Encode in batches
        batch_size = 256
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            emb = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            all_embeds.append(emb.cpu())

        embeddings = torch.cat(all_embeds, dim=0)  # (n_movies, 384)
    except Exception as e:
        print(f"  [sbert] Warning: {e}. Falling back to random embeddings.")
        embeddings = torch.randn(n_movies, SBERT_DIM)

    torch.save({"embeddings": embeddings}, out_path)
    print(f"  [sbert] Saved: {embeddings.shape}")


def _build_genre_table(ml_movies: pd.DataFrame, n_movies: int):
    out_path = PROCESSED_DIR / "genre_table.pt"
    if out_path.exists():
        print("  [genres] Already exists, skipping.")
        return

    print("  [genres] Building genre table ...")
    if "genres" not in ml_movies.columns:
        genre_tensor = torch.zeros(n_movies, NUM_GENRES)
        torch.save({
            "genre_table": genre_tensor,
            "n_genres": NUM_GENRES,
            "genre_names": [],
            "movie_genre_ids": {},
        }, out_path)
        return

    # Collect all genres
    all_genres = set()
    for genres_str in ml_movies["genres"].dropna():
        for g in genres_str.split("|"):
            all_genres.add(g.strip())
    all_genres.discard("(no genres listed)")
    all_genres = sorted(all_genres)
    n_genres = min(len(all_genres), NUM_GENRES)
    genre_to_idx = {g: i for i, g in enumerate(all_genres) if i < n_genres}

    # Build genre matrix and per-movie genre ID list
    genre_table = torch.zeros(n_movies, n_genres)
    movie_genre_ids: dict = {}   # movie_idx -> [genre_id, ...]

    for _, row in ml_movies.iterrows():
        idx = row.get("movie_idx", -1)
        if pd.isna(idx) or int(idx) >= n_movies:
            continue
        idx = int(idx)
        if pd.isna(row.get("genres", "")):
            continue
        gids = []
        for g in str(row["genres"]).split("|"):
            g = g.strip()
            if g in genre_to_idx:
                gid = genre_to_idx[g]
                genre_table[idx, gid] = 1.0
                gids.append(gid)
        if gids:
            movie_genre_ids[idx] = gids

    torch.save({
        "genre_table": genre_table,
        "n_genres": n_genres,
        "genre_names": all_genres[:n_genres],
        "movie_genre_ids": movie_genre_ids,
    }, out_path)
    print(f"  [genres] Saved: {genre_table.shape}, {n_genres} genres, "
          f"{len(movie_genre_ids):,} movies with genre IDs")


def _build_popularity(train_df: pd.DataFrame, n_movies: int):
    out_path = PROCESSED_DIR / "popularity.pt"
    if out_path.exists():
        print("  [popularity] Already exists, skipping.")
        return

    print("  [popularity] Computing item popularity ...")
    counts = train_df["movie_idx"].value_counts()
    pop = torch.zeros(n_movies)
    for idx, cnt in counts.items():
        if int(idx) < n_movies:
            pop[int(idx)] = float(cnt)

    # Normalise to [0, 1]
    max_pop = pop.max()
    if max_pop > 0:
        pop = pop / max_pop

    torch.save(pop, out_path)
    print(f"  [popularity] Saved: {pop.shape}")


def _build_user_features(train_df: pd.DataFrame, n_users: int):
    out_path = PROCESSED_DIR / "user_features.pt"
    if out_path.exists():
        print("  [user_features] Already exists, skipping.")
        return

    print("  [user_features] Computing user features ...")
    # Features: mean_rating, rating_count (normalised)
    user_stats = train_df.groupby("user_idx")["rating"].agg(["mean", "count"])
    mean_ratings = torch.zeros(n_users)
    rating_counts = torch.zeros(n_users)

    for uid, row in user_stats.iterrows():
        if int(uid) < n_users:
            mean_ratings[int(uid)] = float(row["mean"])
            rating_counts[int(uid)] = float(row["count"])

    # Normalize counts
    max_count = rating_counts.max()
    if max_count > 0:
        rating_counts = rating_counts / max_count

    # Shape: (n_users, 2)
    user_features = torch.stack([mean_ratings, rating_counts], dim=1)
    torch.save(user_features, out_path)
    print(f"  [user_features] Saved: {user_features.shape}")


def _build_history_embeddings(
    train_df: pd.DataFrame, n_users: int, n_movies: int, sbert_emb: torch.Tensor
):
    out_path = PROCESSED_DIR / "history_embeddings.pt"
    if out_path.exists():
        print("  [history] Already exists, skipping.")
        return

    print("  [history] Building user history embeddings ...")
    embed_dim = sbert_emb.shape[1]
    history_emb = torch.zeros(n_users, embed_dim)

    user_groups = train_df.groupby("user_idx")["movie_idx"].apply(list)
    for uid, movie_indices in user_groups.items():
        uid = int(uid)
        if uid >= n_users:
            continue
        valid = [int(m) for m in movie_indices if int(m) < n_movies]
        if valid:
            history_emb[uid] = sbert_emb[valid].mean(dim=0)

    torch.save(history_emb, out_path)
    print(f"  [history] Saved: {history_emb.shape}")
