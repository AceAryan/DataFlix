import sys
import torch
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# Fix import path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.hybrid import DataFlixModel

# ================= PATHS =================
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
MOVIES_CSV = PROCESSED_DIR / "movies_metadata.csv"

SBERT_PATH = PROCESSED_DIR / "sbert_embeddings.pt"
HISTORY_PATH = PROCESSED_DIR / "history_embeddings.pt"
POPULARITY_PATH = PROCESSED_DIR / "popularity.pt"
USER_FEAT_PATH = PROCESSED_DIR / "user_features.pt"
GENRE_PATH = PROCESSED_DIR / "genre_table.pt"

MODEL_PATH = RESULTS_DIR / "dataflix_path_a.pt"
# ========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- LOAD ----------------
@st.cache_resource
def load_all():
    train = pd.read_csv(TRAIN_CSV)
    movies = pd.read_csv(MOVIES_CSV)

    # SBERT
    sbert_data = torch.load(SBERT_PATH)
    sbert = sbert_data["embeddings"].to(DEVICE)

    # History
    history_data = torch.load(HISTORY_PATH)
    history = history_data if isinstance(history_data, torch.Tensor) else history_data["embeddings"]
    history = history.to(DEVICE)

    # Popularity
    pop_data = torch.load(POPULARITY_PATH)
    popularity = pop_data if isinstance(pop_data, torch.Tensor) else pop_data["values"]
    popularity = popularity.to(DEVICE)

    # User features
    user_feat_data = torch.load(USER_FEAT_PATH)
    user_feat = user_feat_data if isinstance(user_feat_data, torch.Tensor) else user_feat_data["features"]
    user_feat = user_feat.to(DEVICE)

    # Genres
    genre_data = torch.load(GENRE_PATH)
    movie_genres = genre_data["movie_genre_ids"]

    n_users = history.shape[0]
    n_items = sbert.shape[0]

    return train, movies, sbert, history, popularity, user_feat, movie_genres, n_users, n_items


@st.cache_resource
def load_model(n_users, n_items):
    model = DataFlixModel(n_users, n_items, path="A").to(DEVICE)

    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return model


# ---------------- UTILS ----------------
def get_watched(train, user_id):
    item_col = "movie_idx" if "movie_idx" in train.columns else "item_idx"
    return set(train[train.user_idx == user_id][item_col].values)


# ---------------- RECOMMEND ----------------
def recommend(user_id, top_k,
              train, model,
              sbert, history, popularity,
              user_feat, movie_genres):

    n_items = sbert.shape[0]

    user_ids = torch.full((n_items,), user_id, dtype=torch.long, device=DEVICE)
    item_ids = torch.arange(n_items, device=DEVICE)

    sbert_batch = sbert
    pop_batch = popularity.unsqueeze(1)
    hist_batch = history[user_id].unsqueeze(0).repeat(n_items, 1)
    user_feat_batch = user_feat[user_id].unsqueeze(0).repeat(n_items, 1)

    # Safe genre handling
    genre_list = [movie_genres.get(i, []) for i in range(n_items)]

    with torch.no_grad():
        model_scores = model(
            user_ids,
            item_ids,
            sbert_batch,
            pop_batch,
            genre_list,
            hist_batch,
            user_feat_batch
        )

    # Content score
    content_scores = torch.matmul(sbert, history[user_id])

    # Normalize
    model_scores = model_scores / (model_scores.norm() + 1e-8)
    content_scores = content_scores / (content_scores.norm() + 1e-8)

    # Hybrid
    scores = 0.6 * model_scores + 0.4 * content_scores

    # Add noise
    scores += 0.02 * torch.randn_like(scores)

    # Remove watched
    watched = get_watched(train, user_id)
    if len(watched) > 0:
        scores[list(watched)] = -1e9

    top_items = torch.topk(scores, top_k).indices.cpu().numpy()

    return top_items


# ---------------- UI ----------------
def main():
    st.title("🎬 DataFlix Recommendation System")

    st.markdown("Hybrid Model: **User-based (ALS) + Content-based (SBERT)**")

    train, movies, sbert, history, pop, user_feat, genres, n_users, n_items = load_all()
    model = load_model(n_users, n_items)

    user_id = st.number_input("Enter User ID", min_value=0, max_value=n_users-1, value=0)
    top_k = st.slider("Number of Recommendations", 5, 20, 10)

    if st.button("Recommend"):
        recs = recommend(
            user_id, top_k,
            train, model,
            sbert, history, pop,
            user_feat, genres
        )

        st.subheader("Top Recommendations")

        title_col = "title" if "title" in movies.columns else movies.columns[1]

        for i, idx in enumerate(recs):
            try:
                title = movies.iloc[idx][title_col]
            except:
                title = f"Movie {idx}"

            st.write(f"**{i+1}. {title}**")


if __name__ == "__main__":
    main()