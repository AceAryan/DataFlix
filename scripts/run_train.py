"""
DataFlix — Training Script
Train all models: ALS → baselines → ablation → full DataFlix.
Supports --smoke-test for quick validation.
"""

import sys
import argparse
import json
import pickle
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    set_seed, PROCESSED_DIR, RESULTS_DIR, TRAIN_CSV,
    VAL_CSV, TEST_CSV, CSR_MATRIX_PATH, SBERT_EMBEDDINGS_PATH,
    GENRE_TABLE_PATH, USER_FEATURES_PATH,
    POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH,
    LATENT_DIM_K
)
from src.models.als import ALSSolver
from src.models.hybrid import DataFlixModel
from src.training.trainer import PathATrainer, PathBTrainer, RatingDataset, BPRDataset
from src.evaluation.ablation import run_ablation_study, run_cold_start_evaluation


def load_data():
    """Load all preprocessed data."""
    print("Loading preprocessed data...")
    
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)
    
    with open(PROCESSED_DIR / "stats.json") as f:
        stats = json.load(f)
    n_users = stats["n_users"]
    n_movies = stats["n_movies"]
    
    print(f"  Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
    print(f"  Users: {n_users:,}, Movies: {n_movies:,}")
    
    return train, val, test, n_users, n_movies


def load_features():
    """Load all precomputed features."""
    print("Loading features...")
    
    sbert_data = torch.load(SBERT_EMBEDDINGS_PATH, weights_only=False)
    sbert_embeddings = sbert_data["embeddings"]
    
    genre_table = torch.load(GENRE_TABLE_PATH, weights_only=False)
    user_features = torch.load(USER_FEATURES_PATH, weights_only=False)
    popularity = torch.load(POPULARITY_PATH, weights_only=False)
    history_embeddings = torch.load(HISTORY_EMBEDDINGS_PATH, weights_only=False)
    
    print(f"  SBERT: {sbert_embeddings.shape}")
    print(f"  Genres: {genre_table['n_genres']} genres")
    print(f"  User features: {user_features.shape}")
    print(f"  Popularity: {popularity.shape}")
    print(f"  History: {history_embeddings.shape}")
    
    return sbert_embeddings, genre_table, user_features, popularity, history_embeddings




def run_als(n_users, n_movies):
    """Run ALS as initialisation."""
    print("\n" + "=" * 60)
    print("ALS Training")
    print("=" * 60)
    
    csr = sparse.load_npz(CSR_MATRIX_PATH)
    als = ALSSolver(n_users, n_movies, k=LATENT_DIM_K)
    als.fit(csr)
    
    # Save ALS factors
    P, Q, b_u, b_i, mu = als.get_embeddings()
    np.savez(
        RESULTS_DIR / "als_factors.npz",
        P=P, Q=Q, b_u=b_u, b_i=b_i, mu=np.array([mu])
    )
    print(f"ALS factors saved to {RESULTS_DIR / 'als_factors.npz'}")
    
    return als


def train_full_model(train, val, n_users, n_movies,
                     sbert_embeddings, genre_table, user_features,
                     popularity, history_embeddings,
                     als_model=None, max_epochs=100):
    """Train the full DataFlix model (Path A)."""
    print("\n" + "=" * 60)
    print("Full DataFlix Model — Path A (MSE)")
    print("=" * 60)
    
    # Build datasets
    train_dataset = RatingDataset(
        train, sbert_embeddings, popularity,
        user_features, history_embeddings, genre_table
    )
    val_dataset = RatingDataset(
        val, sbert_embeddings, popularity,
        user_features, history_embeddings, genre_table
    )
    
    # Build model
    model = DataFlixModel(n_users, n_movies, path="A")
    
    # Initialise from ALS if available
    if als_model is not None:
        P, Q, _, _, _ = als_model.get_embeddings()
        model.init_from_als(P, Q)
    
    # Train
    trainer = PathATrainer(model, max_epochs=max_epochs)
    history = trainer.train(train_dataset, val_dataset)
    
    # Save
    torch.save(model.state_dict(), RESULTS_DIR / "dataflix_path_a.pt")
    with open(RESULTS_DIR / "dataflix_path_a_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"Model saved to {RESULTS_DIR / 'dataflix_path_a.pt'}")
    return model, history


def train_bpr_model(train, n_users, n_movies,
                    sbert_embeddings, genre_table, user_features,
                    popularity, history_embeddings,
                    max_epochs=50):
    """Train DataFlix model with BPR (Path B)."""
    print("\n" + "=" * 60)
    print("Full DataFlix Model — Path B (BPR)")
    print("=" * 60)
    
    # Load BPR data
    with open(PROCESSED_DIR / "user_positives.pkl", "rb") as f:
        user_positives = pickle.load(f)
    
    bpr_data = np.load(PROCESSED_DIR / "bpr_data.npz")
    all_items = bpr_data["all_items"]
    item_pop = pd.Series(
        bpr_data["item_pop_values"],
        index=bpr_data["item_pop_index"]
    )
    
    # Build BPR dataset
    is_smoke = "smoke_test" in str(PROCESSED_DIR) or max_epochs <= 5
    n_bpr = 10000 if is_smoke else 500000
    bpr_dataset = BPRDataset(
        user_positives, all_items, item_pop.values,
        sbert_embeddings, popularity, user_features,
        history_embeddings, genre_table,
        n_samples_per_epoch=n_bpr
    )
    
    # Build model
    model = DataFlixModel(n_users, n_movies, path="B")
    
    # Train
    trainer = PathBTrainer(model, max_epochs=max_epochs)
    history = trainer.train(bpr_dataset)
    
    # Save
    torch.save(model.state_dict(), RESULTS_DIR / "dataflix_path_b.pt")
    with open(RESULTS_DIR / "dataflix_path_b_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"Model saved to {RESULTS_DIR / 'dataflix_path_b.pt'}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="DataFlix Training")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick 2-epoch test on 1%% sample")
    parser.add_argument("--skip-als", action="store_true",
                        help="Skip ALS training")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation study")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs for training")
    args = parser.parse_args()
    
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train, val, test, n_users, n_movies = load_data()
    sbert_emb, genre_table, user_feats, popularity, history_emb = load_features()

    # In run_train.py, after load_features():
    from src.training.trainer import init_feature_store

    sbert_emb, genre_table, user_feats, popularity, history_emb = load_features()

    # Add this line — moves everything to GPU once before training starts
    init_feature_store(sbert_emb, popularity, user_feats, history_emb, genre_table)
    
    # Smoke test: subsample
    if args.smoke_test:
        print("\n*** SMOKE TEST MODE — 1% sample, 2 epochs ***")
        frac = 0.01
        train = train.sample(frac=frac, random_state=42)
        val = val.sample(frac=frac, random_state=42)
        test = test.sample(frac=frac, random_state=42)
        args.epochs = 2
    
    # ALS — skip in smoke-test mode (full CSR is 25M entries, too slow for quick test)
    als_model = None
    if not args.skip_als and not args.smoke_test:
        als_model = run_als(n_users, n_movies)
    
    # Full model — Path A
    model_a, history_a = train_full_model(
        train, val, n_users, n_movies,
        sbert_emb, genre_table, user_feats,
        popularity, history_emb,
        als_model, max_epochs=args.epochs
    )
    
    # Full model — Path B
    model_b, history_b = train_bpr_model(
        train, n_users, n_movies,
        sbert_emb, genre_table, user_feats,
        popularity, history_emb,
        max_epochs=min(args.epochs, 50)
    )
    
    # Ablation study
    if not args.skip_ablation:
        run_ablation_study(
            train, val, test, n_users, n_movies,
            sbert_emb, popularity, user_feats, history_emb, genre_table,
            max_epochs=min(args.epochs, 30)
        )
    
    # Cold-start evaluation
    cold_users_df = pd.read_csv(PROCESSED_DIR / "cold_start_users.csv")
    cold_users = cold_users_df["user_idx"].tolist()
    
    run_cold_start_evaluation(
        test, train, cold_users, n_movies,
        sbert_emb, popularity, user_feats, history_emb, genre_table,
        full_model=model_a
    )
    
    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
