"""
DataFlix — Evaluation Harness
Run evaluation on any model, output results.
"""

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DEVICE, RESULTS_DIR, TOP_K_VALUES
from src.evaluation.metrics import (
    compute_rating_metrics, compute_ranking_metrics
)


def evaluate_rating_model(model, test_df: pd.DataFrame,
                          sbert_embeddings=None, popularity=None,
                          user_features=None, history_embeddings=None,
                          genre_table=None, is_torch=True,
                          batch_size=8192) -> Dict[str, float]:
    """
    Evaluate a model on rating prediction (Path A).
    
    Supports both simple models (VanillaMF, NeuMF with just user/item)
    and hybrid models (DataFlix with all features).
    """
    users = test_df["user_idx"].values
    items = test_df["movie_idx"].values
    ratings = test_df["rating"].values
    
    if is_torch:
        model.eval()
        preds = []
        
        with torch.no_grad():
            for start in range(0, len(users), batch_size):
                end = min(start + batch_size, len(users))
                u = torch.tensor(users[start:end], dtype=torch.long, device=DEVICE)
                it = torch.tensor(items[start:end], dtype=torch.long, device=DEVICE)
                
                # Try full hybrid forward
                if (sbert_embeddings is not None and hasattr(model, 'encode_users')):
                    s = sbert_embeddings[items[start:end]].to(DEVICE)
                    p = popularity[items[start:end]].unsqueeze(1).to(DEVICE)
                    uf = user_features[users[start:end]].to(DEVICE)
                    h = history_embeddings[users[start:end]].to(DEVICE)
                    gids = [genre_table.get("movie_genre_ids", {}).get(int(i), [0])
                            for i in items[start:end]]
                    
                    pred = model(u, it, s, p, gids, h, uf)
                else:
                    pred = model(u, it)
                
                preds.append(pred.cpu().numpy())
        
        predictions = np.concatenate(preds)
    else:
        # Non-torch model (e.g., GlobalMean, BiasOnly, ALS)
        predictions = model.predict(users, items)
    
    return compute_rating_metrics(predictions, ratings)


def evaluate_ranking_model(model, test_df: pd.DataFrame,
                           train_df: pd.DataFrame,
                           n_items: int,
                           sbert_embeddings=None, popularity=None,
                           user_features=None, history_embeddings=None,
                           genre_table=None,
                           k_values=TOP_K_VALUES,
                           max_users: int = 5000,
                           is_torch=True) -> Dict[str, float]:
    """
    Evaluate a model on ranking (Path B).
    Generates top-K recommendations for test users and computes metrics.
    """
    # Build test relevance sets
    test_relevant = {}
    for uid, group in test_df.groupby("user_idx"):
        # Items the user rated highly (rating >= 4) in test
        good_items = group[group["rating"] >= 4]["movie_idx"].values
        if len(good_items) > 0:
            test_relevant[uid] = set(good_items.tolist())
    
    # Items each user already rated in training (to exclude)
    train_items = {}
    for uid, group in train_df.groupby("user_idx"):
        train_items[uid] = set(group["movie_idx"].values.tolist())
    
    # Generate recommendations
    user_recommendations = {}
    eval_users = list(test_relevant.keys())[:max_users]
    
    if is_torch:
        model.eval()
    
    sbert_np = None
    if sbert_embeddings is not None:
        sbert_np = sbert_embeddings.numpy() if isinstance(sbert_embeddings, torch.Tensor) else sbert_embeddings
    
    for uid in tqdm(eval_users, desc="Generating recommendations"):
        # Score all items for this user
        if is_torch and hasattr(model, 'encode_users'):
            with torch.no_grad():
                all_items = torch.arange(n_items, device=DEVICE)
                u_tensor = torch.full((n_items,), uid, dtype=torch.long, device=DEVICE)
                
                if sbert_embeddings is not None:
                    s = sbert_embeddings[:n_items].to(DEVICE)
                    p = popularity[:n_items].unsqueeze(1).to(DEVICE)
                    uf = user_features[uid].unsqueeze(0).expand(n_items, -1).to(DEVICE)
                    h = history_embeddings[uid].unsqueeze(0).expand(n_items, -1).to(DEVICE)
                    gids = [genre_table.get("movie_genre_ids", {}).get(i, [0])
                            for i in range(n_items)]
                    
                    scores = model(u_tensor, all_items, s, p, gids, h, uf)
                else:
                    scores = model(u_tensor, all_items)
                
                scores = scores.cpu().numpy()
        elif is_torch:
            with torch.no_grad():
                u_tensor = torch.full((n_items,), uid, dtype=torch.long, device=DEVICE)
                all_items = torch.arange(n_items, device=DEVICE)
                scores = model(u_tensor, all_items).cpu().numpy()
        elif hasattr(model, 'predict_user'):
            scores = model.predict_user(uid, n_items)
        else:
            scores = model.predict(
                np.full(n_items, uid), np.arange(n_items)
            )
        
        # Mask training items
        exclude = train_items.get(uid, set())
        for idx in exclude:
            if idx < len(scores):
                scores[idx] = -np.inf
        
        # Top-K
        max_k = max(k_values)
        top_k_items = np.argsort(scores)[::-1][:max_k].tolist()
        user_recommendations[uid] = top_k_items
    
    # Compute metrics
    results = compute_ranking_metrics(
        user_recommendations, test_relevant,
        embeddings=sbert_np, k_values=k_values
    )
    
    return results


def run_full_evaluation(model, test_df, train_df, n_items,
                        model_name="model", is_torch=True,
                        sbert_embeddings=None, popularity=None,
                        user_features=None, history_embeddings=None,
                        genre_table=None):
    """Run both rating and ranking evaluation."""
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nEvaluating: {model_name}")
    print("-" * 40)
    
    # Path A metrics
    rating_results = evaluate_rating_model(
        model, test_df, sbert_embeddings, popularity,
        user_features, history_embeddings, genre_table, is_torch
    )
    print(f"  RMSE: {rating_results['RMSE']:.4f}")
    print(f"  MAE:  {rating_results['MAE']:.4f}")
    
    # Path B metrics
    ranking_results = evaluate_ranking_model(
        model, test_df, train_df, n_items,
        sbert_embeddings, popularity, user_features,
        history_embeddings, genre_table, is_torch=is_torch
    )
    for metric, value in ranking_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Combined results
    all_results = {**rating_results, **ranking_results}
    
    # Save
    output_path = RESULTS_DIR / f"{model_name}_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved to {output_path}")
    
    return all_results
