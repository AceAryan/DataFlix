"""
DataFlix — Evaluation Metrics
RMSE, MAE, NDCG@K, Precision@K, Recall@K, MRR, ILD@10, Coverage.
"""

import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────
# Rating Prediction Metrics (Path A)
# ──────────────────────────────────────────────

def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(predictions - targets))


# ──────────────────────────────────────────────
# Ranking Metrics (Path B)
# ──────────────────────────────────────────────

def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Precision@K: fraction of recommended items that are relevant.
    """
    rec_k = recommended[:k]
    hits = len(set(rec_k) & relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Recall@K: fraction of relevant items that are recommended.
    """
    if len(relevant) == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.
    Binary relevance (1 if in relevant set, 0 otherwise).
    """
    rec_k = recommended[:k]
    
    # DCG
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because rank starts at 1
    
    # Ideal DCG
    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(recommended: List[int], relevant: set) -> float:
    """
    Mean Reciprocal Rank: 1/rank of the first relevant item.
    """
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ild_at_k(recommended: List[int], embeddings: np.ndarray, k: int) -> float:
    """
    Intra-List Diversity @ K:
    Mean pairwise cosine dissimilarity of recommended items.
    ILD = 1 - mean(cosine_similarity(e_i, e_j)) for all pairs.
    """
    rec_k = recommended[:k]
    if len(rec_k) < 2:
        return 0.0
    
    # Filter valid indices
    valid = [i for i in rec_k if i < len(embeddings)]
    if len(valid) < 2:
        return 0.0
    
    vecs = embeddings[valid]  # (k', dim)
    
    # Pairwise cosine similarity
    sim_matrix = cosine_similarity(vecs)
    
    # Mean of upper triangle (excluding diagonal)
    n = len(valid)
    total_sim = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += sim_matrix[i, j]
            count += 1
    
    if count == 0:
        return 0.0
    
    mean_sim = total_sim / count
    return 1.0 - mean_sim  # Dissimilarity


def coverage(all_recommended: List[List[int]], n_items: int) -> float:
    """
    Coverage: fraction of catalogue recommended to ≥ 1 user.
    """
    recommended_items = set()
    for rec_list in all_recommended:
        recommended_items.update(rec_list)
    return len(recommended_items) / n_items


# ──────────────────────────────────────────────
# Aggregation Helpers
# ──────────────────────────────────────────────

def compute_ranking_metrics(
    user_recommendations: Dict[int, List[int]],
    user_relevant: Dict[int, set],
    embeddings: np.ndarray = None,
    k_values: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Compute all ranking metrics averaged over users.
    
    Args:
        user_recommendations: {user_idx: [ordered list of recommended item_idx]}
        user_relevant: {user_idx: set of relevant item_idx}
        embeddings: item embeddings for ILD computation
        k_values: list of K values
    
    Returns:
        dict of metric_name -> value
    """
    results = {}
    
    for k in k_values:
        prec_scores = []
        rec_scores = []
        ndcg_scores = []
        mrr_scores = []
        ild_scores = []
        
        for uid, rec_list in user_recommendations.items():
            rel = user_relevant.get(uid, set())
            if len(rel) == 0:
                continue
            
            prec_scores.append(precision_at_k(rec_list, rel, k))
            rec_scores.append(recall_at_k(rec_list, rel, k))
            ndcg_scores.append(ndcg_at_k(rec_list, rel, k))
            mrr_scores.append(mrr(rec_list, rel))
            
            if embeddings is not None:
                ild_scores.append(ild_at_k(rec_list, embeddings, k))
        
        results[f"Precision@{k}"] = np.mean(prec_scores) if prec_scores else 0.0
        results[f"Recall@{k}"] = np.mean(rec_scores) if rec_scores else 0.0
        results[f"NDCG@{k}"] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        results[f"MRR@{k}"] = np.mean(mrr_scores) if mrr_scores else 0.0
        
        if ild_scores:
            results[f"ILD@{k}"] = np.mean(ild_scores)
    
    # Coverage
    all_recs = list(user_recommendations.values())
    if embeddings is not None:
        results["Coverage"] = coverage(all_recs, len(embeddings))
    
    return results


def compute_rating_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """Compute rating prediction metrics."""
    return {
        "RMSE": rmse(predictions, targets),
        "MAE": mae(predictions, targets),
    }
