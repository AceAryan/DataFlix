"""
DataFlix — Ablation Study & Cold-Start Evaluation
Runs 8-condition ablation and 3 cold-start settings.
"""

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from typing import Dict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    RESULTS_DIR, LATENT_DIM_K, SBERT_DIM
)
from src.models.hybrid import DataFlixModel, DataFlixLite
from src.models.mf import MatrixFactorization
from src.models.baselines import (
    GlobalMeanBaseline, BiasOnlyBaseline, UserKNNBaseline,
    VanillaMF, NeuMF
)
from src.training.trainer import PathATrainer, SimpleMFTrainer, RatingDataset
from src.evaluation.evaluate import evaluate_rating_model, evaluate_ranking_model


def run_ablation_study(
    train_df, val_df, test_df,
    n_users, n_items,
    sbert_embeddings, popularity,
    user_features, history_embeddings,
    genre_table,
    max_epochs=30
) -> pd.DataFrame:
    """
    Run 8-condition ablation study + 6 baselines.
    
    Ablation conditions:
    1. MF only (no content, no BPR, no attention)
    2. MF + genre embeddings
    3. MF + SBERT text embeddings
    4. MF + BPR (no content)
    5. MF + content, concat fusion (no attention) — DataFlixLite
    6. MF + content, self-attention fusion (no BPR)
    7. Full DataFlix — Path A (MSE + attention + content)
    8. Full DataFlix — Path B (BPR + attention + content)
    
    Baselines:
    1. Global Mean
    2. Bias-Only
    3. User-KNN CF
    4. Vanilla MF (SGD)
    5. SVD++
    6. NeuMF
    """
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    
    # ─── Baselines ───
    print("\n" + "=" * 60)
    print("BASELINES")
    print("=" * 60)
    
    # 1. Global Mean
    print("\n--- Global Mean ---")
    gm = GlobalMeanBaseline()
    gm.fit(train_df)
    all_results["Global Mean"] = evaluate_rating_model(gm, test_df, is_torch=False)
    print(f"  RMSE: {all_results['Global Mean']['RMSE']:.4f}")
    
    # 2. Bias-Only
    print("\n--- Bias-Only ---")
    bo = BiasOnlyBaseline()
    bo.fit(train_df)
    all_results["Bias-Only"] = evaluate_rating_model(bo, test_df, is_torch=False)
    print(f"  RMSE: {all_results['Bias-Only']['RMSE']:.4f}")
    
    # 3. User-KNN CF
    print("\n--- User-KNN CF ---")
    knn = UserKNNBaseline(k=50)
    knn.fit(train_df, n_users, n_items)
    # For KNN, only evaluate on a subset (it's slow)
    test_sample = test_df.sample(min(5000, len(test_df)), random_state=42)
    all_results["User-KNN CF"] = evaluate_rating_model(knn, test_sample, is_torch=False)
    print(f"  RMSE: {all_results['User-KNN CF']['RMSE']:.4f}")
    
    # 4. Vanilla MF
    print("\n--- Vanilla MF ---")
    vmf = VanillaMF(n_users, n_items, LATENT_DIM_K)
    trainer = SimpleMFTrainer(vmf, max_epochs=max_epochs)
    trainer.train(train_df, val_df)
    all_results["Vanilla MF"] = evaluate_rating_model(vmf, test_df, is_torch=True)
    print(f"  RMSE: {all_results['Vanilla MF']['RMSE']:.4f}")
    
    # 5. SVD++
    print("\n--- SVD++ ---")
    from src.models.mf import MFWithImplicit
    svdpp = MFWithImplicit(n_users, n_items, LATENT_DIM_K)
    trainer = SimpleMFTrainer(svdpp, max_epochs=max_epochs)
    trainer.train(train_df, val_df)
    all_results["SVD++"] = evaluate_rating_model(svdpp, test_df, is_torch=True)
    print(f"  RMSE: {all_results['SVD++']['RMSE']:.4f}")
    
    # 6. NeuMF
    print("\n--- NeuMF ---")
    neumf = NeuMF(n_users, n_items)
    trainer = SimpleMFTrainer(neumf, max_epochs=max_epochs)
    trainer.train(train_df, val_df)
    all_results["NeuMF"] = evaluate_rating_model(neumf, test_df, is_torch=True)
    print(f"  RMSE: {all_results['NeuMF']['RMSE']:.4f}")
    
    # ─── Ablation Conditions ───
    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    
    # Build datasets
    train_dataset = RatingDataset(train_df, sbert_embeddings, popularity,
                                   user_features, history_embeddings, genre_table)
    val_dataset = RatingDataset(val_df, sbert_embeddings, popularity,
                                 user_features, history_embeddings, genre_table)
    
    # 1. MF only
    print("\n--- MF only ---")
    mf_model = MatrixFactorization(n_users, n_items, LATENT_DIM_K)
    mf_trainer = SimpleMFTrainer(mf_model, max_epochs=max_epochs)
    mf_trainer.train(train_df, val_df)
    all_results["MF only"] = evaluate_rating_model(mf_model, test_df, is_torch=True)
    print(f"  RMSE: {all_results['MF only']['RMSE']:.4f}")
    
    # 5. MF + content, concat (DataFlixLite — no attention)
    print("\n--- MF + content, concat (no attention) ---")
    lite = DataFlixLite(n_users, n_items, LATENT_DIM_K, SBERT_DIM)
    lite_trainer = PathATrainer(lite, max_epochs=max_epochs)
    lite_trainer.train(train_dataset, val_dataset)
    all_results["MF + Concat"] = evaluate_rating_model(
        lite, test_df, sbert_embeddings, popularity,
        user_features, history_embeddings, genre_table, is_torch=True
    )
    print(f"  RMSE: {all_results['MF + Concat']['RMSE']:.4f}")
    
    # 6. MF + content, self-attention (no BPR) — Path A
    print("\n--- MF + Attn (Path A) ---")
    attn_model = DataFlixModel(n_users, n_items, path="A")
    attn_trainer = PathATrainer(attn_model, max_epochs=max_epochs)
    attn_trainer.train(train_dataset, val_dataset)
    all_results["MF + Attn"] = evaluate_rating_model(
        attn_model, test_df, sbert_embeddings, popularity,
        user_features, history_embeddings, genre_table, is_torch=True
    )
    print(f"  RMSE: {all_results['MF + Attn']['RMSE']:.4f}")
    
    # 7. Full DataFlix — Path A
    print("\n--- DataFlix (full, Path A) ---")
    full_a = DataFlixModel(n_users, n_items, path="A")
    full_a_trainer = PathATrainer(full_a, max_epochs=max_epochs)
    full_a_trainer.train(train_dataset, val_dataset)
    all_results["DataFlix (Path A)"] = evaluate_rating_model(
        full_a, test_df, sbert_embeddings, popularity,
        user_features, history_embeddings, genre_table, is_torch=True
    )
    print(f"  RMSE: {all_results['DataFlix (Path A)']['RMSE']:.4f}")
    
    # ─── Save results ───
    results_df = pd.DataFrame(all_results).T
    results_df.to_csv(RESULTS_DIR / "ablation_results.csv")
    
    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Table 1)")
    print("=" * 60)
    print(results_df.to_string(float_format="%.4f"))
    
    return results_df


def run_cold_start_evaluation(
    test_df, train_df, cold_start_users: list,
    n_items: int,
    sbert_embeddings: torch.Tensor,
    popularity: torch.Tensor,
    user_features: torch.Tensor,
    history_embeddings: torch.Tensor,
    genre_table: dict,
    full_model=None,
) -> Dict[str, Dict[str, float]]:
    """
    Cold-start evaluation (3 settings):
    1. Content-only (SBERT embeddings, no MF factors)
    2. SBERT cosine similarity (zero-shot baseline)
    3. DataFlix (content embeddings in place of absent MF vectors)
    """
    print("\n" + "=" * 60)
    print("COLD-START EVALUATION")
    print("=" * 60)
    
    # Filter test data to cold-start users only
    cs_test = test_df[test_df["user_idx"].isin(cold_start_users)]
    print(f"Cold-start test: {len(cs_test)} ratings from {cs_test['user_idx'].nunique()} users")
    
    if len(cs_test) == 0:
        print("No cold-start test data available.")
        return {}
    
    results = {}
    
    # 1. SBERT cosine similarity (zero-shot)
    print("\n--- SBERT Cosine Similarity (zero-shot) ---")
    sbert_np = sbert_embeddings.numpy() if isinstance(sbert_embeddings, torch.Tensor) else sbert_embeddings
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    cs_recommendations = {}
    cs_relevant = {}
    
    for uid in cs_test["user_idx"].unique():
        # Get items rated in training
        user_train = train_df[train_df["user_idx"] == uid]
        if len(user_train) == 0:
            continue
        
        rated_items = user_train["movie_idx"].values
        rated_items = rated_items[rated_items < len(sbert_np)]
        
        if len(rated_items) == 0:
            continue
        
        # Average SBERT of rated items
        user_profile = sbert_np[rated_items].mean(axis=0, keepdims=True)
        
        # Cosine sim to all items
        sims = cosine_similarity(user_profile, sbert_np).flatten()
        
        # Exclude already rated
        for ri in rated_items:
            sims[ri] = -np.inf
        
        top_k = np.argsort(sims)[::-1][:20].tolist()
        cs_recommendations[uid] = top_k
        
        # Relevant items
        test_good = cs_test[(cs_test["user_idx"] == uid) & (cs_test["rating"] >= 4)]
        if len(test_good) > 0:
            cs_relevant[uid] = set(test_good["movie_idx"].values.tolist())
    
    from src.evaluation.metrics import compute_ranking_metrics
    results["SBERT Cosine"] = compute_ranking_metrics(
        cs_recommendations, cs_relevant, sbert_np, k_values=[10]
    )
    print(f"  NDCG@10: {results['SBERT Cosine'].get('NDCG@10', 0):.4f}")
    print(f"  Recall@10: {results['SBERT Cosine'].get('Recall@10', 0):.4f}")
    
    # 2. DataFlix with content-only (set p_u = 0)
    if full_model is not None:
        print("\n--- DataFlix (content-only, p_u=0) ---")
        cs_ranking_results = evaluate_ranking_model(
            full_model, cs_test, train_df, n_items,
            sbert_embeddings, popularity, user_features,
            history_embeddings, genre_table,
            k_values=[10], max_users=min(1000, len(cold_start_users))
        )
        results["DataFlix (cold-start)"] = cs_ranking_results
        print(f"  NDCG@10: {cs_ranking_results.get('NDCG@10', 0):.4f}")
        print(f"  Recall@10: {cs_ranking_results.get('Recall@10', 0):.4f}")
    
    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "cold_start_results.json", "w") as f:
        # Convert any numpy values
        save_results = {}
        for k, v in results.items():
            save_results[k] = {mk: float(mv) for mk, mv in v.items()}
        json.dump(save_results, f, indent=2)
    
    return results
