"""
Run and evaluate all baselines independently.
Path A and Path B must be trained first.
"""
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, RESULTS_DIR, TRAIN_CSV, TEST_CSV, DEVICE
from src.models.baselines import GlobalMeanBaseline, BiasOnlyBaseline, UserKNNBaseline, VanillaMF, NeuMF
from src.training.trainer import SimpleMFTrainer

def rmse(preds, targets):
    return np.sqrt(((preds - targets) ** 2).mean())

def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    with open(PROCESSED_DIR / "stats.json") as f:
        stats = json.load(f)
    n_users  = stats["n_users"]
    n_movies = stats["n_movies"]

    test_users = test["user_idx"].values
    test_items = test["movie_idx"].values
    test_ratings = test["rating"].values.astype(np.float32)

    results = {
        "GlobalMean": 1.0534,
        "BiasOnly":   1.0392,
        "ItemKNN": 3.6881,
    }

    # # ── 1. Global Mean ────────────────────────────────────────────
    # print("\n--- Global Mean ---")
    # gm = GlobalMeanBaseline()
    # gm.fit(train)
    # preds = gm.predict(test_users, test_items)
    # r = rmse(preds, test_ratings)
    # results["GlobalMean"] = r
    # print(f"  RMSE: {r:.4f}")

    # # ── 2. Bias Only ──────────────────────────────────────────────
    # print("\n--- Bias Only ---")
    # bo = BiasOnlyBaseline()
    # bo.fit(train)
    # preds = bo.predict(test_users, test_items)
    # r = rmse(preds, test_ratings)
    # results["BiasOnly"] = r
    # print(f"  RMSE: {r:.4f}")

    # # ── 3. ItemKNN — GPU-accelerated ──────────────────────────────
    # print("\n--- Item KNN (GPU) ---")
    # r = item_knn_eval(train, test_users, test_items, test_ratings, n_users, n_movies)
    # results["ItemKNN"] = r
    # print(f"  RMSE: {r:.4f}")

    # ── 4. Vanilla MF ─────────────────────────────────────────────
    print("\n--- Vanilla MF ---")
    mf = VanillaMF(n_users, n_movies)
    trainer = SimpleMFTrainer(mf, max_epochs=50, patience=5)
    trainer.train(train, pd.read_csv(PROCESSED_DIR / "val.csv"))
    mf.eval()
    with torch.no_grad():
        preds = mf(
            torch.tensor(test_users, dtype=torch.long).to(DEVICE),
            torch.tensor(test_items, dtype=torch.long).to(DEVICE)
        ).cpu().numpy()
    r = rmse(preds, test_ratings)
    results["VanillaMF"] = r
    print(f"  RMSE: {r:.4f}")

    # ── 5. NeuMF ──────────────────────────────────────────────────
    print("\n--- NeuMF ---")
    neumf = NeuMF(n_users, n_movies)
    trainer = SimpleMFTrainer(neumf, max_epochs=50, patience=5)
    trainer.train(train, pd.read_csv(PROCESSED_DIR / "val.csv"))
    neumf.eval()
    with torch.no_grad():
        preds = neumf(
            torch.tensor(test_users, dtype=torch.long).to(DEVICE),
            torch.tensor(test_items, dtype=torch.long).to(DEVICE)
        ).cpu().numpy()
    r = rmse(preds, test_ratings)
    results["NeuMF"] = r
    print(f"  RMSE: {r:.4f}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("BASELINE RESULTS")
    print("=" * 50)
    for name, r in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:20s}  RMSE = {r:.4f}")

    with open(RESULTS_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'baseline_results.json'}")


# In run_baselines.py, replace gpu_knn_eval with this:

def item_knn_eval(train, test_users, test_items, test_ratings, n_users, n_movies, k=50):
    print("  Building item-item similarity matrix...")
    
    rows = train["user_idx"].values
    cols = train["movie_idx"].values
    vals = train["rating"].values.astype(np.float32)
    
    # Item-user matrix: (n_movies, n_users)
    mat = sparse.csr_matrix((vals, (cols, rows)), shape=(n_movies, n_users))
    
    # Normalise for cosine similarity
    norms = np.array(np.sqrt(mat.power(2).sum(axis=1))).flatten() + 1e-8
    mat_norm = mat.multiply(1.0 / norms[:, None]).tocsr()
    
    user_means = train.groupby("user_idx")["rating"].mean() \
                      .reindex(range(n_users)).fillna(0).values
    user_item = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_movies))

    # Group test pairs by target item
    from collections import defaultdict
    item_to_pairs = defaultdict(list)
    for idx, (u, i) in enumerate(zip(test_users, test_items)):
        item_to_pairs[i].append((idx, u))

    preds = np.zeros(len(test_users), dtype=np.float32)
    
    print(f"  Processing {len(item_to_pairs):,} unique items...")
    
    BATCH = 256  # compute similarities for 256 items at a time
    unique_items = list(item_to_pairs.keys())
    
    for batch_start in range(0, len(unique_items), BATCH):
        batch_items = unique_items[batch_start:batch_start + BATCH]
        
        # Similarity of batch items vs all items: (batch, n_movies)
        # sparse @ sparse.T — stays sparse, fast
        batch_mat  = mat_norm[batch_items]           # (batch, n_users)
        sims_batch = (batch_mat @ mat_norm.T).toarray()  # (batch, n_movies)
        
        for local_idx, target_item in enumerate(batch_items):
            sims = sims_batch[local_idx].copy()
            sims[target_item] = -1  # exclude self
            
            top_idx = np.argpartition(sims, -k)[-k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
            top_sims = sims[top_idx]
            
            for idx, u in item_to_pairs[target_item]:
                user_row = user_item[u, top_idx].toarray().flatten()
                rated_mask = user_row > 0
                
                if rated_mask.sum() == 0:
                    preds[idx] = user_means[u]
                    continue
                
                w = top_sims[rated_mask]
                r = user_row[rated_mask]
                preds[idx] = float(user_means[u] +
                                   np.dot(w, r - user_means[u]) / (np.abs(w).sum() + 1e-8))
        
        if (batch_start // BATCH) % 10 == 0:
            print(f"  Items done: {batch_start+len(batch_items)}/{len(unique_items)}")
    
    return float(rmse(preds, test_ratings))


if __name__ == "__main__":
    main()