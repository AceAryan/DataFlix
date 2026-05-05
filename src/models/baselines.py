"""
DataFlix — Baseline Models
6 baselines for comparison: Global Mean, Bias-Only, User-KNN,
Vanilla MF, SVD++, NeuMF.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import sparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import LATENT_DIM_K, DROPOUT


# ──────────────────────────────────────────────
# 1. Global Mean Baseline
# ──────────────────────────────────────────────

class GlobalMeanBaseline:
    """Predict global mean rating for all pairs."""
    
    def __init__(self):
        self.global_mean = 0.0
    
    def fit(self, train_df):
        self.global_mean = train_df["rating"].mean()
        print(f"GlobalMean: μ = {self.global_mean:.4f}")
    
    def predict(self, user_idx, item_idx):
        return np.full(len(user_idx), self.global_mean, dtype=np.float32)
    
    def predict_user(self, user_idx, n_items):
        return np.full(n_items, self.global_mean, dtype=np.float32)


# ──────────────────────────────────────────────
# 2. Bias-Only Baseline  
# ──────────────────────────────────────────────

class BiasOnlyBaseline:
    """r̂_ui = μ + b_u + b_i"""
    
    def __init__(self, reg: float = 25.0):
        self.reg = reg
        self.mu = 0.0
        self.b_u = {}
        self.b_i = {}
    
    def fit(self, train_df):
        self.mu = train_df["rating"].mean()
        
        # Compute user biases
        user_groups = train_df.groupby("user_idx")
        for uid, group in user_groups:
            self.b_u[uid] = (group["rating"].sum() - len(group) * self.mu) / (len(group) + self.reg)
        
        # Compute item biases (accounting for user bias)
        item_groups = train_df.groupby("movie_idx")
        for iid, group in item_groups:
            residuals = group["rating"] - self.mu - group["user_idx"].map(self.b_u).fillna(0)
            self.b_i[iid] = residuals.sum() / (len(group) + self.reg)
        
        print(f"BiasOnly: μ={self.mu:.4f}, "
              f"{len(self.b_u)} user biases, {len(self.b_i)} item biases")
    
    def predict(self, user_idx, item_idx):
        preds = np.array([
            self.mu + self.b_u.get(u, 0) + self.b_i.get(i, 0)
            for u, i in zip(user_idx, item_idx)
        ], dtype=np.float32)
        return preds
    
    def predict_user(self, user_idx, n_items):
        preds = np.array([
            self.mu + self.b_u.get(user_idx, 0) + self.b_i.get(i, 0)
            for i in range(n_items)
        ], dtype=np.float32)
        return preds


# ──────────────────────────────────────────────
# 3. User-KNN Collaborative Filtering
# ──────────────────────────────────────────────

class UserKNNBaseline:
    """
    User-KNN CF with cosine similarity, k=50 neighbours.
    """
    
    def __init__(self, k: int = 50):
        self.k = k
        self.user_item_matrix = None
        self.user_means = None
        self.n_users = 0
        self.n_items = 0
    
    def fit(self, train_df, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items
        
        # Build sparse user-item matrix
        rows = train_df["user_idx"].values
        cols = train_df["movie_idx"].values
        vals = train_df["rating"].values
        
        self.user_item_matrix = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_users, n_items)
        )
        
        self.user_means = np.array(
            train_df.groupby("user_idx")["rating"].mean().reindex(range(n_users)).fillna(0)
        )
        
        print(f"UserKNN: k={self.k}, matrix shape={self.user_item_matrix.shape}")
    
    def predict(self, user_idx, item_idx):
        preds = []
        for u, i in zip(user_idx, item_idx):
            preds.append(self._predict_single(u, i))
        return np.array(preds, dtype=np.float32)
    
    def _predict_single(self, user_idx, item_idx):
        # Get users who rated this item
        item_col = self.user_item_matrix[:, item_idx].toarray().flatten()
        rated_users = np.where(item_col > 0)[0]
        
        if len(rated_users) == 0:
            return self.user_means[user_idx]
        
        # Compute similarity with target user
        user_vec = self.user_item_matrix[user_idx].toarray().flatten()
        
        similarities = []
        for other in rated_users:
            if other == user_idx:
                continue
            other_vec = self.user_item_matrix[other].toarray().flatten()
            # Cosine similarity
            dot = np.dot(user_vec, other_vec)
            norm = np.linalg.norm(user_vec) * np.linalg.norm(other_vec)
            sim = dot / (norm + 1e-8)
            similarities.append((other, sim))
        
        if not similarities:
            return self.user_means[user_idx]
        
        # Take top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:self.k]
        
        # Weighted average
        num = sum(sim * (item_col[other] - self.user_means[other]) for other, sim in top_k)
        denom = sum(abs(sim) for _, sim in top_k)
        
        if denom > 0:
            return self.user_means[user_idx] + num / denom
        return self.user_means[user_idx]
    
    def predict_user(self, user_idx, n_items):
        return np.array([self._predict_single(user_idx, i) for i in range(n_items)],
                        dtype=np.float32)


# ──────────────────────────────────────────────
# 4. Vanilla MF (SGD, no biases, no content)
# ──────────────────────────────────────────────

class VanillaMF(nn.Module):
    """Plain MF: r̂_ui = p_u^T q_i (no bias, no content)."""
    
    def __init__(self, n_users: int, n_items: int, k: int = LATENT_DIM_K):
        super().__init__()
        self.P = nn.Embedding(n_users, k)
        self.Q = nn.Embedding(n_items, k)
        nn.init.normal_(self.P.weight, 0, 0.01)
        nn.init.normal_(self.Q.weight, 0, 0.01)
    
    def forward(self, user_idx, item_idx):
        return (self.P(user_idx) * self.Q(item_idx)).sum(dim=1)


# ──────────────────────────────────────────────
# 5. SVD++ (imported from mf.py)
# ──────────────────────────────────────────────
# Use MFWithImplicit from src.models.mf


# ──────────────────────────────────────────────
# 6. NeuMF (Neural Collaborative Filtering)
# ──────────────────────────────────────────────

class NeuMF(nn.Module):
    """
    Neural Matrix Factorisation combining GMF and MLP paths.
    Reference: He et al., 2017.
    """
    
    def __init__(self, n_users: int, n_items: int,
                 k_gmf: int = 32, k_mlp: int = 32,
                 mlp_layers: list = None, dropout: float = DROPOUT):
        super().__init__()
        
        mlp_layers = mlp_layers or [128, 64, 32]
        
        # GMF path
        self.gmf_user = nn.Embedding(n_users, k_gmf)
        self.gmf_item = nn.Embedding(n_items, k_gmf)
        
        # MLP path
        self.mlp_user = nn.Embedding(n_users, k_mlp)
        self.mlp_item = nn.Embedding(n_items, k_mlp)
        
        nn.init.normal_(self.gmf_user.weight, 0, 0.01)
        nn.init.normal_(self.gmf_item.weight, 0, 0.01)
        nn.init.normal_(self.mlp_user.weight, 0, 0.01)
        nn.init.normal_(self.mlp_item.weight, 0, 0.01)
        
        # MLP layers
        mlp_modules = []
        input_dim = 2 * k_mlp
        for dim in mlp_layers:
            mlp_modules.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = dim
        self.mlp_net = nn.Sequential(*mlp_modules)
        
        # Final layer
        self.output = nn.Linear(k_gmf + mlp_layers[-1], 1)
    
    def forward(self, user_idx, item_idx):
        # GMF path
        gmf = self.gmf_user(user_idx) * self.gmf_item(item_idx)
        
        # MLP path
        mlp_input = torch.cat([
            self.mlp_user(user_idx),
            self.mlp_item(item_idx)
        ], dim=1)
        mlp_out = self.mlp_net(mlp_input)
        
        # Combine
        combined = torch.cat([gmf, mlp_out], dim=1)
        return self.output(combined).squeeze(-1)
