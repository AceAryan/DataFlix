"""
DataFlix — Matrix Factorisation (PyTorch)
MF with biases trained via Adam with cosine annealing.
"""

import torch
import torch.nn as nn
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import LATENT_DIM_K


class MatrixFactorization(nn.Module):
    """
    Biased Matrix Factorisation:
        r̂_ui = μ + b_u + b_i + p_u^T q_i
    
    Trained with masked MSE on observed entries only.
    """
    
    def __init__(self, n_users: int, n_items: int, k: int = LATENT_DIM_K,
                 global_mean: float = 0.0):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.global_mean = global_mean
        
        # Latent factors
        self.P = nn.Embedding(n_users, k)
        self.Q = nn.Embedding(n_items, k)
        
        # Biases
        self.b_u = nn.Embedding(n_users, 1)
        self.b_i = nn.Embedding(n_items, 1)
        
        # Initialise small random
        nn.init.normal_(self.P.weight, 0, 0.01)
        nn.init.normal_(self.Q.weight, 0, 0.01)
        nn.init.zeros_(self.b_u.weight)
        nn.init.zeros_(self.b_i.weight)
    
    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings for (user, item) pairs.
        
        Args:
            user_idx: (batch_size,) user indices
            item_idx: (batch_size,) item indices
        
        Returns:
            (batch_size,) predicted ratings
        """
        p_u = self.P(user_idx)           # (B, k)
        q_i = self.Q(item_idx)           # (B, k)
        b_u = self.b_u(user_idx).squeeze(-1)  # (B,)
        b_i = self.b_i(item_idx).squeeze(-1)  # (B,)
        
        dot = (p_u * q_i).sum(dim=1)    # (B,)
        
        return self.global_mean + b_u + b_i + dot
    
    def predict_all_items(self, user_idx: int) -> torch.Tensor:
        """Predict scores for all items for one user."""
        with torch.no_grad():
            u = torch.tensor([user_idx], device=next(self.parameters()).device)
            p_u = self.P(u)                       # (1, k)
            scores = (p_u @ self.Q.weight.T).squeeze(0)  # (n_items,)
            scores += self.global_mean + self.b_u(u).squeeze() + self.b_i.weight.squeeze()
        return scores
    
    def get_user_embeddings(self) -> torch.Tensor:
        return self.P.weight.data
    
    def get_item_embeddings(self) -> torch.Tensor:
        return self.Q.weight.data


class MFWithImplicit(nn.Module):
    """
    SVD++ style: adds implicit feedback from rated item set.
        r̂_ui = μ + b_u + b_i + (p_u + |N(u)|^{-0.5} Σ y_j)^T q_i
    """
    
    def __init__(self, n_users: int, n_items: int, k: int = LATENT_DIM_K,
                 global_mean: float = 0.0):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.global_mean = global_mean
        
        self.P = nn.Embedding(n_users, k)
        self.Q = nn.Embedding(n_items, k)
        self.Y = nn.Embedding(n_items, k)  # Implicit factors
        self.b_u = nn.Embedding(n_users, 1)
        self.b_i = nn.Embedding(n_items, 1)
        
        nn.init.normal_(self.P.weight, 0, 0.01)
        nn.init.normal_(self.Q.weight, 0, 0.01)
        nn.init.normal_(self.Y.weight, 0, 0.01)
        nn.init.zeros_(self.b_u.weight)
        nn.init.zeros_(self.b_i.weight)
    
    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor,
                user_items: list = None) -> torch.Tensor:
        """
        Args:
            user_idx: (B,) user indices
            item_idx: (B,) item indices
            user_items: list of tensors, each containing item indices rated by user
        """
        p_u = self.P(user_idx)      # (B, k)
        q_i = self.Q(item_idx)      # (B, k)
        b_u = self.b_u(user_idx).squeeze(-1)
        b_i = self.b_i(item_idx).squeeze(-1)
        
        # Add implicit feedback
        if user_items is not None:
            implicit = torch.zeros_like(p_u)
            for b, items in enumerate(user_items):
                if len(items) > 0:
                    y_sum = self.Y(items).sum(dim=0)
                    implicit[b] = y_sum / np.sqrt(len(items))
            p_u = p_u + implicit
        
        dot = (p_u * q_i).sum(dim=1)
        return self.global_mean + b_u + b_i + dot
