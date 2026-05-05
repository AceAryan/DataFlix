"""
DataFlix — Full Hybrid Model
Combines MF, content features, and self-attention fusion.
Two paths: Path A (MSE for rating prediction) and Path B (BPR for ranking).
"""

import torch
import torch.nn as nn
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    LATENT_DIM_K, EMBED_DIM_D, NUM_HEADS, MLP_HIDDEN,
    DROPOUT, NUM_GENRES, SBERT_DIM
)
from src.models.attention import SelfAttentionFusion, FeatureProjector


class DataFlixModel(nn.Module):
    """
    DataFlix Hybrid Recommendation Model (Figure 1 from proposal).
    
    Movie streams (4 streams → d each):
        1. q_i^(d) = W_q q_i   — MF latent vector projection
        2. g_i^(d)              — Genre embedding (sum + project)
        3. t_i^(d) = W_t t_i   — SBERT synopsis embedding projection
        4. s_i^(d) = W_s [s_i] — Popularity scalar projection
    
    User streams (3 streams → d each):
        1. p_u^(d) = W_p p_u   — MF latent vector projection
        2. h_u^(d) = W_h t̄_u   — Rating-weighted history embedding projection
        3. f_u^(d) = W_f f_u   — Behavioral features projection
    
    Fusion: Self-Attention → mean-pool → MLP [256, 64] → output
    Path A: clipped rating [1, 5]
    Path B: sigmoid → preference score
    """
    
    def __init__(self, n_users: int, n_items: int,
                 k: int = LATENT_DIM_K,
                 d: int = EMBED_DIM_D,
                 n_heads: int = NUM_HEADS,
                 mlp_hidden: list = None,
                 dropout: float = DROPOUT,
                 n_genres: int = NUM_GENRES,
                 sbert_dim: int = SBERT_DIM,
                 user_feat_dim: int = 2,
                 path: str = "A"):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.d = d
        self.path = path  # "A" for MSE, "B" for BPR
        
        mlp_hidden = mlp_hidden or MLP_HIDDEN
        
        # ── MF Embeddings ──
        self.user_embedding = nn.Embedding(n_users, k)
        self.item_embedding = nn.Embedding(n_items, k)
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)
        
        # ── Genre embedding table ──
        self.genre_embedding = nn.Embedding(n_genres + 1, d)  # +1 for padding
        nn.init.normal_(self.genre_embedding.weight, 0, 0.01)
        
        # ── Feature Projectors (→ d) ──
        # Movie projectors
        self.proj_mf_item = FeatureProjector(k, d)
        self.proj_genre = nn.Identity()  # Already d-dimensional
        self.proj_sbert = FeatureProjector(sbert_dim, d)
        self.proj_pop = FeatureProjector(1, d)
        
        # User projectors
        self.proj_mf_user = FeatureProjector(k, d)
        self.proj_history = FeatureProjector(sbert_dim, d)
        self.proj_user_feat = FeatureProjector(user_feat_dim, d)
        
        # ── Self-Attention Fusion ──
        self.item_attention = SelfAttentionFusion(d, n_heads, dropout)
        self.user_attention = SelfAttentionFusion(d, n_heads, dropout)
        
        # ── MLP Prediction Head ──
        layers = []
        input_dim = 2 * d  # concatenated user and item embeddings
        for hidden_dim in mlp_hidden:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def encode_items(self, item_idx: torch.Tensor,
                     sbert_embs: torch.Tensor,
                     popularity: torch.Tensor,
                     genre_ids_list: list) -> torch.Tensor:
        """
        Encode items through all 4 streams + self-attention.
        
        Args:
            item_idx: (B,) item indices
            sbert_embs: (B, sbert_dim) SBERT embeddings for these items
            popularity: (B, 1) popularity scores
            genre_ids_list: list of lists of genre IDs per item
        
        Returns:
            (B, d) fused item embeddings
        """
        B = item_idx.shape[0]
        device = item_idx.device
        
        # Stream 1: MF latent
        q_i = self.item_embedding(item_idx)      # (B, k)
        s1 = self.proj_mf_item(q_i)              # (B, d)
        
        # Stream 2: Genre (sum genre embeddings then project is identity since already d)
        # genre_ids_list is now a (B, max_genres) LongTensor from _GENRE[iids]
        genre_vecs = self.genre_embedding(genre_ids_list)  # (B, max_genres, d)
        s2 = genre_vecs.sum(dim=1)                         # (B, d)
        
        # Stream 3: SBERT
        s3 = self.proj_sbert(sbert_embs)          # (B, d)
        
        # Stream 4: Popularity
        s4 = self.proj_pop(popularity)             # (B, d)
        
        # Stack streams: (B, 4, d)
        X_i = torch.stack([s1, s2, s3, s4], dim=1)
        
        # Self-attention fusion → (B, d)
        e_i = self.item_attention(X_i)
        
        return e_i
    
    def encode_users(self, user_idx: torch.Tensor,
                     history_embs: torch.Tensor,
                     user_features: torch.Tensor) -> torch.Tensor:
        """
        Encode users through all 3 streams + self-attention.
        
        Args:
            user_idx: (B,) user indices
            history_embs: (B, sbert_dim) rating-weighted history embeddings
            user_features: (B, 4) behavioral features
        
        Returns:
            (B, d) fused user embeddings
        """
        # Stream 1: MF latent
        p_u = self.user_embedding(user_idx)        # (B, k)
        s1 = self.proj_mf_user(p_u)                # (B, d)
        
        # Stream 2: History
        s2 = self.proj_history(history_embs)       # (B, d)
        
        # Stream 3: Behavioral features
        s3 = self.proj_user_feat(user_features)    # (B, d)
        
        # Stack streams: (B, 3, d)
        X_u = torch.stack([s1, s2, s3], dim=1)
        
        # Self-attention fusion → (B, d)
        e_u = self.user_attention(X_u)
        
        return e_u
    
    def forward(self, user_idx, item_idx, sbert_embs, popularity,
                genre_ids_list, history_embs, user_features):
        """
        Full forward pass.
        
        Returns:
            (B,) predicted ratings (Path A) or preference scores (Path B)
        """
        e_u = self.encode_users(user_idx, history_embs, user_features)
        e_i = self.encode_items(item_idx, sbert_embs, popularity, genre_ids_list)
        
        # Concatenate and pass through MLP
        combined = torch.cat([e_u, e_i], dim=1)  # (B, 2d)
        out = self.mlp(combined).squeeze(-1)      # (B,)
        
        if self.path == "A":
            # Clip to [1, 5] for rating prediction
            out = out
        else:
            # Sigmoid for ranking score
            out = torch.sigmoid(out)
        
        return out
    
    def predict_pair_scores(self, user_idx, pos_item_idx, neg_item_idx,
                            sbert_embs_pos, sbert_embs_neg,
                            pop_pos, pop_neg,
                            genre_pos, genre_neg,
                            history_embs, user_features):
        """
        For BPR: compute scores for positive and negative items.
        Returns (score_pos, score_neg)
        """
        e_u = self.encode_users(user_idx, history_embs, user_features)
        e_pos = self.encode_items(pos_item_idx, sbert_embs_pos, pop_pos, genre_pos)
        e_neg = self.encode_items(neg_item_idx, sbert_embs_neg, pop_neg, genre_neg)
        
        score_pos = self.mlp(torch.cat([e_u, e_pos], dim=1)).squeeze(-1)
        score_neg = self.mlp(torch.cat([e_u, e_neg], dim=1)).squeeze(-1)
        
        return score_pos, score_neg
    
    def init_from_als(self, P: np.ndarray, Q: np.ndarray):
        """Initialise MF embeddings from pre-trained ALS factors."""
        with torch.no_grad():
            # Handle dimension mismatch
            k = min(P.shape[1], self.k)
            self.user_embedding.weight[:, :k] = torch.tensor(P[:, :k], dtype=torch.float32)
            self.item_embedding.weight[:, :k] = torch.tensor(Q[:, :k], dtype=torch.float32)
        print(f"Initialised MF embeddings from ALS (k={k})")


class DataFlixLite(nn.Module):
    """
    Simplified version without attention — for ablation:
    MF + content features via simple concatenation.
    """
    
    def __init__(self, n_users: int, n_items: int,
                 k: int = LATENT_DIM_K,
                 sbert_dim: int = SBERT_DIM,
                 user_feat_dim: int = 2,
                 n_genres: int = NUM_GENRES,
                 dropout: float = DROPOUT):
        super().__init__()
        
        self.user_embedding = nn.Embedding(n_users, k)
        self.item_embedding = nn.Embedding(n_items, k)
        self.genre_embedding = nn.Embedding(n_genres + 1, 32)
        
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)
        
        # Direct concatenation → MLP
        item_dim = k + 32 + sbert_dim + 1  # MF + genre + SBERT + pop
        user_dim = k + sbert_dim + user_feat_dim  # MF + history + features
        total_dim = item_dim + user_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    
    def forward(self, user_idx, item_idx, sbert_embs, popularity,
                genre_ids_list, history_embs, user_features):
        B = user_idx.shape[0]
        device = user_idx.device
        
        # Item features
        q_i = self.item_embedding(item_idx)
        genre_vecs = torch.zeros(B, 32, device=device)
        for b, gids in enumerate(genre_ids_list):
            if len(gids) > 0:
                gids_t = torch.tensor(gids, device=device, dtype=torch.long)
                genre_vecs[b] = self.genre_embedding(gids_t).sum(dim=0)
        item_feat = torch.cat([q_i, genre_vecs, sbert_embs, popularity], dim=1)
        
        # User features
        p_u = self.user_embedding(user_idx)
        user_feat = torch.cat([p_u, history_embs, user_features], dim=1)
        
        combined = torch.cat([user_feat, item_feat], dim=1)
        out = self.mlp(combined).squeeze(-1)
        out = torch.clamp(out, 0.5, 5.0)
        
        return out
