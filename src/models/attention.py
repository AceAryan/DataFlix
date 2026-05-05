"""
DataFlix — Self-Attention Fusion Module
Multi-head self-attention to fuse multiple feature streams.
"""

import torch
import torch.nn as nn
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import EMBED_DIM_D, NUM_HEADS


class SelfAttentionFusion(nn.Module):
    """
    Multi-head self-attention fusion (Equation 6 from proposal):
    
        Attn(X) = softmax(XW_Q (XW_K)^T / √d_h) XW_V
    
    where X ∈ R^{S×d} is the stacked feature stream matrix (S streams),
    d_h = d / H per head.
    
    The fused embedding is obtained by mean-pooling the attended rows:
        e = (1/S) Σ_m [Attn(X)]_m
    
    This is input-dependent — each movie/user weights its own streams differently.
    """
    
    def __init__(self, d: int = EMBED_DIM_D, n_heads: int = NUM_HEADS,
                 dropout: float = 0.1):
        super().__init__()
        
        assert d % n_heads == 0, f"d={d} must be divisible by n_heads={n_heads}"
        
        self.d = d
        self.n_heads = n_heads
        self.d_h = d // n_heads
        
        # Learned projections
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d, d)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch_size, n_streams, d) — stacked feature streams
        
        Returns:
            (batch_size, d) — fused embedding (mean-pooled)
        """
        B, S, D = X.shape
        
        # Project to Q, K, V
        Q = self.W_Q(X)  # (B, S, d)
        K = self.W_K(X)  # (B, S, d)
        V = self.W_V(X)  # (B, S, d)
        
        # Reshape for multi-head: (B, H, S, d_h)
        Q = Q.view(B, S, self.n_heads, self.d_h).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_h).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_h).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_h)  # (B, H, S, S)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)  # (B, H, S, d_h)
        
        # Reshape back: (B, S, d)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        
        # Output projection with residual connection and layer norm
        attn_output = self.out_proj(attn_output)
        attn_output = self.layer_norm(X + attn_output)
        
        # Mean-pool across streams
        fused = attn_output.mean(dim=1)  # (B, d)
        
        return fused


class FeatureProjector(nn.Module):
    """
    Project a feature vector from any dimension to the common dimension d.
    Used for each individual feature stream.
    """
    
    def __init__(self, input_dim: int, output_dim: int = EMBED_DIM_D):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
