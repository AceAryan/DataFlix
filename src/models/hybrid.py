"""
DataFlix — Model 3: Hybrid CF + Content Model
src/models/hybrid.py

Combines four information streams into a unified scoring model:

  Stream 1 — ALS latent factors       (collaborative signal, k-dim)
  Stream 2 — BPR embeddings           (ranking-optimised CF signal, k-dim)
  Stream 3 — SBERT synopsis embeddings (semantic content, 384-dim)
  Stream 4 — IMDB structured features (genre OHE + runtime + votes, 23-dim)

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  User side                   Item side              │
  │  ─────────                   ─────────              │
  │  ALS p_u (k)                 ALS q_i (k)            │
  │  BPR p_u (k)                 BPR q_i (k)            │
  │  History emb (384)           SBERT emb (384)        │
  │       │                      IMDB feats (23)        │
  │       │                      Popularity (1)         │
  │       │                           │                 │
  │  User MLP → d-dim           Item MLP → d-dim        │
  │       │                           │                 │
  │       └──── dot product ──────────┘                 │
  │                    │                                │
  │             Self-attention                          │
  │                    │                                │
  │             MLP prediction head                     │
  │                    │                                │
  │              rating / score                         │
  └─────────────────────────────────────────────────────┘

Trained end-to-end with MSE loss on mean-centred ratings (Path A).
CF factor weights are frozen for first N epochs, then unfrozen for fine-tuning.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from src.config import (
        PROCESSED_DIR, RESULTS_DIR, DEVICE,
        TRAIN_CSV, VAL_CSV,
        SBERT_EMBEDDINGS_PATH, IMDB_FEATURES_PATH,
        POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH,
        LATENT_DIM_K, EMBED_DIM_D, NUM_HEADS,
        MLP_HIDDEN, DROPOUT, SBERT_DIM, IMDB_FEAT_DIM,
        LR_PATH_A, WEIGHT_DECAY, COSINE_T_MAX,
        EARLY_STOP_PATIENCE, MAX_EPOCHS, BATCH_SIZE,
    )
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR           = _ROOT / "data/processed"
    RESULTS_DIR             = _ROOT / "results"
    DEVICE                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_CSV               = PROCESSED_DIR / "train.csv"
    VAL_CSV                 = PROCESSED_DIR / "val.csv"
    SBERT_EMBEDDINGS_PATH   = PROCESSED_DIR / "sbert_embeddings.pt"
    IMDB_FEATURES_PATH      = PROCESSED_DIR / "imdb_features.pt"
    POPULARITY_PATH         = PROCESSED_DIR / "popularity.pt"
    HISTORY_EMBEDDINGS_PATH = PROCESSED_DIR / "history_embeddings.pt"
    LATENT_DIM_K            = 100
    EMBED_DIM_D             = 128
    NUM_HEADS               = 4
    MLP_HIDDEN              = [256, 64]
    DROPOUT                 = 0.2
    SBERT_DIM               = 384
    IMDB_FEAT_DIM           = 23
    LR_PATH_A               = 1e-3
    WEIGHT_DECAY            = 1e-4
    COSINE_T_MAX            = 50
    EARLY_STOP_PATIENCE     = 5
    MAX_EPOCHS              = 100
    BATCH_SIZE              = 8192

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

HYBRID_CKPT_PATH = RESULTS_DIR / "hybrid_best.pt"


# ── Dataset ───────────────────────────────────────────────────────────────────

class RatingsDataset(Dataset):
    """
    PyTorch Dataset wrapping the ratings DataFrame.
    Returns (user_idx, movie_idx, rating_centered) tensors.
    """

    def __init__(self, df):
        self.users   = torch.tensor(df["user_idx"].values,       dtype=torch.long)
        self.movies  = torch.tensor(df["movie_idx"].values,      dtype=torch.long)
        self.ratings = torch.tensor(df["rating_centered"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


# ── Sub-modules ───────────────────────────────────────────────────────────────

def _make_mlp(
    in_dim:     int,
    hidden:     list[int],
    out_dim:    int,
    dropout:    float,
    activation: nn.Module = None,
) -> nn.Sequential:
    """Build a fully-connected MLP with BatchNorm and Dropout."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [
            nn.Linear(prev, h),
            nn.BatchNorm1d(h),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)


class FeatureFusion(nn.Module):
    """
    Multi-head self-attention over a sequence of feature vectors,
    followed by mean pooling.

    Takes a stack of d-dim feature vectors and lets them attend to each other
    before combining — this allows the model to learn which features are most
    relevant for a given user-item interaction rather than treating all streams
    equally.

    Input:  (B, n_streams, d)
    Output: (B, d)
    """

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim   = d,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_streams, d)
        attn_out, _ = self.attn(x, x, x)          # self-attention
        x = self.norm(x + attn_out)                # residual + layernorm
        return x.mean(dim=1)                        # mean pool → (B, d)


# ── Main Model ────────────────────────────────────────────────────────────────

class HybridModel(nn.Module):
    """
    Hybrid CF + Content recommendation model.

    Parameters
    ----------
    n_users       : total number of users
    n_items       : total number of items
    n_factors     : ALS/BPR latent dimension k
    embed_dim     : common projection dimension d
    n_heads       : attention heads in FeatureFusion
    mlp_hidden    : hidden layer widths for prediction MLP
    dropout       : dropout rate
    sbert_dim     : SBERT embedding dimension (384)
    imdb_feat_dim : IMDB feature dimension (23)
    """

    def __init__(
        self,
        n_users:       int,
        n_items:       int,
        n_factors:     int   = LATENT_DIM_K,
        embed_dim:     int   = EMBED_DIM_D,
        n_heads:       int   = NUM_HEADS,
        mlp_hidden:    list  = MLP_HIDDEN,
        dropout:       float = DROPOUT,
        sbert_dim:     int   = SBERT_DIM,
        imdb_feat_dim: int   = IMDB_FEAT_DIM,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── CF embeddings (initialised from ALS/BPR, optionally frozen) ──
        self.user_cf = nn.Embedding(n_users, n_factors * 2)  # ALS + BPR concatenated
        self.item_cf = nn.Embedding(n_items, n_factors * 2)

        # ── Content feature projections → embed_dim ──
        self.sbert_proj   = nn.Linear(sbert_dim,     embed_dim)
        self.imdb_proj    = nn.Linear(imdb_feat_dim,  embed_dim)
        self.history_proj = nn.Linear(sbert_dim,     embed_dim)

        # ── CF projections → embed_dim ──
        self.user_cf_proj = nn.Linear(n_factors * 2, embed_dim)
        self.item_cf_proj = nn.Linear(n_factors * 2, embed_dim)

        # ── Popularity scalar → embed_dim (via learned embedding) ──
        self.pop_proj = nn.Linear(1, embed_dim)

        # ── Per-side projection MLPs ──
        # User streams: CF projection + history projection  → 2 streams
        # Item streams: CF projection + SBERT + IMDB + pop → 4 streams
        self.user_fusion = FeatureFusion(embed_dim, n_heads, dropout)
        self.item_fusion = FeatureFusion(embed_dim, n_heads, dropout)

        # ── Final prediction MLP ──
        # Input: user_repr (d) ⊕ item_repr (d) ⊕ dot_product (1) = 2d+1
        pred_in_dim = embed_dim * 2 + 1
        self.pred_mlp = _make_mlp(pred_in_dim, mlp_hidden, 1, dropout)

        # ── Layer norms for projected features ──
        self.ln_sbert   = nn.LayerNorm(embed_dim)
        self.ln_imdb    = nn.LayerNorm(embed_dim)
        self.ln_history = nn.LayerNorm(embed_dim)
        self.ln_user_cf = nn.LayerNorm(embed_dim)
        self.ln_item_cf = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        scale = 1.0 / np.sqrt(self.embed_dim)
        nn.init.normal_(self.user_cf.weight, 0, scale)
        nn.init.normal_(self.item_cf.weight, 0, scale)
        for module in [self.sbert_proj, self.imdb_proj,
                       self.history_proj, self.user_cf_proj,
                       self.item_cf_proj, self.pop_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def load_cf_weights(
        self,
        user_factors: torch.Tensor,  # (n_users, k) ALS
        item_factors: torch.Tensor,  # (n_items, k) ALS
        user_bpr:     torch.Tensor,  # (n_users, k) BPR
        item_bpr:     torch.Tensor,  # (n_items, k) BPR
    ) -> None:
        """
        Initialise CF embedding weights from pre-trained ALS + BPR factors.
        Concatenates both along the feature dimension → (n, 2k).
        """
        with torch.no_grad():
            user_init = torch.cat([user_factors, user_bpr], dim=1)
            item_init = torch.cat([item_factors, item_bpr], dim=1)
            self.user_cf.weight.copy_(user_init)
            self.item_cf.weight.copy_(item_init)
        log.info("Hybrid model CF weights initialised from ALS + BPR")

    def freeze_cf(self) -> None:
        """Freeze CF embeddings — train only content projections first."""
        self.user_cf.weight.requires_grad_(False)
        self.item_cf.weight.requires_grad_(False)
        log.info("CF embeddings frozen")

    def unfreeze_cf(self) -> None:
        """Unfreeze CF embeddings for end-to-end fine-tuning."""
        self.user_cf.weight.requires_grad_(True)
        self.item_cf.weight.requires_grad_(True)
        log.info("CF embeddings unfrozen for fine-tuning")

    def forward(
        self,
        user_idx:     torch.Tensor,  # (B,)
        item_idx:     torch.Tensor,  # (B,)
        sbert_emb:    torch.Tensor,  # (B, 384)
        imdb_feats:   torch.Tensor,  # (B, 23)
        history_emb:  torch.Tensor,  # (B, 384)
        popularity:   torch.Tensor,  # (B, 1)
    ) -> torch.Tensor:               # (B,)
        """Forward pass — returns predicted mean-centred ratings."""

        # ── User representation ──
        p_u = self.user_cf(user_idx)                       # (B, 2k)
        user_cf_repr  = self.ln_user_cf(self.user_cf_proj(p_u))   # (B, d)
        user_hist_repr = self.ln_history(self.history_proj(history_emb))  # (B, d)

        # Stack user streams: (B, 2, d)
        user_streams = torch.stack([user_cf_repr, user_hist_repr], dim=1)
        user_repr    = self.user_fusion(user_streams)       # (B, d)

        # ── Item representation ──
        q_i = self.item_cf(item_idx)                       # (B, 2k)
        item_cf_repr    = self.ln_item_cf(self.item_cf_proj(q_i))  # (B, d)
        item_sbert_repr = self.ln_sbert(self.sbert_proj(sbert_emb))  # (B, d)
        item_imdb_repr  = self.ln_imdb(self.imdb_proj(imdb_feats))   # (B, d)
        item_pop_repr   = self.pop_proj(popularity)                   # (B, d)

        # Stack item streams: (B, 4, d)
        item_streams = torch.stack(
            [item_cf_repr, item_sbert_repr, item_imdb_repr, item_pop_repr], dim=1
        )
        item_repr = self.item_fusion(item_streams)          # (B, d)

        # ── Dot product between user and item representations ──
        dot = (user_repr * item_repr).sum(dim=1, keepdim=True)  # (B, 1)

        # ── Prediction MLP ──
        x      = torch.cat([user_repr, item_repr, dot], dim=1)  # (B, 2d+1)
        rating = self.pred_mlp(x).squeeze(1)                     # (B,)

        return rating


# ── Trainer ───────────────────────────────────────────────────────────────────

class HybridTrainer:
    """
    Manages the two-phase training loop for HybridModel.

    Phase 1 (freeze_epochs): CF embeddings frozen — only content projections
    and prediction head are trained. This prevents the rich CF signal from
    being immediately overwritten by random content projection gradients.

    Phase 2 (remaining epochs): All parameters unfrozen for end-to-end
    fine-tuning with a lower effective learning rate (cosine annealing).
    """

    def __init__(
        self,
        model:          HybridModel,
        sbert_emb:      torch.Tensor,   # (n_items, 384)  — on CPU, moved per batch
        imdb_feats:     torch.Tensor,   # (n_items, 23)
        popularity:     torch.Tensor,   # (n_items,)
        history_emb:    torch.Tensor,   # (n_users, 384)
        device:         torch.device = DEVICE,
        lr:             float        = LR_PATH_A,
        weight_decay:   float        = WEIGHT_DECAY,
        n_epochs:       int          = MAX_EPOCHS,
        batch_size:     int          = BATCH_SIZE,
        patience:       int          = EARLY_STOP_PATIENCE,
        freeze_epochs:  int          = 5,
    ):
        self.model         = model.to(device)
        self.device        = device
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.patience      = patience
        self.freeze_epochs = freeze_epochs

        # Feature tensors — kept on CPU, sliced per batch to save GPU memory
        self.sbert_emb   = sbert_emb
        self.imdb_feats  = imdb_feats
        self.popularity  = popularity
        self.history_emb = history_emb

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=COSINE_T_MAX, eta_min=1e-5
        )
        self.criterion = nn.MSELoss()

        self.best_val_rmse   = float("inf")
        self.patience_counter = 0
        self.train_rmse_history: list[float] = []
        self.val_rmse_history:   list[float] = []

    def _batch_features(
        self,
        user_idx:  torch.Tensor,
        movie_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gather feature tensors for a batch by index.
        non_blocking=True overlaps CPU→GPU transfer with compute on the
        previous batch, giving ~10% throughput improvement on large batches.
        """
        cpu_movie = movie_idx.cpu()
        cpu_user  = user_idx.cpu()
        sbert   = self.sbert_emb[cpu_movie].to(self.device, non_blocking=True)
        imdb    = self.imdb_feats[cpu_movie].to(self.device, non_blocking=True)
        pop     = self.popularity[cpu_movie].unsqueeze(1).to(self.device, non_blocking=True)
        history = self.history_emb[cpu_user].to(self.device, non_blocking=True)
        return sbert, imdb, pop, history

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        total_n    = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for user_idx, movie_idx, rating in loader:
                user_idx  = user_idx.to(self.device)
                movie_idx = movie_idx.to(self.device)
                rating    = rating.to(self.device)

                sbert, imdb, pop, history = self._batch_features(user_idx, movie_idx)

                pred = self.model(
                    user_idx, movie_idx,
                    sbert, imdb, history, pop,
                )
                loss = self.criterion(pred, rating)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping — prevents exploding gradients in
                    # early training when content projections are random
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                batch_n      = rating.size(0)
                total_loss  += loss.item() * batch_n
                total_n     += batch_n

        return float(np.sqrt(total_loss / total_n))  # RMSE

    def fit(
        self,
        train_df,
        val_df,
    ) -> "HybridTrainer":
        import pandas as pd

        train_loader = DataLoader(
            RatingsDataset(train_df), batch_size=self.batch_size,
            shuffle=True, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            RatingsDataset(val_df), batch_size=self.batch_size * 2,
            shuffle=False, num_workers=4, pin_memory=True,
        )

        log.info(f"Hybrid training: {self.n_epochs} epochs | "
                 f"batch={self.batch_size} | device={self.device}")
        log.info(f"  Phase 1 (frozen CF): epochs 1–{self.freeze_epochs}")
        log.info(f"  Phase 2 (full):      epochs {self.freeze_epochs+1}–{self.n_epochs}")

        # Phase 1: freeze CF embeddings
        self.model.freeze_cf()

        for epoch in range(1, self.n_epochs + 1):

            # Transition to Phase 2
            if epoch == self.freeze_epochs + 1:
                self.model.unfreeze_cf()

            train_rmse = self._run_epoch(train_loader, train=True)
            val_rmse   = self._run_epoch(val_loader,   train=False)
            self.scheduler.step()

            self.train_rmse_history.append(train_rmse)
            self.val_rmse_history.append(val_rmse)

            lr_now = self.scheduler.get_last_lr()[0]
            phase  = "freeze" if epoch <= self.freeze_epochs else "full  "
            log.info(f"  [{phase}] Epoch {epoch:>3}/{self.n_epochs}  "
                     f"train RMSE={train_rmse:.5f}  val RMSE={val_rmse:.5f}  "
                     f"lr={lr_now:.2e}")

            # ── Early stopping & checkpointing ──
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse    = val_rmse
                self.patience_counter = 0
                self.save(HYBRID_CKPT_PATH)
                log.info(f"    ✓ New best val RMSE={val_rmse:.5f} — checkpoint saved")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    log.info(f"  Early stopping at epoch {epoch} "
                             f"(no improvement for {self.patience} epochs)")
                    break

        log.info(f"\nBest val RMSE: {self.best_val_rmse:.5f}")
        return self

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = HYBRID_CKPT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":       self.model.state_dict(),
            "train_rmse":        self.train_rmse_history,
            "val_rmse":          self.val_rmse_history,
            "best_val_rmse":     self.best_val_rmse,
            "n_users":           self.model.user_cf.num_embeddings,
            "n_items":           self.model.item_cf.num_embeddings,
            "embed_dim":         self.model.embed_dim,
        }, path)

    @classmethod
    def load_model(
        cls,
        path:   Path         = HYBRID_CKPT_PATH,
        device: torch.device = DEVICE,
        **model_kwargs,
    ) -> HybridModel:
        import numpy as np
        torch.serialization.add_safe_globals([np.ndarray, np.dtype, np.float32, np.float64, np.int32, np.int64])
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = HybridModel(
            n_users   = ckpt["n_users"],
            n_items   = ckpt["n_items"],
            embed_dim = ckpt["embed_dim"],
            **model_kwargs,
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        log.info(f"Hybrid model loaded ← {path}  "
                 f"(best val RMSE={ckpt['best_val_rmse']:.5f})")
        return model


# ── Inference (on HybridModel directly) ──────────────────────────────────────

@torch.no_grad()
def score_all_items(
    model:       HybridModel,
    user_idx:    int,
    sbert_emb:   torch.Tensor,   # (n_items, 384)
    imdb_feats:  torch.Tensor,   # (n_items, 23)
    popularity:  torch.Tensor,   # (n_items,)
    history_emb: torch.Tensor,   # (n_users, 384)
    device:      torch.device = DEVICE,
    batch_size:  int = 2048,
) -> np.ndarray:
    """
    Score all items for a single user efficiently.

    Processes items in batches to avoid OOM on large item catalogues.
    Returns (n_items,) numpy array of predicted mean-centred ratings.

    Used by the evaluator to compute NDCG@K and Recall@K — we need
    scores for all items to rank them, not just the ground-truth item.
    """
    model.eval()
    n_items   = sbert_emb.shape[0]
    all_scores = np.empty(n_items, dtype=np.float32)

    # User features — same for every item batch
    u_tensor  = torch.tensor([user_idx], dtype=torch.long, device=device)
    hist      = history_emb[user_idx].unsqueeze(0).to(device, non_blocking=True)  # (1, 384)

    for start in range(0, n_items, batch_size):
        end      = min(start + batch_size, n_items)
        item_ids = torch.arange(start, end, dtype=torch.long, device=device)
        B        = end - start

        sbert  = sbert_emb[start:end].to(device, non_blocking=True)     # (B, 384)
        imdb   = imdb_feats[start:end].to(device, non_blocking=True)    # (B, 23)
        pop    = popularity[start:end].unsqueeze(1).to(device, non_blocking=True)  # (B, 1)
        hist_b = hist.expand(B, -1)                                       # (B, 384)
        u_b    = u_tensor.expand(B)                                       # (B,)

        scores = model(u_b, item_ids, sbert, imdb, hist_b, pop)
        all_scores[start:end] = scores.cpu().numpy()

    return all_scores


@torch.no_grad()
def recommend(
    model:       HybridModel,
    user_idx:    int,
    sbert_emb:   torch.Tensor,
    imdb_feats:  torch.Tensor,
    popularity:  torch.Tensor,
    history_emb: torch.Tensor,
    n:           int = 10,
    seen_items:  set | None = None,
    device:      torch.device = DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recommend top-n items for a user.

    Parameters
    ----------
    seen_items : set of movie_idx to exclude (user's training history)

    Returns
    -------
    top_items  : (n,) array of movie_idx
    top_scores : (n,) array of predicted scores
    """
    scores = score_all_items(
        model, user_idx, sbert_emb, imdb_feats,
        popularity, history_emb, device,
    )

    if seen_items:
        scores[list(seen_items)] = -np.inf

    top_idx    = np.argpartition(scores, -n)[-n:]
    top_idx    = top_idx[np.argsort(scores[top_idx])[::-1]]
    top_scores = scores[top_idx]

    return top_idx, top_scores


def hybrid_repr(model: HybridModel) -> str:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"HybridModel("
        f"n_users={model.user_cf.num_embeddings}, "
        f"n_items={model.item_cf.num_embeddings}, "
        f"embed_dim={model.embed_dim}, "
        f"params={n_params:,})"
    )


# Attach __repr__ to the class
HybridModel.__repr__ = hybrid_repr


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from src.models.als import ALS
    from src.models.bpr import BPR

    log.info("Loading feature tensors...")
    sbert_emb   = torch.load(SBERT_EMBEDDINGS_PATH,   weights_only=True)
    imdb_feats  = torch.load(IMDB_FEATURES_PATH,      weights_only=True)
    popularity  = torch.load(POPULARITY_PATH,          weights_only=True)
    history_emb = torch.load(HISTORY_EMBEDDINGS_PATH, weights_only=True)

    log.info("Loading train/val splits...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    n_users = int(train_df["user_idx"].max()) + 1
    n_items = int(train_df["movie_idx"].max()) + 1

    log.info("Building HybridModel...")
    hybrid_model = HybridModel(n_users=n_users, n_items=n_items)
    log.info(repr(hybrid_model))

    # Load and inject pre-trained CF factors
    als_path = RESULTS_DIR / "als_factors.npz"
    bpr_path = RESULTS_DIR / "bpr_factors.npz"

    if als_path.exists() and bpr_path.exists():
        als = ALS.load(als_path)
        bpr = BPR.load(bpr_path, device=torch.device("cpu"))
        hybrid_model.load_cf_weights(
            user_factors = als.get_user_factors_tensor(),
            item_factors = als.get_item_factors_tensor(),
            user_bpr     = bpr.get_user_embeddings_tensor(),
            item_bpr     = bpr.get_item_embeddings_tensor(),
        )
    else:
        log.warning("ALS/BPR factors not found — training from random init")

    trainer = HybridTrainer(
        model       = hybrid_model,
        sbert_emb   = sbert_emb,
        imdb_feats  = imdb_feats,
        popularity  = popularity,
        history_emb = history_emb,
    )
    trainer.fit(train_df, val_df)