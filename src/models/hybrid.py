"""
DataFlix — Model 3: Hybrid CF + Content Model
src/models/hybrid.py

Combines four information streams into a unified scoring model:

  Stream 1 — ALS latent factors        (collaborative, k-dim)
  Stream 2 — BPR embeddings            (ranking-optimised CF, k-dim)
  Stream 3 — SBERT synopsis embeddings (semantic content, 384-dim)
  Stream 4 — IMDB structured features  (genre OHE + runtime + votes, 23-dim)

Training objective — BPR pairwise ranking loss:
    L = -mean( log σ( s(u,i) - s(u,j) ) )

    where i is a positive item (user rated >= 4.0) and j is a negative
    item sampled by popularity. This directly optimises ranking quality
    (what NDCG and Recall measure) rather than rating prediction (MSE).

    The previous MSE objective caused the model to stop at epoch ~12 and
    produce worse ranking metrics than BPR alone — content features were
    learning to predict ratings, not to rank. Switching to BPR loss means
    content features directly contribute to ranking signal.

Two-phase training:
    Phase 1 (freeze_epochs): CF embeddings frozen — only content
    projections and attention head train. Prevents random gradients
    from corrupting well-trained CF factors early on.

    Phase 2 (remaining): Everything unfreezes for end-to-end fine-tuning.
"""

import logging
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from src.config import (
        PROCESSED_DIR, RESULTS_DIR, DEVICE,
        TRAIN_CSV, VAL_CSV,
        USER_POSITIVES_PATH, BPR_DATA_PATH,
        SBERT_EMBEDDINGS_PATH, IMDB_FEATURES_PATH,
        POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH,
        LATENT_DIM_K, EMBED_DIM_D, NUM_HEADS,
        MLP_HIDDEN, DROPOUT, SBERT_DIM, IMDB_FEAT_DIM,
        LR_PATH_B, WEIGHT_DECAY, COSINE_T_MAX,
        EARLY_STOP_PATIENCE, MAX_EPOCHS, BATCH_SIZE,
        BPR_SAMPLES_PER_EPOCH,
    )
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR           = _ROOT / "data/processed"
    RESULTS_DIR             = _ROOT / "results"
    DEVICE                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_CSV               = PROCESSED_DIR / "train.csv"
    VAL_CSV                 = PROCESSED_DIR / "val.csv"
    USER_POSITIVES_PATH     = PROCESSED_DIR / "user_positives.pkl"
    BPR_DATA_PATH           = PROCESSED_DIR / "bpr_data.npz"
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
    LR_PATH_B               = 1e-3
    WEIGHT_DECAY            = 1e-4
    COSINE_T_MAX            = 50
    EARLY_STOP_PATIENCE     = 7
    MAX_EPOCHS              = 50
    BATCH_SIZE              = 4096
    BPR_SAMPLES_PER_EPOCH   = 200_000

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

HYBRID_CKPT_PATH = RESULTS_DIR / "hybrid_best.pt"


# ── BPR Training Dataset ──────────────────────────────────────────────────────

class BPRTripletDataset(Dataset):
    """
    Samples (user, positive_item, negative_item) triples for BPR training.

    Positive items: movies the user rated >= 4.0 (genuinely liked).
    Negative items: movies not in user's positive set, sampled by popularity.

    Using rating >= 4.0 as positives (not all rated items) gives cleaner
    training signal — we want the model to rank things the user liked above
    things they haven't seen, not things they rated 2 stars.
    """

    def __init__(
        self,
        user_positives: dict[int, set],
        all_items:      np.ndarray,
        item_pop:       np.ndarray,
        n_samples:      int,
        n_users:        int,
    ):
        self.user_positives = user_positives
        self.all_items      = all_items
        self.n_samples      = n_samples

        # Popularity-weighted negative sampling probabilities
        pop = item_pop.astype(np.float64)
        self.item_probs = pop / pop.sum()

        # Only sample from users who have at least one positive
        self.eligible = np.array(
            [u for u in range(n_users) if u in user_positives
             and len(user_positives[u]) > 0],
            dtype=np.int32,
        )

        # Pre-sample all triples for this epoch
        self._sample()

    def _sample(self) -> None:
        """Sample n_samples triples. Called once per epoch in fit()."""
        users     = np.random.choice(self.eligible, size=self.n_samples)
        pos_items = np.empty(self.n_samples, dtype=np.int32)
        neg_items = np.empty(self.n_samples, dtype=np.int32)

        for i, u in enumerate(users):
            pos_set = self.user_positives[int(u)]
            pos_items[i] = int(np.random.choice(list(pos_set)))

            # Popularity-weighted negative with rejection sampling (max 20 tries)
            for _ in range(20):
                neg = int(np.random.choice(self.all_items, p=self.item_probs))
                if neg not in pos_set:
                    break
            neg_items[i] = neg

        self.users     = torch.tensor(users,     dtype=torch.long)
        self.pos_items = torch.tensor(pos_items, dtype=torch.long)
        self.neg_items = torch.tensor(neg_items, dtype=torch.long)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx):
        return self.users[idx], self.pos_items[idx], self.neg_items[idx]


# ── Sub-modules ───────────────────────────────────────────────────────────────

def _make_mlp(in_dim, hidden, out_dim, dropout) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class FeatureFusion(nn.Module):
    """
    Multi-head self-attention over a sequence of d-dim feature vectors.
    Lets the model learn which streams matter most for each user-item pair.

    Input:  (B, n_streams, d)
    Output: (B, d)
    """
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out).mean(dim=1)


# ── Main Model ────────────────────────────────────────────────────────────────

class HybridModel(nn.Module):
    """
    Hybrid CF + Content scoring model.

    Produces a single scalar score s(u,i) for any (user, item) pair.
    Higher score = model predicts user prefers this item.

    Trained with BPR loss: s(u, pos) should be > s(u, neg).
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

        # CF embeddings: ALS (k) + BPR (k) concatenated → (2k)
        self.user_cf = nn.Embedding(n_users, n_factors * 2)
        self.item_cf = nn.Embedding(n_items, n_factors * 2)

        # Content projections → embed_dim
        self.sbert_proj   = nn.Linear(sbert_dim,     embed_dim)
        self.imdb_proj    = nn.Linear(imdb_feat_dim,  embed_dim)
        self.history_proj = nn.Linear(sbert_dim,     embed_dim)
        self.user_cf_proj = nn.Linear(n_factors * 2, embed_dim)
        self.item_cf_proj = nn.Linear(n_factors * 2, embed_dim)
        self.pop_proj     = nn.Linear(1,             embed_dim)

        # LayerNorms
        self.ln_sbert   = nn.LayerNorm(embed_dim)
        self.ln_imdb    = nn.LayerNorm(embed_dim)
        self.ln_history = nn.LayerNorm(embed_dim)
        self.ln_user_cf = nn.LayerNorm(embed_dim)
        self.ln_item_cf = nn.LayerNorm(embed_dim)

        # Attention fusion
        self.user_fusion = FeatureFusion(embed_dim, n_heads, dropout)  # 2 user streams
        self.item_fusion = FeatureFusion(embed_dim, n_heads, dropout)  # 4 item streams

        # Final scoring MLP: [user_repr, item_repr, dot] → scalar
        self.score_mlp = _make_mlp(embed_dim * 2 + 1, mlp_hidden, 1, dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        scale = 1.0 / np.sqrt(self.embed_dim)
        nn.init.normal_(self.user_cf.weight, 0, scale)
        nn.init.normal_(self.item_cf.weight, 0, scale)
        for m in [self.sbert_proj, self.imdb_proj, self.history_proj,
                  self.user_cf_proj, self.item_cf_proj, self.pop_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def load_cf_weights(
        self,
        user_factors: torch.Tensor,
        item_factors: torch.Tensor,
        user_bpr:     torch.Tensor,
        item_bpr:     torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.user_cf.weight.copy_(torch.cat([user_factors, user_bpr], dim=1))
            self.item_cf.weight.copy_(torch.cat([item_factors, item_bpr], dim=1))
        log.info("Hybrid CF weights initialised from ALS + BPR")

    def freeze_cf(self) -> None:
        self.user_cf.weight.requires_grad_(False)
        self.item_cf.weight.requires_grad_(False)
        log.info("CF embeddings frozen")

    def unfreeze_cf(self) -> None:
        self.user_cf.weight.requires_grad_(True)
        self.item_cf.weight.requires_grad_(True)
        log.info("CF embeddings unfrozen")

    def _encode_user(
        self,
        user_idx:    torch.Tensor,   # (B,)
        history_emb: torch.Tensor,   # (B, 384)
    ) -> torch.Tensor:               # (B, d)
        p_u  = self.user_cf(user_idx)
        u_cf = self.ln_user_cf(self.user_cf_proj(p_u))
        u_hi = self.ln_history(self.history_proj(history_emb))
        return self.user_fusion(torch.stack([u_cf, u_hi], dim=1))

    def _encode_item(
        self,
        item_idx:  torch.Tensor,   # (B,)
        sbert_emb: torch.Tensor,   # (B, 384)
        imdb_feat: torch.Tensor,   # (B, 23)
        pop:       torch.Tensor,   # (B, 1)
    ) -> torch.Tensor:             # (B, d)
        q_i    = self.item_cf(item_idx)
        i_cf   = self.ln_item_cf(self.item_cf_proj(q_i))
        i_sb   = self.ln_sbert(self.sbert_proj(sbert_emb))
        i_im   = self.ln_imdb(self.imdb_proj(imdb_feat))
        i_pop  = self.pop_proj(pop)
        return self.item_fusion(torch.stack([i_cf, i_sb, i_im, i_pop], dim=1))

    def forward(
        self,
        user_idx:    torch.Tensor,
        item_idx:    torch.Tensor,
        sbert_emb:   torch.Tensor,
        imdb_feat:   torch.Tensor,
        history_emb: torch.Tensor,
        pop:         torch.Tensor,
    ) -> torch.Tensor:
        """Returns scalar score for each (user, item) pair. Shape: (B,)"""
        u_repr = self._encode_user(user_idx, history_emb)
        i_repr = self._encode_item(item_idx, sbert_emb, imdb_feat, pop)
        dot    = (u_repr * i_repr).sum(dim=1, keepdim=True)
        x      = torch.cat([u_repr, i_repr, dot], dim=1)
        return self.score_mlp(x).squeeze(1)

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (f"HybridModel(n_users={self.user_cf.num_embeddings}, "
                f"n_items={self.item_cf.num_embeddings}, "
                f"embed_dim={self.embed_dim}, params={n:,})")


# ── Trainer ───────────────────────────────────────────────────────────────────

class HybridTrainer:
    """
    Trains HybridModel with BPR pairwise ranking loss.

    The training loop:
      1. Sample (user, pos_item, neg_item) triples each epoch
      2. Score both pos and neg items using HybridModel.forward()
      3. Compute BPR loss: -mean( log σ( score(pos) - score(neg) ) )
      4. Validate on a held-out set of (user, pos_item, neg_item) triples

    Why BPR loss instead of MSE:
      MSE trains the model to predict the exact rating value.
      BPR trains the model to rank positive items above negative items.
      NDCG and Recall measure ranking quality, not rating prediction accuracy.
      A model trained with MSE optimises the wrong objective for ranking tasks.
    """

    def __init__(
        self,
        model:          HybridModel,
        sbert_emb:      torch.Tensor,
        imdb_feats:     torch.Tensor,
        popularity:     torch.Tensor,
        history_emb:    torch.Tensor,
        user_positives: dict[int, set],
        all_items:      np.ndarray,
        item_pop:       np.ndarray,
        device:         torch.device = DEVICE,
        lr:             float        = LR_PATH_B,
        weight_decay:   float        = WEIGHT_DECAY,
        n_epochs:       int          = MAX_EPOCHS,
        batch_size:     int          = BATCH_SIZE,
        samples_per_epoch: int       = BPR_SAMPLES_PER_EPOCH,
        patience:       int          = EARLY_STOP_PATIENCE,
        freeze_epochs:  int          = 5,
    ):
        self.model             = model.to(device)
        self.device            = device
        self.n_epochs          = n_epochs
        self.batch_size        = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.patience          = patience
        self.freeze_epochs     = freeze_epochs

        # Feature tensors on CPU — moved per batch with non_blocking
        self.sbert_emb   = sbert_emb
        self.imdb_feats  = imdb_feats
        self.popularity  = popularity
        self.history_emb = history_emb

        # BPR data
        self.user_positives = user_positives
        self.all_items      = all_items
        self.item_pop       = item_pop
        self.n_users        = model.user_cf.num_embeddings
        self.n_items        = model.item_cf.num_embeddings

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=COSINE_T_MAX, eta_min=1e-5
        )

        self.best_val_loss           = float("inf")
        self.patience_counter        = 0
        self.train_loss_history: list[float] = []
        self.val_loss_history:   list[float] = []

    def _get_item_features(
        self, item_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cpu_idx = item_idx.cpu()
        sbert = self.sbert_emb[cpu_idx].to(self.device, non_blocking=True)
        imdb  = self.imdb_feats[cpu_idx].to(self.device, non_blocking=True)
        pop   = self.popularity[cpu_idx].unsqueeze(1).to(self.device, non_blocking=True)
        return sbert, imdb, pop

    def _get_user_features(
        self, user_idx: torch.Tensor
    ) -> torch.Tensor:
        return self.history_emb[user_idx.cpu()].to(self.device, non_blocking=True)

    def _bpr_loss(
        self,
        user_idx: torch.Tensor,
        pos_idx:  torch.Tensor,
        neg_idx:  torch.Tensor,
    ) -> torch.Tensor:
        """
        BPR loss = -mean( log σ( s(u,pos) - s(u,neg) ) )

        Numerically equivalent to binary cross-entropy with all labels=1:
        the model should score pos higher than neg for every user.
        """
        hist = self._get_user_features(user_idx)

        s_pos_feat = self._get_item_features(pos_idx)
        s_neg_feat = self._get_item_features(neg_idx)

        s_pos = self.model(user_idx, pos_idx, *s_pos_feat, hist)
        s_neg = self.model(user_idx, neg_idx, *s_neg_feat, hist)

        return -torch.nn.functional.logsigmoid(s_pos - s_neg).mean()

    def _run_epoch(
        self,
        loader: DataLoader,
        train:  bool,
    ) -> float:
        self.model.train(train)
        total_loss, total_n = 0.0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for user_idx, pos_idx, neg_idx in loader:
                user_idx = user_idx.to(self.device)
                pos_idx  = pos_idx.to(self.device)
                neg_idx  = neg_idx.to(self.device)

                loss = self._bpr_loss(user_idx, pos_idx, neg_idx)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()

                total_loss += loss.item() * len(user_idx)
                total_n    += len(user_idx)

        return total_loss / total_n if total_n > 0 else float("inf")

    def fit(self) -> "HybridTrainer":
        import time

        # Build positive set for training: only items rated >= 4.0
        # user_positives from BPR data uses all rated items — we need
        # to filter to high-rated items for cleaner BPR training
        log.info("  Building high-rated positive sets (rating >= 4.0)...")
        import pandas as pd
        train_df = pd.read_csv(TRAIN_CSV)
        high_rated = train_df[train_df["rating"] >= 4.0]
        pos_high: dict[int, set] = defaultdict(set)
        for row in high_rated.itertuples(index=False):
            pos_high[int(row.user_idx)].add(int(row.movie_idx))

        n_eligible = sum(1 for v in pos_high.values() if len(v) > 0)
        log.info(f"  {n_eligible:,} users have >= 1 high-rated item")

        # Build val positive set from val split
        val_df   = pd.read_csv(VAL_CSV)
        val_high = val_df[val_df["rating"] >= 4.0]
        val_pos: dict[int, set] = defaultdict(set)
        for row in val_high.itertuples(index=False):
            val_pos[int(row.user_idx)].add(int(row.movie_idx))

        log.info(f"\nHybrid (BPR loss) training: {self.n_epochs} epochs | "
                 f"batch={self.batch_size} | device={self.device}")
        log.info(f"  Phase 1 (frozen CF) : epochs 1–{self.freeze_epochs}")
        log.info(f"  Phase 2 (full)      : epochs {self.freeze_epochs+1}–{self.n_epochs}")

        self.model.freeze_cf()

        for epoch in range(1, self.n_epochs + 1):
            t = time.time()

            if epoch == self.freeze_epochs + 1:
                self.model.unfreeze_cf()

            # Re-sample triples every epoch — fresh negatives each time
            train_dataset = BPRTripletDataset(
                user_positives = pos_high,
                all_items      = self.all_items,
                item_pop       = self.item_pop,
                n_samples      = self.samples_per_epoch,
                n_users        = self.n_users,
            )
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size,
                shuffle=True, num_workers=0, pin_memory=True,
            )

            # Val dataset — smaller, 10% of train samples
            val_dataset = BPRTripletDataset(
                user_positives = val_pos,
                all_items      = self.all_items,
                item_pop       = self.item_pop,
                n_samples      = max(10000, self.samples_per_epoch // 10),
                n_users        = self.n_users,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size * 2,
                shuffle=False, num_workers=0,
            )

            train_loss = self._run_epoch(train_loader, train=True)
            val_loss   = self._run_epoch(val_loader,   train=False)
            self.scheduler.step()

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            phase = "freeze" if epoch <= self.freeze_epochs else "full  "
            lr_now = self.scheduler.get_last_lr()[0]
            log.info(f"  [{phase}] Epoch {epoch:>3}/{self.n_epochs}  "
                     f"train={train_loss:.5f}  val={val_loss:.5f}  "
                     f"lr={lr_now:.2e}  ({time.time()-t:.1f}s)")

            if val_loss < self.best_val_loss:
                self.best_val_loss    = val_loss
                self.patience_counter = 0
                self.save(HYBRID_CKPT_PATH)
                log.info(f"    ✓ Best val loss={val_loss:.5f} — checkpoint saved")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    log.info(f"  Early stopping at epoch {epoch}")
                    break

        log.info(f"\nBest val loss: {self.best_val_loss:.5f}")
        return self

    def save(self, path: Path = HYBRID_CKPT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":   self.model.state_dict(),
            "train_loss":    [float(x) for x in self.train_loss_history],
            "val_loss":      [float(x) for x in self.val_loss_history],
            "best_val_loss": float(self.best_val_loss),
            "n_users":       self.model.user_cf.num_embeddings,
            "n_items":       self.model.item_cf.num_embeddings,
            "embed_dim":     self.model.embed_dim,
        }, path)

    @classmethod
    def load_model(
        cls,
        path:   Path         = HYBRID_CKPT_PATH,
        device: torch.device = DEVICE,
        **model_kwargs,
    ) -> HybridModel:
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = HybridModel(
            n_users   = ckpt["n_users"],
            n_items   = ckpt["n_items"],
            embed_dim = ckpt["embed_dim"],
            **model_kwargs,
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        log.info(f"Hybrid loaded ← {path}  (best val loss={ckpt['best_val_loss']:.5f})")
        return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_all_items(
    model:       HybridModel,
    user_idx:    int,
    sbert_emb:   torch.Tensor,
    imdb_feats:  torch.Tensor,
    popularity:  torch.Tensor,
    history_emb: torch.Tensor,
    device:      torch.device = DEVICE,
    batch_size:  int = 1024,
) -> np.ndarray:
    """
    Score all items for one user in batches.
    Returns (n_items,) numpy array.
    Excludes seen items is handled by caller (evaluate.py).
    """
    model.eval()
    n_items    = sbert_emb.shape[0]
    all_scores = np.empty(n_items, dtype=np.float32)

    u_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
    hist     = history_emb[user_idx].unsqueeze(0).to(device, non_blocking=True)

    for start in range(0, n_items, batch_size):
        end      = min(start + batch_size, n_items)
        item_ids = torch.arange(start, end, dtype=torch.long, device=device)
        B        = end - start

        sbert = sbert_emb[start:end].to(device, non_blocking=True)
        imdb  = imdb_feats[start:end].to(device, non_blocking=True)
        pop   = popularity[start:end].unsqueeze(1).to(device, non_blocking=True)
        hist_b = hist.expand(B, -1)
        u_b    = u_tensor.expand(B)

        all_scores[start:end] = model(u_b, item_ids, sbert, imdb, hist_b, pop).cpu().numpy()

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
    scores = score_all_items(
        model, user_idx, sbert_emb, imdb_feats,
        popularity, history_emb, device,
    )
    if seen_items:
        scores = scores.copy()
        scores[list(seen_items)] = -np.inf
    top_idx    = np.argpartition(scores, -n)[-n:]
    top_idx    = top_idx[np.argsort(scores[top_idx])[::-1]]
    return top_idx, scores[top_idx]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from src.models.als import ALS
    from src.models.bpr import BPR, BPR_FACTORS_PATH

    log.info("Loading feature tensors...")
    sbert_emb   = torch.load(SBERT_EMBEDDINGS_PATH,   weights_only=False)
    imdb_feats  = torch.load(IMDB_FEATURES_PATH,      weights_only=False)
    popularity  = torch.load(POPULARITY_PATH,          weights_only=False)
    history_emb = torch.load(HISTORY_EMBEDDINGS_PATH, weights_only=False)

    log.info("Loading BPR data...")
    bpr_data = np.load(BPR_DATA_PATH)
    all_items = bpr_data["all_items"]
    item_pop  = bpr_data["item_pop_values"]
    with open(USER_POSITIVES_PATH, "rb") as f:
        user_positives = pickle.load(f)

    train_df = pd.read_csv(TRAIN_CSV)
    n_users  = int(train_df["user_idx"].max()) + 1
    n_items  = int(train_df["movie_idx"].max()) + 1

    model = HybridModel(n_users=n_users, n_items=n_items)
    log.info(repr(model))

    als_path = RESULTS_DIR / "als_factors.npz"
    if als_path.exists() and BPR_FACTORS_PATH.exists():
        als = ALS.load(als_path)
        bpr = BPR.load(BPR_FACTORS_PATH, device=torch.device("cpu"))
        model.load_cf_weights(
            user_factors = als.get_user_factors_tensor(),
            item_factors = als.get_item_factors_tensor(),
            user_bpr     = bpr.get_user_embeddings_tensor(),
            item_bpr     = bpr.get_item_embeddings_tensor(),
        )

    trainer = HybridTrainer(
        model          = model,
        sbert_emb      = sbert_emb,
        imdb_feats     = imdb_feats,
        popularity     = popularity,
        history_emb    = history_emb,
        user_positives = user_positives,
        all_items      = all_items,
        item_pop       = item_pop,
    )
    trainer.fit()