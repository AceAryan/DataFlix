"""
DataFlix — Model 3: Hybrid CF + Content  (corrected)
src/models/hybrid.py

Architecture:
  User side : [ALS+BPR factors (2k)] + [history embedding (384)] → fusion → d-dim
  Item side : [ALS+BPR factors (2k)] + [SBERT (384)] + [IMDB (23)] + [log-pop (1)]
              → cross-attention fusion → d-dim
  Score     : MLP( [user_repr, item_repr, dot_product] ) → scalar
            + learnable-gated BPR residual

Fixes vs previous version
--------------------------
FIX 1 — Gated BPR residual
    Old : return bpr_score + mlp_score          (raw sum, different scales)
    New : return mlp_score + gate * bpr_score   (gate is a learned sigmoid scalar)
    Why : BPR dot-product scores and MLP outputs live on different numerical scales.
          A raw sum means whichever is larger dominates. The gate lets the model
          learn how much to trust pure-CF vs the content-enhanced MLP.

FIX 2 — CF stays frozen for the entire training run
    Old : freeze for freeze_epochs, then unfreeze
    New : CF embeddings are always frozen (requires_grad=False permanently)
    Why : Once you unfreeze CF, noisy content gradients corrupt well-trained
          ALS+BPR weights. The CF tower is already trained; the hybrid should
          only learn to *use* it, not re-train it.

FIX 3 — Log-scaled popularity
    Old : self.proj_pop = nn.Linear(1, embed_dim)  fed raw popularity count
    New : log1p(pop) normalised to [0,1] before projection
    Why : Popularity follows a power-law. Raw values make the model learn
          "recommend popular items" instead of personalised ranking.

FIX 4 — Cross-attention item fusion (content features gate on CF)
    Old : item_fusion = MLP(concat(cf, sbert, imdb, pop))  — equal weight always
    New : CF repr is the query; content reprs are keys/values.
          Attention weights show how much each content source adjusts the CF repr.
    Why : A movie with sparse IMDB data should rely more on CF; a new movie with
          rich metadata should rely more on content. Attention learns this per-item.

FIX 5 — Recency-aware user history (learned positional decay)
    Old : proj_history(mean_pooled_sbert)  — collapses all history to one vector
    New : per-interaction SBERT embeddings weighted by learned exponential decay
          over recency rank, then projected.
    Why : Recent interactions are more predictive of current taste. A single
          mean-pool treats a movie watched 5 years ago the same as last week.
"""

import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from src.config import (
        PROCESSED_DIR, RESULTS_DIR, DEVICE,
        TRAIN_CSV, VAL_CSV,
        BPR_DATA_PATH, USER_POSITIVES_PATH,
        SBERT_EMBEDDINGS_PATH, IMDB_FEATURES_PATH,
        POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH,
        LATENT_DIM_K, EMBED_DIM_D, NUM_HEADS, MLP_HIDDEN, DROPOUT,
        SBERT_DIM, IMDB_FEAT_DIM,
        LR_HYBRID, HYBRID_WEIGHT_DECAY, COSINE_T_MAX,
        HYBRID_EPOCHS, HYBRID_BATCH_SIZE, HYBRID_SAMPLES_PER_EPOCH,
        EARLY_STOP_PATIENCE, FREEZE_EPOCHS,
        HYBRID_CKPT_PATH, RELEVANCE_RATING,
    )
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR              = _ROOT / "data/processed"
    RESULTS_DIR                = _ROOT / "results"
    DEVICE                     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_CSV                  = PROCESSED_DIR / "train.csv"
    VAL_CSV                    = PROCESSED_DIR / "val.csv"
    BPR_DATA_PATH              = PROCESSED_DIR / "bpr_data.npz"
    USER_POSITIVES_PATH        = PROCESSED_DIR / "user_positives.pkl"
    SBERT_EMBEDDINGS_PATH      = PROCESSED_DIR / "sbert_embeddings.pt"
    IMDB_FEATURES_PATH         = PROCESSED_DIR / "imdb_features.pt"
    POPULARITY_PATH            = PROCESSED_DIR / "popularity.pt"
    HISTORY_EMBEDDINGS_PATH    = PROCESSED_DIR / "history_embeddings.pt"
    LATENT_DIM_K               = 128
    EMBED_DIM_D                = 256
    NUM_HEADS                  = 4
    MLP_HIDDEN                 = [512, 128]
    DROPOUT                    = 0.2
    SBERT_DIM                  = 384
    IMDB_FEAT_DIM              = 23
    LR_HYBRID                  = 1e-3
    HYBRID_WEIGHT_DECAY        = 1e-4
    COSINE_T_MAX               = 30
    HYBRID_EPOCHS              = 50
    HYBRID_BATCH_SIZE          = 4096
    HYBRID_SAMPLES_PER_EPOCH   = 200_000
    EARLY_STOP_PATIENCE        = 7
    FREEZE_EPOCHS              = 5          # kept in config but no longer used for unfreeze
    HYBRID_CKPT_PATH           = RESULTS_DIR / "hybrid_best.pt"
    RELEVANCE_RATING           = 4.0

# Maximum history length used for FIX 5 recency weighting.
# Interactions beyond this rank are still included but get near-zero weight.
MAX_HIST_LEN = 50

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Triplet Dataset  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    (user, pos_item, neg_item) triples for BPR loss.
    pos_item : item rated >= RELEVANCE_RATING
    neg_item : popularity-weighted sample not in user's positive set
    """
    def __init__(self, pos_sets, all_items, item_pop, n_samples, eligible):
        pop             = item_pop.astype(np.float64)
        self.probs      = pop / pop.sum()
        self.pos_sets   = pos_sets
        self.all_items  = all_items
        self.eligible   = eligible

        self.users     = torch.empty(n_samples, dtype=torch.long)
        self.pos_items = torch.empty(n_samples, dtype=torch.long)
        self.neg_items = torch.empty(n_samples, dtype=torch.long)
        self._resample(n_samples)

    def _resample(self, n):
        users = np.random.choice(self.eligible, size=n)
        for k, u in enumerate(users):
            pos_set = self.pos_sets[int(u)]
            self.users[k]     = int(u)
            self.pos_items[k] = int(np.random.choice(list(pos_set)))
            for _ in range(20):
                neg = int(np.random.choice(self.all_items, p=self.probs))
                if neg not in pos_set:
                    break
            self.neg_items[k] = neg

    def __len__(self): return len(self.users)
    def __getitem__(self, i): return self.users[i], self.pos_items[i], self.neg_items[i]


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: list, out_dim: int, dropout: float) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ContentCrossAttention(nn.Module):
    """
    FIX 4 — Cross-attention item fusion.

    CF repr  → query   (what the CF model expects)
    Content reprs → keys / values  (what content can add)

    The attention score for each content stream tells us how much it
    should shift the CF-based item representation. If IMDB features are
    weak for this item, attention weight → 0 and CF dominates.
    """
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_q     = nn.LayerNorm(d)
        self.norm_kv    = nn.LayerNorm(d)
        self.norm_out   = nn.LayerNorm(d)
        self.ff         = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 2, d)
        )
        self.norm_ff    = nn.LayerNorm(d)

    def forward(self, cf_repr: torch.Tensor, content_reprs: torch.Tensor) -> torch.Tensor:
        """
        cf_repr      : (B, d)       — CF embedding as query
        content_reprs: (B, n_src, d) — stacked content stream embeddings as K/V
        Returns      : (B, d)       — content-adjusted item representation
        """
        q   = self.norm_q(cf_repr).unsqueeze(1)        # (B, 1, d)
        kv  = self.norm_kv(content_reprs)              # (B, n_src, d)
        out, _ = self.cross_attn(q, kv, kv)            # (B, 1, d)
        out = out.squeeze(1)                            # (B, d)
        # Residual: blend CF query with attended content
        fused = self.norm_out(cf_repr + out)            # (B, d)
        return self.norm_ff(fused + self.ff(fused))     # (B, d)


class RecencyWeightedHistory(nn.Module):
    """
    FIX 5 — Recency-aware user history encoder.

    Learns an exponential decay over interaction recency rank so that
    recent items contribute more to the user representation.

    decay_logit is a scalar parameter; actual weight for rank r is:
        w_r = sigmoid(decay_logit) ^ r    (r=0 is most recent)
    This is initialised near 0.5 so early training behaves like mean-pool.
    """
    def __init__(self, sbert_dim: int, embed_dim: int, max_len: int = MAX_HIST_LEN):
        super().__init__()
        self.max_len     = max_len
        self.proj        = nn.Linear(sbert_dim, embed_dim)
        self.norm        = nn.LayerNorm(embed_dim)
        # Learned decay base; init at 0 → sigmoid(0)=0.5 (mild decay)
        self.decay_logit = nn.Parameter(torch.zeros(1))

    def forward(self, history_emb: torch.Tensor) -> torch.Tensor:
        """
        history_emb : (B, max_len, sbert_dim)  — padded, most-recent-first
                      OR (B, sbert_dim)  — pre-aggregated fallback (mean pool)
        Returns     : (B, embed_dim)
        """
        if history_emb.dim() == 2:
            # Fallback: pre-aggregated mean-pool vector (backward compat)
            return self.norm(self.proj(history_emb))

        # history_emb: (B, L, sbert_dim)
        L    = history_emb.size(1)
        base = torch.sigmoid(self.decay_logit)                    # scalar in (0,1)
        # ranks: 0=most recent, L-1=oldest
        ranks  = torch.arange(L, device=history_emb.device).float()   # (L,)
        weights = base ** ranks                                    # (L,) — decaying
        weights = weights / (weights.sum() + 1e-8)                # normalise
        weights = weights.unsqueeze(0).unsqueeze(-1)              # (1, L, 1)

        aggregated = (history_emb * weights).sum(dim=1)           # (B, sbert_dim)
        return self.norm(self.proj(aggregated))                    # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
#  HybridModel
# ─────────────────────────────────────────────────────────────────────────────

class HybridModel(nn.Module):

    def __init__(
        self,
        n_users:    int,
        n_items:    int,
        n_factors:  int   = LATENT_DIM_K,
        embed_dim:  int   = EMBED_DIM_D,
        n_heads:    int   = NUM_HEADS,
        mlp_hidden: list  = MLP_HIDDEN,
        dropout:    float = DROPOUT,
        sbert_dim:  int   = SBERT_DIM,
        imdb_dim:   int   = IMDB_FEAT_DIM,
        max_hist_len: int = MAX_HIST_LEN,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.n_factors   = n_factors
        cf_dim           = n_factors * 2   # ALS + BPR concatenated

        # ── CF embeddings (FIX 2: always frozen after load_cf_weights) ──
        self.user_cf = nn.Embedding(n_users, cf_dim)
        self.item_cf = nn.Embedding(n_items, cf_dim)

        # ── Projections → embed_dim ──
        self.proj_user_cf = nn.Linear(cf_dim,    embed_dim)
        self.proj_item_cf = nn.Linear(cf_dim,    embed_dim)
        self.proj_sbert   = nn.Linear(sbert_dim, embed_dim)
        self.proj_imdb    = nn.Linear(imdb_dim,  embed_dim)
        self.proj_pop     = nn.Linear(1,         embed_dim)  # receives log-normed pop

        # Layer norms on projected content
        self.ln_item_cf = nn.LayerNorm(embed_dim)
        self.ln_sbert   = nn.LayerNorm(embed_dim)
        self.ln_imdb    = nn.LayerNorm(embed_dim)
        self.ln_pop     = nn.LayerNorm(embed_dim)
        self.ln_user_cf = nn.LayerNorm(embed_dim)

        # ── FIX 5: recency-aware history encoder ──
        self.history_encoder = RecencyWeightedHistory(sbert_dim, embed_dim, max_hist_len)

        # ── User fusion: CF + history → embed_dim ──
        self.user_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── FIX 4: cross-attention item fusion ──
        # CF is the query; [SBERT, IMDB, pop] are the content keys/values
        self.item_cross_attn = ContentCrossAttention(embed_dim, n_heads, dropout)

        # ── Scoring MLP ──
        self.score_mlp = _mlp(embed_dim * 2 + 1, mlp_hidden, 1, dropout)

        # ── FIX 1: learnable BPR residual gate ──
        # Initialised to 0 → gate = sigmoid(0) = 0.5 (balanced starting point)
        self.bpr_gate = nn.Parameter(torch.zeros(1))

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        s = 1.0 / np.sqrt(self.embed_dim)
        nn.init.normal_(self.user_cf.weight, 0, s)
        nn.init.normal_(self.item_cf.weight, 0, s)
        for m in [self.proj_user_cf, self.proj_item_cf,
                  self.proj_sbert, self.proj_imdb, self.proj_pop]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    # ── CF weight loading + permanent freeze (FIX 2) ─────────────────────────

    def load_cf_weights(self, uf, if_, ubpr, ibpr):
        """Load pre-trained ALS + BPR factors and permanently freeze them."""
        with torch.no_grad():
            self.user_cf.weight.copy_(torch.cat([uf,  ubpr], dim=1))
            self.item_cf.weight.copy_(torch.cat([if_, ibpr], dim=1))
        # FIX 2: freeze permanently — content gradients must not corrupt CF
        self.user_cf.weight.requires_grad_(False)
        self.item_cf.weight.requires_grad_(False)
        log.info("Hybrid: CF weights loaded from ALS+BPR and permanently frozen.")

    # ── Encoders ─────────────────────────────────────────────────────────────

    def encode_user(self, user_idx: torch.Tensor, history_emb: torch.Tensor) -> torch.Tensor:
        """
        user_idx    : (B,)
        history_emb : (B, L, sbert_dim)  or  (B, sbert_dim)
        Returns     : (B, embed_dim)
        """
        u_cf   = self.ln_user_cf(self.proj_user_cf(self.user_cf(user_idx)))  # (B, d)
        u_hist = self.history_encoder(history_emb)                            # (B, d)
        return self.user_fusion(torch.cat([u_cf, u_hist], dim=1))             # (B, d)

    def encode_item(
        self,
        item_idx:  torch.Tensor,   # (B,)
        sbert_emb: torch.Tensor,   # (B, sbert_dim)
        imdb_feat: torch.Tensor,   # (B, imdb_dim)
        pop_norm:  torch.Tensor,   # (B, 1)  — log-normalised (FIX 3)
    ) -> torch.Tensor:             # (B, embed_dim)
        # Project CF
        i_cf  = self.ln_item_cf(self.proj_item_cf(self.item_cf(item_idx)))  # (B, d)
        # Project content streams
        i_sb  = self.ln_sbert(self.proj_sbert(sbert_emb))                   # (B, d)
        i_im  = self.ln_imdb(self.proj_imdb(imdb_feat))                     # (B, d)
        i_pop = self.ln_pop(self.proj_pop(pop_norm))                        # (B, d)
        # FIX 4: cross-attention — CF queries content, learns to ignore weak features
        content = torch.stack([i_sb, i_im, i_pop], dim=1)                   # (B, 3, d)
        return self.item_cross_attn(i_cf, content)                           # (B, d)

    def score(self, user_repr: torch.Tensor, item_repr: torch.Tensor) -> torch.Tensor:
        """MLP scorer. (B, d), (B, d) → (B,)"""
        dot = (user_repr * item_repr).sum(dim=1, keepdim=True)
        return self.score_mlp(torch.cat([user_repr, item_repr, dot], dim=1)).squeeze(1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        user_idx:    torch.Tensor,   # (B,)
        item_idx:    torch.Tensor,   # (B,)
        sbert_emb:   torch.Tensor,   # (B, sbert_dim)
        imdb_feat:   torch.Tensor,   # (B, imdb_dim)
        history_emb: torch.Tensor,   # (B, L, sbert_dim) or (B, sbert_dim)
        pop_norm:    torch.Tensor,   # (B, 1) — log-normalised
    ) -> torch.Tensor:               # (B,)
        u_repr = self.encode_user(user_idx, history_emb)
        i_repr = self.encode_item(item_idx, sbert_emb, imdb_feat, pop_norm)

        mlp_score = self.score(u_repr, i_repr)                              # (B,)

        # FIX 1: gated BPR residual — gate is learned, starts at 0.5
        u_bpr     = self.user_cf(user_idx)[:, self.n_factors:]              # (B, K)
        i_bpr     = self.item_cf(item_idx)[:, self.n_factors:]              # (B, K)
        bpr_score = (u_bpr * i_bpr).sum(dim=1)                             # (B,)
        gate      = torch.sigmoid(self.bpr_gate)                            # scalar
        return mlp_score + gate * bpr_score

    def __repr__(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (f"HybridModel(n_users={self.user_cf.num_embeddings}, "
                f"n_items={self.item_cf.num_embeddings}, "
                f"embed_dim={self.embed_dim}, trainable_params={n:,})")


# ─────────────────────────────────────────────────────────────────────────────
#  HybridTrainer
# ─────────────────────────────────────────────────────────────────────────────

def _log_normalise_pop(pop: torch.Tensor) -> torch.Tensor:
    """
    FIX 3 — Log-scale + min-max normalise popularity to [0, 1].
    Called once at trainer init; result stored as self.pop_norm.
    """
    lp  = torch.log1p(pop.float())
    mn, mx = lp.min(), lp.max()
    return ((lp - mn) / (mx - mn + 1e-8)).unsqueeze(1)   # (n_items, 1)


class HybridTrainer:

    def __init__(
        self,
        model:             HybridModel,
        sbert_emb:         torch.Tensor,
        imdb_feats:        torch.Tensor,
        popularity:        torch.Tensor,
        history_emb:       torch.Tensor,
        all_items:         np.ndarray,
        item_pop:          np.ndarray,
        device:            torch.device = DEVICE,
        lr:                float        = LR_HYBRID,
        weight_decay:      float        = HYBRID_WEIGHT_DECAY,
        n_epochs:          int          = HYBRID_EPOCHS,
        batch_size:        int          = HYBRID_BATCH_SIZE,
        samples_per_epoch: int          = HYBRID_SAMPLES_PER_EPOCH,
        patience:          int          = EARLY_STOP_PATIENCE,
        freeze_epochs:     int          = FREEZE_EPOCHS,   # kept for API compat, unused
    ):
        self.model             = model.to(device)
        self.device            = device
        self.n_epochs          = n_epochs
        self.batch_size        = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.patience          = patience
        self.all_items         = all_items
        self.item_pop          = item_pop

        # Feature tensors — stay on CPU, moved per-batch
        self.sbert_emb   = sbert_emb
        self.imdb_feats  = imdb_feats
        self.history_emb = history_emb
        # FIX 3: pre-compute log-normalised popularity once
        self.pop_norm    = _log_normalise_pop(popularity)   # (n_items, 1) CPU

        # Optimiser only touches non-frozen params
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=COSINE_T_MAX, eta_min=1e-5
        )
        self.best_val_loss           = float("inf")
        self.patience_counter        = 0
        self.train_loss_history: list = []
        self.val_loss_history:   list = []

    # ── Feature helpers ───────────────────────────────────────────────────────

    def _item_feats(self, item_idx: torch.Tensor):
        cpu = item_idx.cpu()
        s   = self.sbert_emb[cpu].to(self.device, non_blocking=True)
        im  = self.imdb_feats[cpu].to(self.device, non_blocking=True)
        p   = self.pop_norm[cpu].to(self.device, non_blocking=True)   # already (N,1)
        return s, im, p

    def _user_feats(self, user_idx: torch.Tensor) -> torch.Tensor:
        return self.history_emb[user_idx.cpu()].to(self.device, non_blocking=True)

    # ── BPR step ─────────────────────────────────────────────────────────────

    def _bpr_step(self, u, pos, neg) -> torch.Tensor:
        hist  = self._user_feats(u)
        u_repr = self.model.encode_user(u, hist)

        s_pos, si_pos, p_pos = self._item_feats(pos)
        s_neg, si_neg, p_neg = self._item_feats(neg)

        i_pos_repr = self.model.encode_item(pos, s_pos, si_pos, p_pos)
        i_neg_repr = self.model.encode_item(neg, s_neg, si_neg, p_neg)

        # FIX 1: include gated BPR residual in training loss too
        u_bpr     = self.model.user_cf(u)[:, self.model.n_factors:]
        gate      = torch.sigmoid(self.model.bpr_gate)

        def _score(u_r, i_r, i_idx):
            mlp  = self.model.score(u_r, i_r)
            i_b  = self.model.item_cf(i_idx)[:, self.model.n_factors:]
            bpr  = (u_bpr * i_b).sum(dim=1)
            return mlp + gate * bpr

        s_p = _score(u_repr, i_pos_repr, pos)
        s_n = _score(u_repr, i_neg_repr, neg)
        return -torch.nn.functional.logsigmoid(s_p - s_n).mean()

    # ── Epoch loop ────────────────────────────────────────────────────────────

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total, n = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for u, pos, neg in loader:
                u   = u.to(self.device)
                pos = pos.to(self.device)
                neg = neg.to(self.device)
                loss = self._bpr_step(u, pos, neg)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                total += loss.item() * len(u)
                n     += len(u)
        return total / n if n > 0 else float("inf")

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self) -> "HybridTrainer":
        log.info(f"  Building positive sets (rating >= {RELEVANCE_RATING})...")
        train_df = pd.read_csv(TRAIN_CSV)
        val_df   = pd.read_csv(VAL_CSV)

        def _pos_sets(df):
            ps = defaultdict(set)
            for row in df[df["rating"] >= RELEVANCE_RATING].itertuples(index=False):
                ps[int(row.user_idx)].add(int(row.movie_idx))
            return dict(ps)

        train_pos = _pos_sets(train_df)
        val_pos   = _pos_sets(val_df)

        eligible_train = np.array([u for u, s in train_pos.items() if s], dtype=np.int32)
        eligible_val   = np.array([u for u, s in val_pos.items()   if s], dtype=np.int32)
        log.info(f"  {len(eligible_train):,} train users | {len(eligible_val):,} val users")

        # FIX 2: CF already frozen from load_cf_weights — no phase switching needed
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(f"\nHybrid training: {self.n_epochs} epochs | "
                 f"batch={self.batch_size} | device={self.device} | "
                 f"trainable params={trainable_count:,}  (CF frozen)")
        log.info(f"  BPR gate init: {torch.sigmoid(self.model.bpr_gate).item():.3f}")

        for epoch in range(1, self.n_epochs + 1):
            t = time.time()

            train_ds = TripletDataset(
                train_pos, self.all_items, self.item_pop,
                self.samples_per_epoch, eligible_train,
            )
            val_ds = TripletDataset(
                val_pos, self.all_items, self.item_pop,
                max(10_000, self.samples_per_epoch // 10), eligible_val,
            )
            tr_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                   shuffle=True, num_workers=0, pin_memory=True)
            va_loader = DataLoader(val_ds,   batch_size=self.batch_size * 2,
                                   shuffle=False, num_workers=0)

            tr_loss = self._run_epoch(tr_loader, train=True)
            va_loss = self._run_epoch(va_loader, train=False)
            self.scheduler.step()

            self.train_loss_history.append(tr_loss)
            self.val_loss_history.append(va_loss)

            gate_val = torch.sigmoid(self.model.bpr_gate).item()
            lr_now   = self.scheduler.get_last_lr()[0]
            log.info(f"  Epoch {epoch:>3}/{self.n_epochs}  "
                     f"train={tr_loss:.5f}  val={va_loss:.5f}  "
                     f"gate={gate_val:.3f}  lr={lr_now:.2e}  "
                     f"({time.time()-t:.1f}s)")

            if va_loss < self.best_val_loss:
                self.best_val_loss    = va_loss
                self.patience_counter = 0
                self.save(HYBRID_CKPT_PATH)
                log.info(f"    ✓ Best val={va_loss:.5f} — saved")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    log.info(f"  Early stop at epoch {epoch}")
                    break

        log.info(f"\nBest val loss  : {self.best_val_loss:.5f}")
        log.info(f"Final BPR gate : {torch.sigmoid(self.model.bpr_gate).item():.3f}")
        return self

    # ── Persistence ───────────────────────────────────────────────────────────

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
    def load_model(cls, path=HYBRID_CKPT_PATH, device=DEVICE, **kw) -> HybridModel:
        ck = torch.load(path, map_location=device, weights_only=False)
        m  = HybridModel(n_users=ck["n_users"], n_items=ck["n_items"],
                         embed_dim=ck["embed_dim"], **kw)
        m.load_state_dict(ck["model_state"])
        m.to(device).eval()
        log.info(f"Hybrid loaded ← {path}  (best val={ck['best_val_loss']:.5f})")
        return m


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_all_items(
    model:       HybridModel,
    user_idx:    int,
    sbert_emb:   torch.Tensor,
    imdb_feats:  torch.Tensor,
    popularity:  torch.Tensor,
    history_emb: torch.Tensor,
    device:      torch.device = DEVICE,
    batch_size:  int          = 1024,
) -> np.ndarray:
    """
    Score all items for one user in batches.
    popularity should be the RAW popularity tensor (log-norm applied here).
    """
    model.eval()
    n_items    = sbert_emb.shape[0]
    all_scores = np.empty(n_items, dtype=np.float32)

    # FIX 3: log-normalise popularity at inference too
    pop_norm = _log_normalise_pop(popularity)   # (n_items, 1) CPU

    # User repr — computed once
    u_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
    hist_row = history_emb[user_idx]
    if hist_row.device.type != device.type:
        hist_row = hist_row.to(device, non_blocking=True)
    # Support both (sbert_dim,) and (L, sbert_dim) history shapes
    hist = hist_row.unsqueeze(0) if hist_row.dim() == 1 else hist_row.unsqueeze(0)
    u_repr = model.encode_user(u_tensor, hist)                   # (1, d)

    # BPR user factors — for FIX 1 gate
    u_bpr  = model.user_cf(u_tensor)[:, model.n_factors:]        # (1, K)
    gate   = torch.sigmoid(model.bpr_gate)                       # scalar

    for start in range(0, n_items, batch_size):
        end      = min(start + batch_size, n_items)
        B        = end - start
        item_ids = torch.arange(start, end, dtype=torch.long, device=device)

        s  = sbert_emb[start:end].to(device, non_blocking=True)
        im = imdb_feats[start:end].to(device, non_blocking=True)
        p  = pop_norm[start:end].to(device, non_blocking=True)   # (B, 1)

        i_repr    = model.encode_item(item_ids, s, im, p)        # (B, d)
        mlp_score = model.score(u_repr.expand(B, -1), i_repr)    # (B,)

        # FIX 1: gated BPR residual
        i_bpr     = model.item_cf(item_ids)[:, model.n_factors:] # (B, K)
        bpr_score = (u_bpr.expand(B, -1) * i_bpr).sum(dim=1)    # (B,)

        all_scores[start:end] = (mlp_score + gate * bpr_score).cpu().numpy()

    return all_scores