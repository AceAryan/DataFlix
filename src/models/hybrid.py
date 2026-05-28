"""
DataFlix — Model 3: Hybrid CF + Content
src/models/hybrid.py

Architecture:
  User side: [ALS+BPR factors (2k)] + [history embedding (384)] → attention → d-dim
  Item side: [ALS+BPR factors (2k)] + [SBERT (384)] + [IMDB (23)] + [pop (1)] → attention → d-dim
  Score: MLP( [user_repr, item_repr, dot_product] ) → scalar

Training loss: BPR pairwise ranking loss (same as BPR model)
  L = -mean( log σ( s(u, pos) - s(u, neg) ) )

Positives: items rated >= 4.0 (cleaner signal than all rated items)
Negatives: popularity-weighted sampling with rejection

Why BPR loss (not MSE):
  The model is evaluated on NDCG/Recall (ranking metrics).
  MSE trains the model to predict ratings accurately.
  These are different objectives. MSE causes early stopping at ~epoch 12
  because content features plateau on rating prediction.
  BPR loss directly optimises what we measure.
"""

import logging
import pickle
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
    FREEZE_EPOCHS              = 5
    HYBRID_CKPT_PATH           = RESULTS_DIR / "hybrid_best.pt"
    RELEVANCE_RATING           = 4.0

log = logging.getLogger(__name__)


# ── Triplet Dataset ───────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    (user, pos_item, neg_item) triples for BPR loss.
    pos_item: item rated >= RELEVANCE_RATING
    neg_item: popularity-weighted sample not in user's positive set
    """
    def __init__(
        self,
        pos_sets:   dict[int, set],
        all_items:  np.ndarray,
        item_pop:   np.ndarray,
        n_samples:  int,
        eligible:   np.ndarray,
    ):
        pop             = item_pop.astype(np.float64)
        self.probs      = pop / pop.sum()
        self.pos_sets   = pos_sets
        self.all_items  = all_items
        self.eligible   = eligible

        # Pre-sample all triples for this epoch
        self.users     = torch.empty(n_samples, dtype=torch.long)
        self.pos_items = torch.empty(n_samples, dtype=torch.long)
        self.neg_items = torch.empty(n_samples, dtype=torch.long)
        self._resample(n_samples)

    def _resample(self, n: int) -> None:
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


# ── Sub-modules ───────────────────────────────────────────────────────────────

def _mlp(in_dim, hidden, out_dim, dropout) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Fusion(nn.Module):
    """Self-attention over n_streams d-dim vectors → mean-pooled d-dim output."""
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_streams, d)
        out, _ = self.attn(x, x, x)
        return self.norm(x + out).mean(dim=1)  # (B, d)


# ── Model ─────────────────────────────────────────────────────────────────────

class HybridModel(nn.Module):

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
        imdb_dim:      int   = IMDB_FEAT_DIM,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        cf_dim         = n_factors * 2  # ALS + BPR concatenated

        # CF embeddings
        self.user_cf = nn.Embedding(n_users, cf_dim)
        self.item_cf = nn.Embedding(n_items, cf_dim)

        # Projections → embed_dim
        self.proj_user_cf   = nn.Linear(cf_dim,    embed_dim)
        self.proj_history   = nn.Linear(sbert_dim, embed_dim)
        self.proj_item_cf   = nn.Linear(cf_dim,    embed_dim)
        self.proj_sbert     = nn.Linear(sbert_dim, embed_dim)
        self.proj_imdb      = nn.Linear(imdb_dim,  embed_dim)
        self.proj_pop       = nn.Linear(1,         embed_dim)

        # Layer norms
        self.ln_user_cf  = nn.LayerNorm(embed_dim)
        self.ln_history  = nn.LayerNorm(embed_dim)
        self.ln_item_cf  = nn.LayerNorm(embed_dim)
        self.ln_sbert    = nn.LayerNorm(embed_dim)
        self.ln_imdb     = nn.LayerNorm(embed_dim)

        # Attention fusion
        self.user_fusion = Fusion(embed_dim, n_heads, dropout)  # 2 streams
        self.item_fusion = Fusion(embed_dim, n_heads, dropout)  # 4 streams

        # Scoring MLP: [u_repr, i_repr, dot] → scalar
        self.score_mlp = _mlp(embed_dim * 2 + 1, mlp_hidden, 1, dropout)

        self._init()

    def _init(self):
        s = 1.0 / np.sqrt(self.embed_dim)
        nn.init.normal_(self.user_cf.weight, 0, s)
        nn.init.normal_(self.item_cf.weight, 0, s)
        for m in [self.proj_user_cf, self.proj_history, self.proj_item_cf,
                  self.proj_sbert, self.proj_imdb, self.proj_pop]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def load_cf_weights(self, uf, if_, ubpr, ibpr):
        with torch.no_grad():
            self.user_cf.weight.copy_(torch.cat([uf,   ubpr], dim=1))
            self.item_cf.weight.copy_(torch.cat([if_,  ibpr], dim=1))
        log.info("Hybrid CF weights loaded from ALS + BPR")

    def freeze_cf(self):
        self.user_cf.weight.requires_grad_(False)
        self.item_cf.weight.requires_grad_(False)
        log.info("CF frozen")

    def unfreeze_cf(self):
        self.user_cf.weight.requires_grad_(True)
        self.item_cf.weight.requires_grad_(True)
        log.info("CF unfrozen")

    def encode_user(
        self,
        user_idx:    torch.Tensor,  # (B,)
        history_emb: torch.Tensor,  # (B, sbert_dim)
    ) -> torch.Tensor:              # (B, embed_dim)
        u_cf   = self.ln_user_cf(self.proj_user_cf(self.user_cf(user_idx)))
        u_hist = self.ln_history(self.proj_history(history_emb))
        return self.user_fusion(torch.stack([u_cf, u_hist], dim=1))

    def encode_item(
        self,
        item_idx:  torch.Tensor,  # (B,)
        sbert_emb: torch.Tensor,  # (B, sbert_dim)
        imdb_feat: torch.Tensor,  # (B, imdb_dim)
        pop:       torch.Tensor,  # (B, 1)
    ) -> torch.Tensor:            # (B, embed_dim)
        i_cf   = self.ln_item_cf(self.proj_item_cf(self.item_cf(item_idx)))
        i_sb   = self.ln_sbert(self.proj_sbert(sbert_emb))
        i_im   = self.ln_imdb(self.proj_imdb(imdb_feat))
        i_pop  = self.proj_pop(pop)
        return self.item_fusion(torch.stack([i_cf, i_sb, i_im, i_pop], dim=1))

    def score(
        self,
        user_repr: torch.Tensor,  # (B, d)
        item_repr: torch.Tensor,  # (B, d)
    ) -> torch.Tensor:            # (B,)
        dot = (user_repr * item_repr).sum(dim=1, keepdim=True)
        return self.score_mlp(torch.cat([user_repr, item_repr, dot], dim=1)).squeeze(1)

    def forward(
        self,
        user_idx:    torch.Tensor,
        item_idx:    torch.Tensor,
        sbert_emb:   torch.Tensor,
        imdb_feat:   torch.Tensor,
        history_emb: torch.Tensor,
        pop:         torch.Tensor,
    ) -> torch.Tensor:
        u = self.encode_user(user_idx, history_emb)
        i = self.encode_item(item_idx, sbert_emb, imdb_feat, pop)
        return self.score(u, i)

    def __repr__(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (f"HybridModel(n_users={self.user_cf.num_embeddings}, "
                f"n_items={self.item_cf.num_embeddings}, "
                f"embed_dim={self.embed_dim}, params={n:,})")


# ── Trainer ───────────────────────────────────────────────────────────────────

class HybridTrainer:

    def __init__(
        self,
        model:          HybridModel,
        sbert_emb:      torch.Tensor,   # (n_items, 384) — CPU
        imdb_feats:     torch.Tensor,   # (n_items, 23)  — CPU
        popularity:     torch.Tensor,   # (n_items,)     — CPU
        history_emb:    torch.Tensor,   # (n_users, 384) — CPU
        all_items:      np.ndarray,
        item_pop:       np.ndarray,
        device:         torch.device = DEVICE,
        lr:             float        = LR_HYBRID,
        weight_decay:   float        = HYBRID_WEIGHT_DECAY,
        n_epochs:       int          = HYBRID_EPOCHS,
        batch_size:     int          = HYBRID_BATCH_SIZE,
        samples_per_epoch: int       = HYBRID_SAMPLES_PER_EPOCH,
        patience:       int          = EARLY_STOP_PATIENCE,
        freeze_epochs:  int          = FREEZE_EPOCHS,
    ):
        self.model             = model.to(device)
        self.device            = device
        self.n_epochs          = n_epochs
        self.batch_size        = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.patience          = patience
        self.freeze_epochs     = freeze_epochs
        self.all_items         = all_items
        self.item_pop          = item_pop

        # Keep feature tensors on CPU; move per batch with non_blocking
        self.sbert_emb   = sbert_emb
        self.imdb_feats  = imdb_feats
        self.popularity  = popularity
        self.history_emb = history_emb

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=COSINE_T_MAX, eta_min=1e-5
        )
        self.best_val_loss           = float("inf")
        self.patience_counter        = 0
        self.train_loss_history: list[float] = []
        self.val_loss_history:   list[float] = []

    def _item_feats(self, item_idx: torch.Tensor):
        """Fetch item feature tensors for a batch. Returns (sbert, imdb, pop)."""
        cpu = item_idx.cpu()
        s   = self.sbert_emb[cpu].to(self.device, non_blocking=True)
        im  = self.imdb_feats[cpu].to(self.device, non_blocking=True)
        p   = self.popularity[cpu].unsqueeze(1).to(self.device, non_blocking=True)
        return s, im, p

    def _user_feats(self, user_idx: torch.Tensor) -> torch.Tensor:
        """Fetch user history embedding for a batch. Returns (B, 384)."""
        # user_idx is on GPU; index CPU tensor then move to GPU
        return self.history_emb[user_idx.cpu()].to(self.device, non_blocking=True)

    def _bpr_step(self, u, pos, neg) -> torch.Tensor:
        hist     = self._user_feats(u)
        u_repr   = self.model.encode_user(u, hist)

        s_pos, si_pos, p_pos = self._item_feats(pos)
        s_neg, si_neg, p_neg = self._item_feats(neg)

        i_pos_repr = self.model.encode_item(pos, s_pos, si_pos, p_pos)
        i_neg_repr = self.model.encode_item(neg, s_neg, si_neg, p_neg)

        score_pos = self.model.score(u_repr, i_pos_repr)
        score_neg = self.model.score(u_repr, i_neg_repr)

        return -torch.nn.functional.logsigmoid(score_pos - score_neg).mean()

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

    def fit(self) -> "HybridTrainer":
        # Build positive sets: only items rated >= RELEVANCE_RATING
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
        n_users   = self.model.user_cf.num_embeddings

        eligible_train = np.array(
            [u for u, s in train_pos.items() if len(s) > 0], dtype=np.int32
        )
        eligible_val = np.array(
            [u for u, s in val_pos.items() if len(s) > 0], dtype=np.int32
        )
        log.info(f"  {len(eligible_train):,} users have >= 1 high-rated train item")

        log.info(f"\nHybrid training: {self.n_epochs} epochs | "
                 f"batch={self.batch_size} | device={self.device}")
        log.info(f"  Phase 1 (frozen CF) : epochs 1–{self.freeze_epochs}")
        log.info(f"  Phase 2 (full)      : epochs {self.freeze_epochs+1}–{self.n_epochs}")

        self.model.freeze_cf()

        for epoch in range(1, self.n_epochs + 1):
            t = time.time()
            if epoch == self.freeze_epochs + 1:
                self.model.unfreeze_cf()

            # Fresh triplet sampling every epoch
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

            phase = "freeze" if epoch <= self.freeze_epochs else "full  "
            lr_now = self.scheduler.get_last_lr()[0]
            log.info(f"  [{phase}] Epoch {epoch:>3}/{self.n_epochs}  "
                     f"train={tr_loss:.5f}  val={va_loss:.5f}  "
                     f"lr={lr_now:.2e}  ({time.time()-t:.1f}s)")

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
    def load_model(cls, path=HYBRID_CKPT_PATH,
                   device=DEVICE, **kw) -> HybridModel:
        ck = torch.load(path, map_location=device, weights_only=False)
        m  = HybridModel(n_users=ck["n_users"], n_items=ck["n_items"],
                         embed_dim=ck["embed_dim"], **kw)
        m.load_state_dict(ck["model_state"])
        m.to(device).eval()
        log.info(f"Hybrid loaded ← {path}  (best val={ck['best_val_loss']:.5f})")
        return m


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
    Tensors can be on CPU or GPU — we handle both.
    Returns (n_items,) numpy array.
    """
    model.eval()
    n_items    = sbert_emb.shape[0]
    all_scores = np.empty(n_items, dtype=np.float32)

    # User representation — computed once
    u_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
    # history_emb may be on CPU or GPU
    hist_row = history_emb[user_idx]  # (384,)
    if hist_row.device != torch.device(device):
        hist_row = hist_row.to(device, non_blocking=True)
    hist = hist_row.unsqueeze(0)  # (1, 384)
    u_repr = model.encode_user(u_tensor, hist)  # (1, d)

    for start in range(0, n_items, batch_size):
        end      = min(start + batch_size, n_items)
        B        = end - start
        item_ids = torch.arange(start, end, dtype=torch.long, device=device)

        s  = sbert_emb[start:end].to(device, non_blocking=True)
        im = imdb_feats[start:end].to(device, non_blocking=True)
        p  = popularity[start:end].unsqueeze(1).to(device, non_blocking=True)

        i_repr = model.encode_item(item_ids, s, im, p)             # (B, d)
        u_exp  = u_repr.expand(B, -1)                               # (B, d)
        scores = model.score(u_exp, i_repr)                         # (B,)
        all_scores[start:end] = scores.cpu().numpy()

    return all_scores