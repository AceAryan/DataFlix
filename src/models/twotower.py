import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.hybrid import TripletDataset
from src.config import (
    DEVICE, TRAIN_CSV, VAL_CSV, RELEVANCE_RATING,
    LR_HYBRID, HYBRID_WEIGHT_DECAY, HYBRID_EPOCHS, COSINE_T_MAX,
    HYBRID_BATCH_SIZE, HYBRID_SAMPLES_PER_EPOCH, EARLY_STOP_PATIENCE,
    RESULTS_DIR
)

log = logging.getLogger(__name__)
TWOTOWER_CKPT_PATH = RESULTS_DIR / "twotower_best.pt"

class TwoTowerModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 128,
        sbert_dim: int = 384,
        imdb_dim: int = 23,
        dropout: float = 0.5   # Keep dropout high
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # NOTE: user_id_emb and item_id_emb are completely DELETED.
        
        # User Tower: strictly history_emb (384)
        self.user_tower = nn.Sequential(
            nn.Linear(sbert_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim)
        )
        
        # Item Tower: strictly sbert (384) + imdb (23) + pop (1) = 408
        self.item_tower = nn.Sequential(
            nn.Linear(sbert_dim + imdb_dim + 1, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim)
        )

    # --- Dummy methods to prevent HybridTrainer/API from crashing ---
    def load_cf_weights(self, *args, **kwargs): pass
    def freeze_cf(self): pass
    def unfreeze_cf(self): pass

    def encode_user(self, user_idx: torch.Tensor, history_emb: torch.Tensor) -> torch.Tensor:
        # Ignore user_idx entirely
        u_repr = self.user_tower(history_emb)
        return F.normalize(u_repr, p=2, dim=1) 

    def encode_item(self, item_idx: torch.Tensor, sbert_emb: torch.Tensor, imdb_feat: torch.Tensor, pop: torch.Tensor) -> torch.Tensor:
        # Ignore item_idx entirely
        x = torch.cat([sbert_emb, imdb_feat, pop], dim=1)
        i_repr = self.item_tower(x)
        return F.normalize(i_repr, p=2, dim=1)

    def score(self, user_repr: torch.Tensor, item_repr: torch.Tensor) -> torch.Tensor:
        # Cosine similarity multiplied by a low temperature
        return (user_repr * item_repr).sum(dim=1) * 5.0

    def forward(self, user_idx, item_idx, sbert_emb, imdb_feat, history_emb, pop):
        u_repr = self.encode_user(user_idx, history_emb)
        i_repr = self.encode_item(item_idx, sbert_emb, imdb_feat, pop)
        return self.score(u_repr, i_repr)
    
class TwoTowerTrainer:
    def __init__(
        self,
        model:          TwoTowerModel, 
        sbert_emb:      torch.Tensor,
        imdb_feats:     torch.Tensor,
        popularity:     torch.Tensor,
        history_emb:    torch.Tensor,
        all_items:      np.ndarray,
        item_pop:       np.ndarray,
        device:         torch.device = DEVICE,
        lr:             float        = LR_HYBRID,
        weight_decay:   float        = HYBRID_WEIGHT_DECAY,
        n_epochs:       int          = HYBRID_EPOCHS,
        batch_size:     int          = HYBRID_BATCH_SIZE,
        samples_per_epoch: int       = HYBRID_SAMPLES_PER_EPOCH,
        patience:       int          = EARLY_STOP_PATIENCE,
    ):
        self.model             = model.to(device)
        self.device            = device
        self.n_epochs          = n_epochs
        self.batch_size        = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.patience          = patience
        self.all_items         = all_items
        self.item_pop          = item_pop

        self.sbert_emb   = sbert_emb
        self.imdb_feats  = imdb_feats
        self.popularity  = popularity
        self.history_emb = history_emb

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=COSINE_T_MAX, eta_min=1e-5)
        
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_loss_history = []
        self.val_loss_history = []

    def _item_feats(self, item_idx: torch.Tensor):
        cpu = item_idx.cpu()
        s   = self.sbert_emb[cpu].to(self.device, non_blocking=True)
        im  = self.imdb_feats[cpu].to(self.device, non_blocking=True)
        p   = self.popularity[cpu].unsqueeze(1).to(self.device, non_blocking=True)
        return s, im, p

    def _user_feats(self, user_idx: torch.Tensor) -> torch.Tensor:
        return self.history_emb[user_idx.cpu()].to(self.device, non_blocking=True)

    def _bpr_step(self, u, pos, neg) -> torch.Tensor:
        hist     = self._user_feats(u)
        u_repr   = self.model.encode_user(u, hist)

        s_pos, im_pos, p_pos = self._item_feats(pos)
        s_neg, im_neg, p_neg = self._item_feats(neg)

        i_pos_repr = self.model.encode_item(pos, s_pos, im_pos, p_pos)
        i_neg_repr = self.model.encode_item(neg, s_neg, im_neg, p_neg)

        score_pos = self.model.score(u_repr, i_pos_repr)
        score_neg = self.model.score(u_repr, i_neg_repr)

        return -torch.nn.functional.logsigmoid(score_pos - score_neg).mean()

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total, n = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for u, pos, neg in loader:
                u, pos, neg = u.to(self.device), pos.to(self.device), neg.to(self.device)
                loss = self._bpr_step(u, pos, neg)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                total += loss.item() * len(u)
                n     += len(u)
        return total / n if n > 0 else float("inf")

    def fit(self) -> "TwoTowerTrainer":
        log.info(f"Building positive sets (rating >= {RELEVANCE_RATING})...")
        def _pos_sets(csv_path):
            df = pd.read_csv(csv_path)
            ps = defaultdict(set)
            for row in df[df["rating"] >= RELEVANCE_RATING].itertuples(index=False):
                ps[int(row.user_idx)].add(int(row.movie_idx))
            return dict(ps)

        train_pos = _pos_sets(TRAIN_CSV)
        val_pos   = _pos_sets(VAL_CSV)

        eligible_train = np.array([u for u, s in train_pos.items() if len(s) > 0], dtype=np.int32)
        eligible_val   = np.array([u for u, s in val_pos.items() if len(s) > 0], dtype=np.int32)

        log.info(f"Two-Tower training: {self.n_epochs} epochs | batch={self.batch_size}")

        for epoch in range(1, self.n_epochs + 1):
            t = time.time()
            
            tr_loader = DataLoader(TripletDataset(train_pos, self.all_items, self.item_pop, self.samples_per_epoch, eligible_train), batch_size=self.batch_size, shuffle=True, num_workers=0)
            va_loader = DataLoader(TripletDataset(val_pos, self.all_items, self.item_pop, max(10_000, self.samples_per_epoch // 10), eligible_val), batch_size=self.batch_size * 2, shuffle=False, num_workers=0)

            tr_loss = self._run_epoch(tr_loader, train=True)
            va_loss = self._run_epoch(va_loader, train=False)
            self.scheduler.step()

            self.train_loss_history.append(tr_loss)
            self.val_loss_history.append(va_loss)

            lr_now = self.scheduler.get_last_lr()[0]
            log.info(f"  Epoch {epoch:>3}/{self.n_epochs}  train={tr_loss:.5f}  val={va_loss:.5f}  lr={lr_now:.2e}  ({time.time()-t:.1f}s)")

            if va_loss < self.best_val_loss:
                self.best_val_loss = va_loss
                self.patience_counter = 0
                self.save(TWOTOWER_CKPT_PATH)
                log.info(f"    ✓ Best val={va_loss:.5f} — saved")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    log.info(f"  Early stop at epoch {epoch}")
                    break

        return self

    def save(self, path: Path = TWOTOWER_CKPT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":   self.model.state_dict(),
            "best_val_loss": float(self.best_val_loss),
            # Infer sizes from the feature tensors since ID embeddings are gone
            "n_users":       self.history_emb.shape[0],
            "n_items":       self.sbert_emb.shape[0],
            "embed_dim":     self.model.embed_dim,
        }, path)

@torch.no_grad()
def score_twotower(
    model:       TwoTowerModel,
    user_idx:    int,
    sbert_emb:   torch.Tensor,
    imdb_feats:  torch.Tensor,
    popularity:  torch.Tensor,
    history_emb: torch.Tensor,
    device:      torch.device,
    batch_size:  int = 1024,
) -> np.ndarray:
    model.eval()
    n_items = sbert_emb.shape[0]
    all_scores = np.empty(n_items, dtype=np.float32)

    # 1. User Representation
    u_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
    hist_row = history_emb[user_idx]
    if hist_row.device != torch.device(device):
        hist_row = hist_row.to(device, non_blocking=True)
    hist = hist_row.unsqueeze(0)
    
    u_repr = model.encode_user(u_tensor, hist)

    # 2. Batch Processing for Items
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        B = end - start
        item_ids = torch.arange(start, end, dtype=torch.long, device=device)

        s  = sbert_emb[start:end].to(device, non_blocking=True)
        im = imdb_feats[start:end].to(device, non_blocking=True)
        p  = popularity[start:end].unsqueeze(1).to(device, non_blocking=True)

        i_repr = model.encode_item(item_ids, s, im, p)
        u_exp  = u_repr.expand(B, -1)
        
        # Two-Tower score is just the native dot product
        scores = model.score(u_exp, i_repr)
        all_scores[start:end] = scores.cpu().numpy()

    return all_scores