"""
DataFlix — Training Module
Path A (MSE loss) and Path B (BPR loss) trainers.

Architecture:
  - Datasets store ONLY indices (user_idx, item_idx, rating) — tiny, fast workers
  - All feature tensors (sbert, popularity, user_features, history, genres) live on
    GPU as module-level globals, looked up once per batch with vectorised indexing
  - This eliminates per-sample CPU tensor copies that caused 2 it/s
  - num_workers=4, pin_memory=True, prefetch_factor=4 for parallel prefetch
  - torch.amp mixed precision (AMP) for 1.5-2x GPU throughput
  - Epoch checkpointing — resume across sessions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE, LR_PATH_A, LR_PATH_B, WEIGHT_DECAY, COSINE_T_MAX,
    EARLY_STOP_PATIENCE, MAX_EPOCHS, BATCH_SIZE, BPR_REG,
    BPR_EPOCHS, BPR_BATCH_SIZE, SEED, RESULTS_DIR
)

NUM_WORKERS = 4   # parallel DataLoader workers (set 0 on Windows if issues)

# ─────────────────────────────────────────────────────────────────────────────
# GPU feature store — loaded once, shared across all batches
# ─────────────────────────────────────────────────────────────────────────────

# These are populated by init_feature_store() before training starts.
# Workers never touch them — only the main process indexes into them per batch.
_SBERT        = None   # (n_movies, 384)
_POPULARITY   = None   # (n_movies,)
_USER_FEAT    = None   # (n_users,  feat_dim)
_HISTORY      = None   # (n_users,  384)
_GENRE        = None   # (n_movies, max_genres)


def _build_genre_tensor(genre_table: dict, n_movies: int) -> torch.Tensor:
    genre_ids_map = genre_table.get("movie_genre_ids", {})
    if not genre_ids_map:
        return torch.zeros(n_movies, 1, dtype=torch.long)
    max_g = max(len(v) for v in genre_ids_map.values())
    t = torch.zeros(n_movies, max_g, dtype=torch.long)
    for mid, gids in genre_ids_map.items():
        mid = int(mid)
        if mid < n_movies:
            t[mid, :len(gids)] = torch.tensor(gids, dtype=torch.long)
    return t


def init_feature_store(sbert_embeddings: torch.Tensor,
                       popularity: torch.Tensor,
                       user_features: torch.Tensor,
                       history_embeddings: torch.Tensor,
                       genre_table: dict):
    """
    Move all feature tensors to GPU once at startup.
    Call this BEFORE constructing any trainer or dataset.
    """
    global _SBERT, _POPULARITY, _USER_FEAT, _HISTORY, _GENRE
    print("Loading feature tensors onto GPU...")
    _SBERT      = sbert_embeddings.to(DEVICE)
    _POPULARITY = popularity.to(DEVICE)
    _USER_FEAT  = user_features.to(DEVICE)
    _HISTORY    = history_embeddings.to(DEVICE)
    n_movies    = sbert_embeddings.shape[0]
    _GENRE      = _build_genre_tensor(genre_table, n_movies).to(DEVICE)
    print(f"  SBERT:      {_SBERT.shape}  on {_SBERT.device}")
    print(f"  Popularity: {_POPULARITY.shape}  on {_POPULARITY.device}")
    print(f"  User feat:  {_USER_FEAT.shape}  on {_USER_FEAT.device}")
    print(f"  History:    {_HISTORY.shape}  on {_HISTORY.device}")
    print(f"  Genres:     {_GENRE.shape}  on {_GENRE.device}")


def _lookup(uids: torch.Tensor, iids: torch.Tensor):
    """
    Batch-level GPU feature lookup — one vectorised index per feature per batch.
    Called in the train loop after uids/iids are on GPU.
    """
    sberts = _SBERT[iids]
    pops   = _POPULARITY[iids].unsqueeze(1)
    ufeats = _USER_FEAT[uids]
    hists  = _HISTORY[uids]
    genres = _GENRE[iids]
    return sberts, pops, ufeats, hists, genres


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scheduler, scaler,
                    epoch, history, best_val_rmse, patience_counter, best_state):
    ckpt = {
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict() if scheduler else None,
        "scaler_state":     scaler.state_dict(),
        "history":          history,
        "best_val_rmse":    best_val_rmse,
        "patience_counter": patience_counter,
        "best_state":       best_state,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)
    print(f"  [ckpt] Saved -> {path.name}  (epoch {epoch+1})")


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    blank = (0, {"train_loss": [], "val_loss": [], "val_rmse": [], "lr": []},
             float("inf"), 0, None)
    if not path.exists():
        return blank
    print(f"  [ckpt] Resuming from {path.name}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt["scheduler_state"]:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    ep = ckpt["epoch"] + 1
    print(f"  [ckpt] Epoch {ep+1}, best RMSE {ckpt['best_val_rmse']:.4f}, "
          f"patience {ckpt['patience_counter']}/{EARLY_STOP_PATIENCE}")
    return (ep, ckpt["history"], ckpt["best_val_rmse"],
            ckpt["patience_counter"], ckpt["best_state"])


def save_bpr_checkpoint(path, model, optimizer, scaler, epoch, history):
    ckpt = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "history":         history,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)
    print(f"  [ckpt] BPR saved -> {path.name}  (epoch {epoch+1})")


def load_bpr_checkpoint(path, model, optimizer, scaler):
    if not path.exists():
        return 0, {"train_loss": []}
    print(f"  [ckpt] Resuming BPR from {path.name}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    ep = ckpt["epoch"] + 1
    print(f"  [ckpt] BPR epoch {ep+1}")
    return ep, ckpt["history"]


# ─────────────────────────────────────────────────────────────────────────────
# Datasets — indices only, no feature tensors
# ─────────────────────────────────────────────────────────────────────────────

class RatingDataset(Dataset):
    """
    Stores only (user_idx, item_idx, rating) as contiguous CPU arrays.
    Workers copy nothing but three small scalars per sample.
    Feature lookup happens once per batch in the train loop via _lookup().
    """

    def __init__(self, df: pd.DataFrame, *args, **kwargs):
        # Accept (and ignore) old feature tensor arguments so run_train.py
        # call sites do not need to change.
        self.users   = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items   = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating_centered"].values,   dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class BPRDataset(Dataset):
    """
    Stores only (user_idx, pos_idx, neg_idx) triplets.
    Negative sampling is fully vectorised at construction time.
    Feature lookup happens once per batch via _lookup().
    """

    def __init__(self, user_positives: dict, all_items: np.ndarray,
                 item_pop: np.ndarray,
                 *args,                          # sbert/pop/user_feat/hist/genre ignored
                 n_samples_per_epoch: int = 500_000,
                 **kwargs):
        self.user_positives = user_positives
        self.users          = np.array(list(user_positives.keys()))
        self.all_items      = all_items
        self.item_pop       = item_pop / item_pop.sum()
        self.n_samples      = n_samples_per_epoch
        self._generate_samples()

    def _generate_samples(self):
        """Vectorised: draw all negatives in one call, fix collisions in bulk."""
        rng = np.random.default_rng(SEED)

        u_idx    = rng.integers(0, len(self.users), size=self.n_samples)
        u_sample = self.users[u_idx]
        pos_sample = np.array([
            rng.choice(list(self.user_positives[u])) for u in u_sample
        ])

        # All negatives in one vectorised call
        neg_sample = rng.choice(self.all_items, size=self.n_samples,
                                p=self.item_pop)

        # Fix the ~1% collision rate
        for i in range(self.n_samples):
            while neg_sample[i] in self.user_positives[u_sample[i]]:
                neg_sample[i] = rng.choice(self.all_items, p=self.item_pop)

        self.u_arr   = u_sample.astype(np.int32)
        self.pos_arr = pos_sample.astype(np.int32)
        self.neg_arr = neg_sample.astype(np.int32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (int(self.u_arr[idx]),
                int(self.pos_arr[idx]),
                int(self.neg_arr[idx]))


# ─────────────────────────────────────────────────────────────────────────────
# Collate functions — build plain index batches only
# ─────────────────────────────────────────────────────────────────────────────

def collate_rating(batch):
    uids, iids, ratings = zip(*batch)
    return (torch.stack(uids),
            torch.stack(iids),
            torch.stack(ratings))


def collate_bpr(batch):
    uids, pos_ids, neg_ids = zip(*batch)
    return (torch.tensor(uids,    dtype=torch.long),
            torch.tensor(pos_ids, dtype=torch.long),
            torch.tensor(neg_ids, dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# Pure GPU batch iterator — no DataLoader, no workers, no IPC overhead
# ─────────────────────────────────────────────────────────────────────────────

class _GPUBatchIter:
    """
    Holds the entire index dataset as GPU tensors and yields shuffled
    mini-batches with zero CPU-GPU transfer overhead.
    On Windows, DataLoader multiprocessing (spawn) is very slow — this
    replaces it entirely for datasets that fit in GPU memory.
    """

    def __init__(self, users: torch.Tensor, items: torch.Tensor,
                 ratings: torch.Tensor, batch_size: int, shuffle: bool = True):
        self.users     = users.to(DEVICE)
        self.items     = items.to(DEVICE)
        self.ratings   = ratings.to(DEVICE)
        self.bs        = batch_size
        self.shuffle   = shuffle
        self.n         = len(users)

    def __len__(self):
        return (self.n + self.bs - 1) // self.bs

    def __iter__(self):
        perm = torch.randperm(self.n, device=DEVICE) if self.shuffle \
               else torch.arange(self.n, device=DEVICE)
        for start in range(0, self.n, self.bs):
            idx = perm[start:start + self.bs]
            yield self.users[idx], self.items[idx], self.ratings[idx]


class _GPUBPRIter:
    """Same concept for BPR triplets."""

    def __init__(self, u_arr, pos_arr, neg_arr, batch_size: int):
        self.uids    = torch.tensor(u_arr,   dtype=torch.long,  device=DEVICE)
        self.pos_ids = torch.tensor(pos_arr, dtype=torch.long,  device=DEVICE)
        self.neg_ids = torch.tensor(neg_arr, dtype=torch.long,  device=DEVICE)
        self.bs      = batch_size
        self.n       = len(u_arr)

    def __len__(self):
        return (self.n + self.bs - 1) // self.bs

    def __iter__(self):
        perm = torch.randperm(self.n, device=DEVICE)
        for start in range(0, self.n, self.bs):
            idx = perm[start:start + self.bs]
            yield self.uids[idx], self.pos_ids[idx], self.neg_ids[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Path A Trainer (MSE)
# ─────────────────────────────────────────────────────────────────────────────

class PathATrainer:
    """
    MSE rating prediction trainer.
    Uses pure GPU tensor iteration — no DataLoader, no workers.
    Each batch: GPU randperm slice -> vectorised feature lookup -> forward ->
    AMP loss -> scaler backward. GPU stays busy the entire time.
    """

    CKPT_NAME = "ckpt_path_a.pt"

    def __init__(self, model, lr=LR_PATH_A, weight_decay=WEIGHT_DECAY,
                 t_max=COSINE_T_MAX, patience=EARLY_STOP_PATIENCE,
                 max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE):
        self.model     = model.to(DEVICE)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                             self.optimizer, T_max=t_max)
        self.patience   = patience
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.scaler     = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
        self.ckpt_path  = RESULTS_DIR / self.CKPT_NAME

    def train(self, train_dataset, val_dataset):
        # Build GPU iterators once — entire index arrays live on GPU
        train_iter = _GPUBatchIter(
            train_dataset.users, train_dataset.items, train_dataset.ratings,
            self.batch_size, shuffle=True)
        val_iter = _GPUBatchIter(
            val_dataset.users, val_dataset.items, val_dataset.ratings,
            self.batch_size * 2, shuffle=False)

        print(f"Train batches/epoch: {len(train_iter):,}  "
              f"Val batches: {len(val_iter):,}  "
              f"Batch size: {self.batch_size:,}")

        start_epoch, history, best_val_rmse, patience_counter, best_state = \
            load_checkpoint(self.ckpt_path, self.model, self.optimizer,
                            self.scheduler, self.scaler)

        for epoch in range(start_epoch, self.max_epochs):
            t0 = time.time()
            self.model.train()
            total_loss = 0
            n_batches  = 0

            for uids, iids, ratings in tqdm(train_iter,
                                            desc=f"Epoch {epoch+1}",
                                            leave=False,
                                            total=len(train_iter)):
                # uids/iids/ratings already on GPU — zero transfer cost
                sberts, pops, ufeats, hists, genres = _lookup(uids, iids)

                with torch.amp.autocast(device_type=DEVICE.type,
                                        enabled=DEVICE.type == "cuda"):
                    preds = self.model(uids, iids, sberts, pops,
                                       genres, hists, ufeats)
                    loss  = self.criterion(preds, ratings)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                n_batches  += 1

            self.scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            val_rmse, val_loss = self._evaluate(val_iter)
            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            history["train_loss"].append(avg_loss)
            history["val_loss"].append(val_loss)
            history["val_rmse"].append(val_rmse)
            history["lr"].append(lr)

            print(f"Epoch {epoch+1}/{self.max_epochs}: "
                  f"train={avg_loss:.4f}  val_rmse={val_rmse:.4f}  "
                  f"lr={lr:.2e}  time={elapsed:.1f}s")

            if val_rmse < best_val_rmse:
                best_val_rmse    = val_rmse
                best_state       = {k: v.cpu().clone()
                                    for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            save_checkpoint(self.ckpt_path, self.model, self.optimizer,
                            self.scheduler, self.scaler, epoch, history,
                            best_val_rmse, patience_counter, best_state)

            if patience_counter >= self.patience:
                print(f"Early stopping -- best val RMSE: {best_val_rmse:.4f}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if self.ckpt_path.exists():
            self.ckpt_path.unlink()
            print(f"  [ckpt] Done -- removed {self.CKPT_NAME}")
        return history

    def _evaluate(self, val_iter):
        self.model.eval()
        total_se = total_loss = n = 0
        with torch.no_grad():
            for uids, iids, ratings in val_iter:
                sberts, pops, ufeats, hists, genres = _lookup(uids, iids)
                with torch.amp.autocast(device_type=DEVICE.type,
                                        enabled=DEVICE.type == "cuda"):
                    preds = self.model(uids, iids, sberts, pops,
                                       genres, hists, ufeats)
                    loss  = self.criterion(preds, ratings)
                total_se   += ((preds - ratings) ** 2).sum().item()
                total_loss += loss.item() * len(ratings)
                n          += len(ratings)
        return np.sqrt(total_se / max(n, 1)), total_loss / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Path B Trainer (BPR)
# ─────────────────────────────────────────────────────────────────────────────

class PathBTrainer:
    """
    BPR ranking trainer — pure GPU tensor iteration, no DataLoader.
    """

    CKPT_NAME = "ckpt_path_b.pt"

    def __init__(self, model, lr=LR_PATH_B, reg=BPR_REG,
                 max_epochs=BPR_EPOCHS, batch_size=BPR_BATCH_SIZE):
        self.model      = model.to(DEVICE)
        self.optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.scaler     = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
        self.ckpt_path  = RESULTS_DIR / self.CKPT_NAME

    @staticmethod
    def bpr_loss(pos, neg):
        return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()

    @staticmethod
    def _lookup_bpr(uids, pos_ids, neg_ids):
        return (_SBERT[pos_ids], _SBERT[neg_ids],
                _POPULARITY[pos_ids].unsqueeze(1),
                _POPULARITY[neg_ids].unsqueeze(1),
                _USER_FEAT[uids], _HISTORY[uids],
                _GENRE[pos_ids],  _GENRE[neg_ids])

    def train(self, bpr_dataset):
        bpr_iter = _GPUBPRIter(
            bpr_dataset.u_arr, bpr_dataset.pos_arr, bpr_dataset.neg_arr,
            self.batch_size)

        print(f"BPR batches/epoch: {len(bpr_iter):,}  "
              f"Batch size: {self.batch_size:,}")

        start_epoch, history = load_bpr_checkpoint(
            self.ckpt_path, self.model, self.optimizer, self.scaler)

        for epoch in range(start_epoch, self.max_epochs):
            t0 = time.time()
            self.model.train()
            total_loss = 0
            n_batches  = 0

            for uids, pos_ids, neg_ids in tqdm(bpr_iter,
                                               desc=f"BPR Epoch {epoch+1}",
                                               leave=False,
                                               total=len(bpr_iter)):
                (sbert_pos, sbert_neg, pop_pos, pop_neg,
                 ufeats, hists, genre_pos, genre_neg) = self._lookup_bpr(
                    uids, pos_ids, neg_ids)

                with torch.amp.autocast(device_type=DEVICE.type,
                                        enabled=DEVICE.type == "cuda"):
                    score_pos, score_neg = self.model.predict_pair_scores(
                        uids, pos_ids, neg_ids,
                        sbert_pos, sbert_neg, pop_pos, pop_neg,
                        genre_pos, genre_neg, hists, ufeats)
                    loss = self.bpr_loss(score_pos, score_neg)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                n_batches  += 1

            avg_loss = total_loss / max(n_batches, 1)
            history["train_loss"].append(avg_loss)
            elapsed = time.time() - t0
            print(f"BPR Epoch {epoch+1}/{self.max_epochs}: "
                  f"loss={avg_loss:.4f}  time={elapsed:.1f}s")

            save_bpr_checkpoint(self.ckpt_path, self.model, self.optimizer,
                                self.scaler, epoch, history)

        if self.ckpt_path.exists():
            self.ckpt_path.unlink()
            print(f"  [ckpt] BPR done -- removed {self.CKPT_NAME}")
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Simple MF Trainer (baselines)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleMFTrainer:
    """Trainer for simple PyTorch MF models (VanillaMF, NeuMF)."""

    def __init__(self, model, lr=5e-3, weight_decay=1e-4,
                 max_epochs=50, batch_size=4096, patience=5):
        self.model      = model.to(DEVICE)
        self.criterion  = nn.MSELoss()
        self.optimizer  = optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                              self.optimizer, T_max=max_epochs)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience   = patience

    def train(self, train_df, val_df):
        train_users   = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
        train_items   = torch.tensor(train_df["movie_idx"].values, dtype=torch.long)
        train_ratings = torch.tensor(train_df["rating"].values,   dtype=torch.float32)
        val_users     = torch.tensor(val_df["user_idx"].values,   dtype=torch.long)
        val_items     = torch.tensor(val_df["movie_idx"].values,  dtype=torch.long)
        val_ratings   = torch.tensor(val_df["rating"].values,     dtype=torch.float32)

        n = len(train_users)
        best_rmse = float("inf")
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(n)
            total_loss = 0
            n_batches  = 0

            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                u   = train_users[idx].to(DEVICE,   non_blocking=True)
                it  = train_items[idx].to(DEVICE,   non_blocking=True)
                r   = train_ratings[idx].to(DEVICE, non_blocking=True)
                pred = self.model(u, it)
                loss = self.criterion(pred, r)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            self.scheduler.step()
            self.model.eval()
            with torch.no_grad():
                vp = self.model(val_users.to(DEVICE), val_items.to(DEVICE))
                val_rmse = torch.sqrt(
                    ((vp - val_ratings.to(DEVICE)) ** 2).mean()).item()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: loss={total_loss/n_batches:.4f} "
                      f"val_rmse={val_rmse:.4f}")

            if val_rmse < best_rmse:
                best_rmse        = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        print(f"  Best val RMSE: {best_rmse:.4f}")
        return best_rmse