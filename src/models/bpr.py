"""
DataFlix — Model 2: BPR (Bayesian Personalized Ranking)
src/models/bpr.py

Optimises for ranking directly via pairwise loss:
  L = -mean( log σ( s(u,pos) - s(u,neg) ) )

s(u,i) = p_u · q_i + b_u + b_i

Negative items sampled proportional to popularity — popular unrated
items are more informative negatives than obscure ones.
Warm-started from ALS factors for faster convergence.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from src.config import (
        PROCESSED_DIR, RESULTS_DIR, DEVICE,
        BPR_FACTORS_PATH,
        LATENT_DIM_K, LR_BPR, BPR_REG,
        BPR_EPOCHS, BPR_BATCH_SIZE, BPR_SAMPLES_PER_EPOCH,
    )
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR     = _ROOT / "data/processed"
    RESULTS_DIR       = _ROOT / "results"
    DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BPR_FACTORS_PATH  = RESULTS_DIR / "bpr_factors.npz"
    LATENT_DIM_K      = 128
    LR_BPR            = 1e-3
    BPR_REG           = 1e-4
    BPR_EPOCHS        = 50
    BPR_BATCH_SIZE    = 4096
    BPR_SAMPLES_PER_EPOCH = 200_000

log = logging.getLogger(__name__)


class BPRModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int = LATENT_DIM_K):
        super().__init__()
        self.n_factors       = n_factors
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        self.user_biases     = nn.Embedding(n_users, 1)
        self.item_biases     = nn.Embedding(n_items, 1)
        scale = 1.0 / np.sqrt(n_factors)
        nn.init.normal_(self.user_embeddings.weight, 0, scale)
        nn.init.normal_(self.item_embeddings.weight, 0, scale)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)

    def forward(self, u, i, j):
        """Returns s(u,i) - s(u,j) for BPR loss."""
        p_u  = self.user_embeddings(u)
        q_i  = self.item_embeddings(i)
        q_j  = self.item_embeddings(j)
        b_u  = self.user_biases(u).squeeze(1)
        b_i  = self.item_biases(i).squeeze(1)
        b_j  = self.item_biases(j).squeeze(1)
        s_ui = (p_u * q_i).sum(1) + b_u + b_i
        s_uj = (p_u * q_j).sum(1) + b_u + b_j
        return s_ui - s_uj

    @torch.no_grad()
    def score_all_items(self, user_idx: int) -> torch.Tensor:
        """Score all items for one user. Returns (n_items,) tensor."""
        p_u    = self.user_embeddings.weight[user_idx]
        scores = self.item_embeddings.weight @ p_u
        scores = scores + self.item_biases.weight.squeeze(1)
        scores = scores + self.user_biases.weight[user_idx]
        return scores


class BPRSampler:
    def __init__(
        self,
        user_positives: dict[int, set],
        all_items:      np.ndarray,
        item_pop:       np.ndarray,
        n_users:        int,
    ):
        self.user_positives = user_positives
        self.all_items      = all_items
        pop                 = item_pop.astype(np.float64)
        self.item_probs     = pop / pop.sum()
        self.eligible       = np.array(
            [u for u in range(n_users)
             if u in user_positives and len(user_positives[u]) > 0],
            dtype=np.int32,
        )

    def sample(self, n: int):
        users     = np.random.choice(self.eligible, size=n)
        pos_items = np.empty(n, dtype=np.int32)
        neg_items = np.empty(n, dtype=np.int32)
        for k, u in enumerate(users):
            pos_set     = self.user_positives[int(u)]
            pos_items[k] = int(np.random.choice(list(pos_set)))
            for _ in range(20):
                neg = int(np.random.choice(self.all_items, p=self.item_probs))
                if neg not in pos_set:
                    break
            neg_items[k] = neg
        return users, pos_items, neg_items


class BPR:
    def __init__(
        self,
        n_users:           int,
        n_items:           int,
        n_factors:         int   = LATENT_DIM_K,
        lr:                float = LR_BPR,
        reg:               float = BPR_REG,
        n_epochs:          int   = BPR_EPOCHS,
        batch_size:        int   = BPR_BATCH_SIZE,
        samples_per_epoch: int   = BPR_SAMPLES_PER_EPOCH,
        device:            torch.device = DEVICE,
    ):
        self.n_epochs          = n_epochs
        self.batch_size        = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.device            = device
        self.model             = BPRModel(n_users, n_items, n_factors).to(device)
        self.optimizer         = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=reg
        )
        self.train_loss_history: list[float] = []

    def init_from_als(self, als) -> None:
        with torch.no_grad():
            u = als.get_user_factors_tensor().to(self.device)
            i = als.get_item_factors_tensor().to(self.device)
            k_min = min(self.model.n_factors, u.shape[1])
            self.model.user_embeddings.weight[:, :k_min] = u[:, :k_min]
            self.model.item_embeddings.weight[:, :k_min] = i[:, :k_min]
        log.info(f"BPR warm-started from ALS (k={k_min})")

    def fit(self, user_positives, all_items, item_pop) -> "BPR":
        n_users  = self.model.user_embeddings.num_embeddings
        sampler  = BPRSampler(user_positives, all_items, item_pop, n_users)
        n_batches = self.samples_per_epoch // self.batch_size

        log.info(f"BPR: {self.n_epochs} epochs | "
                 f"{self.samples_per_epoch:,} samples/ep | device={self.device}")

        for epoch in range(1, self.n_epochs + 1):
            t = time.time()
            users, pos_items, neg_items = sampler.sample(self.samples_per_epoch)
            losses = []
            self.model.train()

            for b in range(n_batches):
                s = b * self.batch_size
                e = s + self.batch_size
                u = torch.tensor(users[s:e],     dtype=torch.long, device=self.device)
                i = torch.tensor(pos_items[s:e], dtype=torch.long, device=self.device)
                j = torch.tensor(neg_items[s:e], dtype=torch.long, device=self.device)
                self.optimizer.zero_grad()
                loss = -torch.nn.functional.logsigmoid(self.model(u, i, j)).mean()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            ep_loss = float(np.mean(losses))
            self.train_loss_history.append(ep_loss)
            log.info(f"  Epoch {epoch:>3}/{self.n_epochs}  "
                     f"loss={ep_loss:.5f}  ({time.time()-t:.1f}s)")
        return self

    def save(self, path: Path = BPR_FACTORS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":  self.model.state_dict(),
            "n_users":      self.model.user_embeddings.num_embeddings,
            "n_items":      self.model.item_embeddings.num_embeddings,
            "n_factors":    self.model.n_factors,
            "loss_history": [float(x) for x in self.train_loss_history],
        }, path)
        log.info(f"BPR saved → {path}")

    @classmethod
    def load(cls, path: Path = BPR_FACTORS_PATH,
             device: torch.device = DEVICE) -> "BPR":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obj  = cls(n_users=ckpt["n_users"], n_items=ckpt["n_items"],
                   n_factors=ckpt["n_factors"], device=device)
        obj.model.load_state_dict(ckpt["model_state"])
        obj.train_loss_history = ckpt["loss_history"]
        log.info(f"BPR loaded ← {path}")
        return obj

    def get_user_embeddings_tensor(self) -> torch.Tensor:
        return self.model.user_embeddings.weight.detach().cpu()

    def get_item_embeddings_tensor(self) -> torch.Tensor:
        return self.model.item_embeddings.weight.detach().cpu()

    def __repr__(self):
        return f"BPR(k={self.model.n_factors}, epochs={self.n_epochs})"