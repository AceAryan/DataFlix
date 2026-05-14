"""
DataFlix — Model 2: Bayesian Personalized Ranking (BPR)
src/models/bpr.py

Optimises for ranking rather than rating prediction.

Objective: for each user u, the model should score a positive item i
(one the user rated) higher than a negative item j (one they haven't seen).

Loss: L = -Σ ln σ(x_ui - x_uj) + λ(||P||² + ||Q||²)
  where x_ui = p_u · q_i  (dot product score)

Key difference from ALS:
  - ALS minimises rating prediction error (RMSE)
  - BPR maximises ranking quality (directly optimises what NDCG/Recall measure)
  - BPR treats all unrated items as implicit negatives, not missing data

Training via mini-batch SGD on sampled (user, pos_item, neg_item) triples.
Negative items are sampled proportional to popularity — popular unrated items
are more informative negatives than obscure ones.
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
        BPR_DATA_PATH, USER_POSITIVES_PATH,
        LATENT_DIM_K, EMBED_DIM_D,
        LR_PATH_B, BPR_REG, BPR_EPOCHS,
        BPR_BATCH_SIZE, BPR_SAMPLES_PER_EPOCH,
        TOP_K_VALUES,
    )
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR         = _ROOT / "data/processed"
    RESULTS_DIR           = _ROOT / "results"
    DEVICE                = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BPR_DATA_PATH         = PROCESSED_DIR / "bpr_data.npz"
    USER_POSITIVES_PATH   = PROCESSED_DIR / "user_positives.pkl"
    LATENT_DIM_K          = 100
    EMBED_DIM_D           = 128
    LR_PATH_B             = 1e-3
    BPR_REG               = 1e-4
    BPR_EPOCHS            = 50
    BPR_BATCH_SIZE        = 8192
    BPR_SAMPLES_PER_EPOCH = 200_000
    TOP_K_VALUES          = [5, 10, 20]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BPR_FACTORS_PATH = RESULTS_DIR / "bpr_factors.npz"


# ── Model ─────────────────────────────────────────────────────────────────────

class BPRModel(nn.Module):
    """
    Embedding-based BPR model.

    Learnable parameters:
      user_embeddings : (n_users,  k)
      item_embeddings : (n_items,  k)
      user_biases     : (n_users,)   — per-user preference offset
      item_biases     : (n_items,)   — per-item popularity offset

    Score for (u, i): s(u,i) = p_u · q_i + b_u + b_i
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = LATENT_DIM_K,
    ):
        super().__init__()
        self.n_factors = n_factors

        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        self.user_biases     = nn.Embedding(n_users, 1)
        self.item_biases     = nn.Embedding(n_items, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        scale = 1.0 / np.sqrt(self.n_factors)
        nn.init.normal_(self.user_embeddings.weight, 0, scale)
        nn.init.normal_(self.item_embeddings.weight, 0, scale)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)

    def forward(
        self,
        user_ids:     torch.Tensor,   # (B,)
        pos_item_ids: torch.Tensor,   # (B,)
        neg_item_ids: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """
        Returns x_ui - x_uj for each triple in the batch.
        BPR loss = -mean(log σ(x_ui - x_uj))
        """
        p_u   = self.user_embeddings(user_ids)       # (B, k)
        q_i   = self.item_embeddings(pos_item_ids)   # (B, k)
        q_j   = self.item_embeddings(neg_item_ids)   # (B, k)
        b_u   = self.user_biases(user_ids).squeeze(1)      # (B,)
        b_i   = self.item_biases(pos_item_ids).squeeze(1)  # (B,)
        b_j   = self.item_biases(neg_item_ids).squeeze(1)  # (B,)

        # Score difference: s(u,i) - s(u,j)
        x_ui  = (p_u * q_i).sum(dim=1) + b_u + b_i   # (B,)
        x_uj  = (p_u * q_j).sum(dim=1) + b_u + b_j   # (B,)

        return x_ui - x_uj  # (B,)  — positive means u prefers i over j

    def score_all_items(self, user_idx: int) -> torch.Tensor:
        """
        Score all items for a single user. Used during evaluation.
        Returns (n_items,) tensor.
        """
        p_u    = self.user_embeddings.weight[user_idx]   # (k,)
        scores = self.item_embeddings.weight @ p_u       # (n_items,)
        scores += self.item_biases.weight.squeeze(1)
        scores += self.user_biases.weight[user_idx]
        return scores


# ── Sampler ───────────────────────────────────────────────────────────────────

class BPRSampler:
    """
    Samples (user, pos_item, neg_item) triples for BPR training.

    Negative items are sampled proportional to item popularity.
    This is a form of hard negative mining — popular items that the user
    hasn't rated are more informative than rare items.

    Rejects negatives that are in the user's positive set.
    Capped at 10 rejection attempts per sample to avoid infinite loops
    for users who have rated almost everything.
    """

    def __init__(
        self,
        user_positives: dict[int, set],
        all_items:      np.ndarray,
        item_pop:       np.ndarray,
        n_users:        int,
    ):
        self.user_positives = user_positives
        self.all_items      = all_items
        self.n_users        = n_users

        # Popularity-weighted sampling probabilities
        pop_weights = item_pop.astype(np.float64)
        self.item_probs = pop_weights / pop_weights.sum()

        # Users who have at least one positive item
        self.eligible_users = np.array(
            [u for u in range(n_users) if u in user_positives and
             len(user_positives[u]) > 0],
            dtype=np.int32,
        )

    def sample(self, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample n_samples triples.

        Returns
        -------
        users     : (n_samples,) int32
        pos_items : (n_samples,) int32
        neg_items : (n_samples,) int32
        """
        users     = np.empty(n_samples, dtype=np.int32)
        pos_items = np.empty(n_samples, dtype=np.int32)
        neg_items = np.empty(n_samples, dtype=np.int32)

        # Sample users uniformly from eligible users
        sampled_users = np.random.choice(self.eligible_users, size=n_samples)

        for i, u in enumerate(sampled_users):
            pos_set = self.user_positives[int(u)]

            # Sample positive item uniformly from user's history
            pos_item = int(np.random.choice(list(pos_set)))

            # Sample negative with popularity weighting + rejection
            for _ in range(10):  # Max rejection attempts
                neg_item = int(np.random.choice(self.all_items, p=self.item_probs))
                if neg_item not in pos_set:
                    break
            # If still in pos_set after 10 attempts, keep it (rare edge case)

            users[i]     = u
            pos_items[i] = pos_item
            neg_items[i] = neg_item

        return users, pos_items, neg_items


# ── Trainer ───────────────────────────────────────────────────────────────────

class BPR:
    """
    BPR trainer — wraps BPRModel + BPRSampler + training loop.

    Parameters
    ----------
    n_users           : total number of users
    n_items           : total number of items
    n_factors         : latent dimension k
    lr                : learning rate
    reg               : L2 regularisation weight
    n_epochs          : training epochs
    batch_size        : triples per gradient step
    samples_per_epoch : total triples sampled per epoch
    device            : torch device
    """

    def __init__(
        self,
        n_users:           int,
        n_items:           int,
        n_factors:         int   = LATENT_DIM_K,
        lr:                float = LR_PATH_B,
        reg:               float = BPR_REG,
        n_epochs:          int   = BPR_EPOCHS,
        batch_size:        int   = BPR_BATCH_SIZE,
        samples_per_epoch: int   = BPR_SAMPLES_PER_EPOCH,
        device:            torch.device = DEVICE,
    ):
        self.n_epochs          = n_epochs
        self.batch_size        = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.reg               = reg
        self.device            = device

        self.model = BPRModel(n_users, n_items, n_factors).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr,
                                    weight_decay=reg)

        self.train_loss_history: list[float] = []

    # ── Initialise from ALS factors ───────────────────────────────────────────

    def init_from_als(self, als_model) -> None:
        """
        Warm-start BPR embeddings from ALS factors.
        ALS factors encode collaborative signal; BPR then refines for ranking.
        This converges faster than random init and often gives better NDCG.
        """
        with torch.no_grad():
            als_user = als_model.get_user_factors_tensor().to(self.device)
            als_item = als_model.get_item_factors_tensor().to(self.device)

            # Truncate or pad if factor dimensions differ
            k_bpr = self.model.n_factors
            k_als = als_user.shape[1]
            k_min = min(k_bpr, k_als)

            self.model.user_embeddings.weight[:, :k_min] = als_user[:, :k_min]
            self.model.item_embeddings.weight[:, :k_min] = als_item[:, :k_min]

        log.info(f"BPR embeddings warm-started from ALS (k_als={k_als}, k_bpr={k_bpr})")

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        user_positives: dict[int, set],
        all_items:      np.ndarray,
        item_pop:       np.ndarray,
    ) -> "BPR":
        """
        Train BPR on sampled triples.

        Parameters
        ----------
        user_positives : {user_idx → set of positive movie_idx}
        all_items      : array of all movie_idx values
        item_pop       : popularity count per item (same order as all_items)
        """
        n_users = self.model.user_embeddings.num_embeddings

        sampler = BPRSampler(user_positives, all_items, item_pop, n_users)
        n_batches = self.samples_per_epoch // self.batch_size

        log.info(f"BPR training: {self.n_epochs} epochs | "
                 f"{self.samples_per_epoch:,} samples/epoch | "
                 f"batch={self.batch_size} | device={self.device}")

        for epoch in range(1, self.n_epochs + 1):
            t      = time.time()
            losses = []

            # Sample all triples for this epoch upfront
            users, pos_items, neg_items = sampler.sample(self.samples_per_epoch)

            self.model.train()
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end   = start + self.batch_size

                u = torch.tensor(users[start:end],     dtype=torch.long, device=self.device)
                i = torch.tensor(pos_items[start:end], dtype=torch.long, device=self.device)
                j = torch.tensor(neg_items[start:end], dtype=torch.long, device=self.device)

                self.optimizer.zero_grad()

                # Score difference x_ui - x_uj
                diff = self.model(u, i, j)

                # BPR loss: -mean(log σ(diff))
                # Equivalent to binary cross-entropy with all labels = 1
                loss = -torch.nn.functional.logsigmoid(diff).mean()

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            epoch_loss = np.mean(losses)
            self.train_loss_history.append(epoch_loss)

            elapsed = time.time() - t
            log.info(f"  Epoch {epoch:>3}/{self.n_epochs}  "
                     f"loss={epoch_loss:.5f}  ({elapsed:.1f}s)")

        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def recommend(
        self,
        user_idx:    int,
        n:           int = 10,
        seen_items:  set | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recommend top-n items for a user.

        Returns
        -------
        top_items  : np.ndarray of movie_idx, shape (n,)
        top_scores : np.ndarray of predicted scores, shape (n,)
        """
        self.model.eval()
        scores = self.model.score_all_items(user_idx).cpu().numpy()

        if seen_items:
            scores[list(seen_items)] = -np.inf

        top_idx    = np.argpartition(scores, -n)[-n:]
        top_idx    = top_idx[np.argsort(scores[top_idx])[::-1]]
        top_scores = scores[top_idx]

        return top_idx, top_scores

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = BPR_FACTORS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":  self.model.state_dict(),
            "n_users":      self.model.user_embeddings.num_embeddings,
            "n_items":      self.model.item_embeddings.num_embeddings,
            "n_factors":    self.model.n_factors,
            "loss_history": self.train_loss_history,
        }, path)
        log.info(f"BPR model saved → {path}")

    @classmethod
    def load(cls, path: Path = BPR_FACTORS_PATH, device: torch.device = DEVICE) -> "BPR":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        trainer = cls(
            n_users   = ckpt["n_users"],
            n_items   = ckpt["n_items"],
            n_factors = ckpt["n_factors"],
            device    = device,
        )
        trainer.model.load_state_dict(ckpt["model_state"])
        trainer.train_loss_history = ckpt["loss_history"]
        log.info(f"BPR model loaded ← {path}")
        return trainer

    # ── Expose embeddings for hybrid model ────────────────────────────────────

    def get_user_embeddings_tensor(self) -> torch.Tensor:
        return self.model.user_embeddings.weight.detach().cpu()

    def get_item_embeddings_tensor(self) -> torch.Tensor:
        return self.model.item_embeddings.weight.detach().cpu()

    def __repr__(self) -> str:
        return (
            f"BPR(n_factors={self.model.n_factors}, "
            f"n_epochs={self.n_epochs}, reg={self.reg})"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle
    from src.models.als import ALS

    log.info("Loading BPR training data...")
    bpr_data       = np.load(BPR_DATA_PATH)
    all_items      = bpr_data["all_items"]
    item_pop_index = bpr_data["item_pop_index"]
    item_pop_vals  = bpr_data["item_pop_values"]

    with open(USER_POSITIVES_PATH, "rb") as f:
        user_positives = pickle.load(f)

    n_users = max(user_positives.keys()) + 1
    n_items = int(all_items.max()) + 1

    # Item popularity array aligned with all_items
    item_pop = np.zeros(len(all_items), dtype=np.float32)
    for idx, val in zip(item_pop_index, item_pop_vals):
        item_pop[np.where(all_items == idx)[0]] = val

    trainer = BPR(n_users=n_users, n_items=n_items)

    # Warm-start from ALS if available
    if ALS_PATH.exists():
        als = ALS.load()
        trainer.init_from_als(als)

    trainer.fit(user_positives, all_items, item_pop)
    trainer.save()