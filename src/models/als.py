"""
DataFlix — Model 1: ALS (Alternating Least Squares)
src/models/als.py

Explicit feedback matrix factorisation baseline.
Predicts mean-centred ratings via: r̂_ui = p_u · q_i + b_u + b_i

Role in this project:
  - Rating prediction baseline (RMSE/MAE)
  - Produces latent factors that warm-start BPR and Hybrid
  - NOT evaluated on ranking metrics (ALS is not a ranking model)

Training: closed-form alternating ridge regression (no learning rate needed).
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

try:
    from src.config import (
        PROCESSED_DIR, RESULTS_DIR, DEVICE,
        CSR_MATRIX_PATH, ALS_PATH,
        LATENT_DIM_K, ALS_ITERATIONS, ALS_REG, ALS_CONVERGENCE_TOL,
    )
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR       = _ROOT / "data/processed"
    RESULTS_DIR         = _ROOT / "results"
    DEVICE              = torch.device("cpu")
    CSR_MATRIX_PATH     = PROCESSED_DIR / "train_csr.npz"
    ALS_PATH            = RESULTS_DIR / "als_factors.npz"
    LATENT_DIM_K        = 128
    ALS_ITERATIONS      = 20
    ALS_REG             = 0.1
    ALS_CONVERGENCE_TOL = 1e-4

log = logging.getLogger(__name__)


class ALS:

    def __init__(
        self,
        n_factors:    int   = LATENT_DIM_K,
        n_iterations: int   = ALS_ITERATIONS,
        reg:          float = ALS_REG,
        tol:          float = ALS_CONVERGENCE_TOL,
    ):
        self.n_factors    = n_factors
        self.n_iterations = n_iterations
        self.reg          = reg
        self.tol          = tol

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_biases:  np.ndarray | None = None
        self.item_biases:  np.ndarray | None = None
        self.global_mean:  float             = 0.0
        self.train_rmse_history: list[float] = []

    def fit(self, csr_mat: sparse.csr_matrix) -> "ALS":
        n_users, n_items = csr_mat.shape
        log.info(f"ALS: {n_users:,}u × {n_items:,}i | k={self.n_factors} | λ={self.reg}")

        self.global_mean  = float(csr_mat.data.mean()) if len(csr_mat.data) else 0.0
        rng               = np.random.default_rng(42)
        scale             = 1.0 / np.sqrt(self.n_factors)
        self.user_factors = rng.normal(0, scale, (n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, scale, (n_items, self.n_factors)).astype(np.float32)
        self.user_biases  = np.zeros(n_users, dtype=np.float32)
        self.item_biases  = np.zeros(n_items, dtype=np.float32)
        csc_mat           = csr_mat.tocsc()
        prev_rmse         = float("inf")

        for it in range(1, self.n_iterations + 1):
            t = time.time()
            self._update(self.user_factors, self.item_factors,
                         csr_mat,   self.user_biases, self.item_biases)
            self._update(self.item_factors, self.user_factors,
                         csc_mat.T, self.item_biases, self.user_biases)
            rmse = self._rmse(csr_mat)
            self.train_rmse_history.append(rmse)
            log.info(f"  Iter {it:>3}/{self.n_iterations}  RMSE={rmse:.5f}  ({time.time()-t:.1f}s)")
            if prev_rmse - rmse < self.tol and it > 1:
                log.info("  Converged")
                break
            prev_rmse = rmse
        return self

    def _update(self, factors, fixed, mat, biases, fixed_biases):
        k   = self.n_factors
        reg = self.reg * np.eye(k, dtype=np.float32)
        for idx in range(factors.shape[0]):
            row     = mat.getrow(idx)
            iids    = row.indices
            ratings = row.data.astype(np.float32)
            if len(iids) == 0:
                factors[idx] = 0.0
                biases[idx]  = 0.0
                continue
            Q      = fixed[iids]
            b_f    = fixed_biases[iids]
            r_adj  = ratings - b_f
            biases[idx] = r_adj.mean() / (1.0 + reg[0,0] / len(iids))
            r_adj -= biases[idx]
            A = Q.T @ Q + reg
            b = Q.T @ r_adj
            try:
                factors[idx] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                factors[idx], _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    def _rmse(self, csr_mat):
        rows, cols = csr_mat.nonzero()
        true_r     = np.array(csr_mat[rows, cols]).flatten()
        pred       = ((self.user_factors[rows] * self.item_factors[cols]).sum(1)
                      + self.user_biases[rows] + self.item_biases[cols])
        return float(np.sqrt(np.mean((true_r - pred) ** 2)))

    def predict(self, user_idx: int, item_idx: int) -> float:
        return float(
            self.user_factors[user_idx] @ self.item_factors[item_idx]
            + self.user_biases[user_idx]
            + self.item_biases[item_idx]
        )

    def save(self, path: Path = ALS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 user_factors=self.user_factors, item_factors=self.item_factors,
                 user_biases=self.user_biases,   item_biases=self.item_biases,
                 global_mean=np.array([self.global_mean]),
                 train_rmse=np.array(self.train_rmse_history),
                 n_factors=np.array([self.n_factors]),
                 reg=np.array([self.reg]))
        log.info(f"ALS saved → {path}")

    @classmethod
    def load(cls, path: Path = ALS_PATH) -> "ALS":
        d = np.load(path)
        m = cls(n_factors=int(d["n_factors"][0]), reg=float(d["reg"][0]))
        m.user_factors       = d["user_factors"]
        m.item_factors       = d["item_factors"]
        m.user_biases        = d["user_biases"]
        m.item_biases        = d["item_biases"]
        m.global_mean        = float(d["global_mean"][0])
        m.train_rmse_history = d["train_rmse"].tolist()
        log.info(f"ALS loaded ← {path}")
        return m

    def get_user_factors_tensor(self) -> torch.Tensor:
        return torch.tensor(self.user_factors, dtype=torch.float32)

    def get_item_factors_tensor(self) -> torch.Tensor:
        return torch.tensor(self.item_factors, dtype=torch.float32)

    def __repr__(self):
        return f"ALS(k={self.n_factors}, reg={self.reg}, fitted={self.user_factors is not None})"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    csr = sparse.load_npz(CSR_MATRIX_PATH)
    ALS().fit(csr).save()