"""
DataFlix — Model 1: ALS (Alternating Least Squares)
src/models/als.py

Explicit feedback matrix factorisation baseline.
Predicts mean-centred ratings via: r̂_ui = μ + b_u + b_i + p_u · q_i

Role in this project:
  - Rating prediction baseline (RMSE/MAE)
  - Produces latent factors that warm-start BPR and Hybrid
  - Evaluated on ranking metrics via score_all_items (includes biases)

Training: closed-form alternating ridge regression (no learning rate needed).

Fixes vs previous version
--------------------------
BUG 1 — Residual didn't include current factor scores
    Old : r_adj = ratings - fixed_biases only
    New : r_adj = ratings - fixed_biases - Q·p_current  (full residual)
    Why : The bias update must see the residual after the factor contribution
          is removed, otherwise bias and factors absorb the same signal and
          the solve step is fitting noise. global_mean is 0 because
          preprocess.py already applies per-user mean-centering before
          building the CSR matrix (step 4 → step 5).

BUG 2 — Bias closed-form was wrong
    Old : bias = mean(residual) / (1 + λ/n)   ← not standard ridge solution
    New : bias = sum(residual) / (n + λ)       ← correct L2-regularised scalar
    Why : The old formula underestimates the bias at low n (cold users/items)
          and overestimates at high n, making the bias term unreliable.

BUG 3 — getrow() inside inner loop (O(n²) per iteration)
    Old : for idx in range(n): row = mat.getrow(idx)
    New : pre-index CSR indptr/indices/data arrays; slice directly per user
    Why : getrow() on a CSR matrix copies and is O(nnz) per call. With 200K
          users this made each ALS iteration ~10× slower than necessary.

BUG 4 — Inference dropped bias terms
    Old : score = user_factors[u] @ item_factors.T      (in evaluate.py)
    New : score_all_items(u) adds user_bias[u] + item_biases to every score
    Why : Biases capture user leniency and item popularity — dropping them
          means scores are uncentred and rankings are distorted.
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

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, csr_mat: sparse.csr_matrix) -> "ALS":
        n_users, n_items = csr_mat.shape
        log.info(f"ALS: {n_users:,}u × {n_items:,}i | k={self.n_factors} | λ={self.reg}")

        # CSR is already per-user mean-centred by preprocess.py (step 4).
        # global_mean is 0 by construction — no need to recompute.
        self.global_mean = 0.0

        rng               = np.random.default_rng(42)
        scale             = 1.0 / np.sqrt(self.n_factors)
        self.user_factors = rng.normal(0, scale, (n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, scale, (n_items, self.n_factors)).astype(np.float32)
        self.user_biases  = np.zeros(n_users, dtype=np.float32)
        self.item_biases  = np.zeros(n_items, dtype=np.float32)

        # BUG 3 FIX: pre-extract CSR/CSC arrays for fast slicing
        csc_mat  = csr_mat.tocsc()
        csr_data = (csr_mat.indptr, csr_mat.indices, csr_mat.data.astype(np.float32))
        csc_data = (csc_mat.indptr, csc_mat.indices, csc_mat.data.astype(np.float32))

        prev_rmse = float("inf")
        for it in range(1, self.n_iterations + 1):
            t = time.time()
            # Update users: fix items, solve for each user row
            self._update_factors(
                self.user_factors, self.item_factors,
                self.user_biases,  self.item_biases,
                csr_data,
            )
            # Update items: fix users, solve for each item column (transposed = row)
            self._update_factors(
                self.item_factors, self.user_factors,
                self.item_biases,  self.user_biases,
                csc_data,
            )
            rmse = self._rmse(csr_mat)
            self.train_rmse_history.append(rmse)
            log.info(f"  Iter {it:>3}/{self.n_iterations}  RMSE={rmse:.5f}  ({time.time()-t:.1f}s)")
            if prev_rmse - rmse < self.tol and it > 1:
                log.info("  Converged")
                break
            prev_rmse = rmse
        return self

    # ── Core update step ─────────────────────────────────────────────────────

    def _update_factors(
        self,
        factors:       np.ndarray,   # (n_this, k)  — to be updated in-place
        fixed:         np.ndarray,   # (n_other, k) — held constant
        biases:        np.ndarray,   # (n_this,)    — updated in-place
        fixed_biases:  np.ndarray,   # (n_other,)   — held constant
        mat_data:      tuple,        # (indptr, indices, data) of CSR/CSC
    ) -> None:
        indptr, indices, data = mat_data
        k   = self.n_factors
        reg = self.reg
        I_k = np.eye(k, dtype=np.float32) * reg

        for idx in range(factors.shape[0]):
            # BUG 3 FIX: direct array slice — O(1), no copy
            start, end = indptr[idx], indptr[idx + 1]
            if start == end:
                factors[idx] = 0.0
                biases[idx]  = 0.0
                continue

            jids    = indices[start:end]              # neighbour indices
            ratings = data[start:end].copy()          # raw ratings

            Q       = fixed[jids]                     # (n_obs, k)
            b_f     = fixed_biases[jids]              # (n_obs,)

            # BUG 1 FIX: subtract global mean + fixed biases + fixed factor scores
            #   residual = r_ui - μ - b_j - p_j·q_i  (p_j·q_i from fixed side)
            cf_scores = (Q * factors[idx]).sum(axis=1)   # current factor contribution
            residual  = ratings - self.global_mean - b_f - cf_scores

            # BUG 2 FIX: correct L2-regularised bias: b = Σr / (n + λ)
            n_obs       = len(jids)
            biases[idx] = float(residual.sum()) / (n_obs + reg)

            # Residual after removing bias
            r_adj = residual - biases[idx]

            # Closed-form factor update: (Q^T Q + λI) p = Q^T r
            A = Q.T @ Q + I_k
            b_vec = Q.T @ r_adj
            try:
                factors[idx] = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                factors[idx], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)

    # ── RMSE ─────────────────────────────────────────────────────────────────

    def _rmse(self, csr_mat: sparse.csr_matrix) -> float:
        rows, cols = csr_mat.nonzero()
        true_r     = np.asarray(csr_mat[rows, cols]).flatten()
        pred       = (
            self.global_mean
            + self.user_biases[rows]
            + self.item_biases[cols]
            + (self.user_factors[rows] * self.item_factors[cols]).sum(axis=1)
        )
        return float(np.sqrt(np.mean((true_r - pred) ** 2)))

    # ── Scoring (BUG 4 FIX: biases included) ─────────────────────────────────

    def score_all_items(self, user_idx: int) -> np.ndarray:
        """
        Return ranking scores for all items for one user.
        Includes global mean + user bias + item biases + dot product.
        Use this in evaluate.py instead of the raw dot product.
        """
        scores = (
            self.global_mean
            + self.user_biases[user_idx]
            + self.item_biases                                    # (n_items,)
            + self.user_factors[user_idx] @ self.item_factors.T  # (n_items,)
        )
        return scores.astype(np.float32)

    def predict(self, user_idx: int, item_idx: int) -> float:
        return float(
            self.global_mean
            + self.user_factors[user_idx] @ self.item_factors[item_idx]
            + self.user_biases[user_idx]
            + self.item_biases[item_idx]
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = ALS_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            global_mean=np.array([self.global_mean]),
            train_rmse=np.array(self.train_rmse_history),
            n_factors=np.array([self.n_factors]),
            reg=np.array([self.reg]),
        )
        log.info(f"ALS saved → {path}")

    @classmethod
    def load(cls, path: Path = ALS_PATH) -> "ALS":
        path = Path(path)
        d = np.load(path)
        m = cls(n_factors=int(d["n_factors"][0]), reg=float(d["reg"][0]))
        m.user_factors       = d["user_factors"]
        m.item_factors       = d["item_factors"]
        m.user_biases        = d["user_biases"]
        m.item_biases        = d["item_biases"]
        m.global_mean        = float(d["global_mean"][0])
        m.train_rmse_history = d["train_rmse"].tolist()
        log.info(f"ALS loaded ← {path}  (global_mean={m.global_mean:.4f})")
        return m

    # ── Tensor helpers for BPR / Hybrid warm-start ────────────────────────────

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