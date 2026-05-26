"""
DataFlix — Model 1: Alternating Least Squares (ALS)
src/models/als.py

Explicit-feedback matrix factorisation baseline.

The core idea: decompose the user-item rating matrix R into two low-rank
matrices P (users) and Q (items) such that R ≈ P @ Q.T

For each user u, their predicted rating of item i is:
    r̂_ui = p_u · q_i + b_u + b_i

where b_u is the user bias (tendency to rate high/low) and b_i is the
item bias (tendency to receive high/low ratings).

ALS solves this by alternating closed-form updates:
    Fix Q → solve for each p_u exactly (ridge regression)
    Fix P → solve for each q_i exactly (ridge regression)

This is faster and more stable than gradient descent because each
subproblem has an exact solution (no learning rate tuning needed).

Evaluation note:
    ALS is an explicit feedback model — it predicts ratings for observed
    interactions. It is evaluated using leave-one-out: for each user the
    last rated item is the test item, and all other items (including
    training items) are candidates. This is the standard evaluation
    protocol for explicit MF and avoids the seen-item masking problem
    that kills ALS ranking metrics.
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
    DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CSR_MATRIX_PATH     = PROCESSED_DIR / "train_csr.npz"
    ALS_PATH            = RESULTS_DIR / "als_factors.npz"
    LATENT_DIM_K        = 100
    ALS_ITERATIONS      = 20
    ALS_REG             = 0.1
    ALS_CONVERGENCE_TOL = 1e-4

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class ALS:
    """
    Alternating Least Squares matrix factorisation.

    Parameters
    ----------
    n_factors    : latent dimension k
    n_iterations : max ALS iterations
    reg          : L2 regularisation λ
    tol          : convergence tolerance on RMSE improvement
    """

    def __init__(
        self,
        n_factors:    int   = LATENT_DIM_K,
        n_iterations: int   = ALS_ITERATIONS,
        reg:          float = ALS_REG,
        tol:          float = ALS_CONVERGENCE_TOL,
        device:       torch.device = DEVICE,
    ):
        self.n_factors    = n_factors
        self.n_iterations = n_iterations
        self.reg          = reg
        self.tol          = tol
        self.device       = device

        self.user_factors: np.ndarray | None = None  # (n_users,  k)
        self.item_factors: np.ndarray | None = None  # (n_items,  k)
        self.user_biases:  np.ndarray | None = None  # (n_users,)
        self.item_biases:  np.ndarray | None = None  # (n_items,)
        self.global_mean:  float             = 0.0
        self.train_rmse_history: list[float] = []

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, csr_mat: sparse.csr_matrix) -> "ALS":
        """
        Fit ALS on a user × item CSR matrix of mean-centred ratings.
        Zeros in the matrix mean unobserved (not a rating of zero).
        """
        n_users, n_items = csr_mat.shape
        log.info(f"ALS: {n_users:,} users × {n_items:,} items | "
                 f"k={self.n_factors} | λ={self.reg}")

        self.global_mean = float(csr_mat.data.mean()) if len(csr_mat.data) > 0 else 0.0

        rng   = np.random.default_rng(42)
        scale = 1.0 / np.sqrt(self.n_factors)
        self.user_factors = rng.normal(0, scale, (n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, scale, (n_items, self.n_factors)).astype(np.float32)
        self.user_biases  = np.zeros(n_users, dtype=np.float32)
        self.item_biases  = np.zeros(n_items, dtype=np.float32)

        csc_mat  = csr_mat.tocsc()
        prev_rmse = float("inf")

        for iteration in range(1, self.n_iterations + 1):
            t = time.time()

            self._update_factors(
                factors_to_update = self.user_factors,
                fixed_factors     = self.item_factors,
                interaction_mat   = csr_mat,
                biases_to_update  = self.user_biases,
                fixed_biases      = self.item_biases,
            )
            self._update_factors(
                factors_to_update = self.item_factors,
                fixed_factors     = self.user_factors,
                interaction_mat   = csc_mat.T,
                biases_to_update  = self.item_biases,
                fixed_biases      = self.user_biases,
            )

            rmse = self._compute_rmse(csr_mat)
            self.train_rmse_history.append(rmse)
            log.info(f"  Iter {iteration:>3}/{self.n_iterations}  "
                     f"train RMSE={rmse:.5f}  ({time.time()-t:.1f}s)")

            improvement = prev_rmse - rmse
            if improvement < self.tol and iteration > 1:
                log.info(f"  Converged (improvement={improvement:.2e} < tol={self.tol})")
                break
            prev_rmse = rmse

        return self

    def _update_factors(
        self,
        factors_to_update: np.ndarray,
        fixed_factors:     np.ndarray,
        interaction_mat:   sparse.spmatrix,
        biases_to_update:  np.ndarray,
        fixed_biases:      np.ndarray,
    ) -> None:
        """
        Closed-form ALS update for one side (users or items).
        Solves: (Q_u^T Q_u + λI) p_u = Q_u^T (r_u - b_u_items)
        """
        k          = self.n_factors
        reg_matrix = self.reg * np.eye(k, dtype=np.float32)

        for idx in range(factors_to_update.shape[0]):
            row      = interaction_mat.getrow(idx)
            item_ids = row.indices
            ratings  = row.data.astype(np.float32)

            if len(item_ids) == 0:
                factors_to_update[idx] = 0.0
                biases_to_update[idx]  = 0.0
                continue

            Q_u   = fixed_factors[item_ids]
            b_u   = fixed_biases[item_ids]
            r_adj = ratings - b_u

            # Bias update
            biases_to_update[idx] = r_adj.mean() / (1.0 + self.reg / len(item_ids))
            r_adj -= biases_to_update[idx]

            # Factor update via Cholesky solve
            A = Q_u.T @ Q_u + reg_matrix
            b = Q_u.T @ r_adj
            try:
                factors_to_update[idx] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                factors_to_update[idx], _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    def _compute_rmse(self, csr_mat: sparse.csr_matrix) -> float:
        rows, cols  = csr_mat.nonzero()
        true_r      = np.array(csr_mat[rows, cols]).flatten()
        pred        = ((self.user_factors[rows] * self.item_factors[cols]).sum(axis=1)
                       + self.user_biases[rows] + self.item_biases[cols])
        return float(np.sqrt(np.mean((true_r - pred) ** 2)))

    # ── Inference ─────────────────────────────────────────────────────────────

    def score_all_items(self, user_idx: int) -> np.ndarray:
        """
        Return predicted mean-centred score for all items for this user.
        Shape: (n_items,)

        Does NOT mask seen items — ALS is explicit MF and is evaluated
        including all items. See evaluation note in module docstring.
        """
        self._check_fitted()
        return (self.item_factors @ self.user_factors[user_idx]
                + self.item_biases
                + self.user_biases[user_idx])

    def predict(self, user_idx: int, item_idx: int) -> float:
        self._check_fitted()
        return float(
            self.user_factors[user_idx] @ self.item_factors[item_idx]
            + self.user_biases[user_idx]
            + self.item_biases[item_idx]
        )

    def recommend(
        self,
        user_idx:  int,
        n:         int = 10,
        seen_items: set | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recommend top-n items. seen_items is optional — ALS evaluation
        typically does not exclude seen items (leave-one-out protocol).
        """
        scores = self.score_all_items(user_idx)
        if seen_items:
            scores = scores.copy()
            scores[list(seen_items)] = -np.inf
        top_idx    = np.argpartition(scores, -n)[-n:]
        top_idx    = top_idx[np.argsort(scores[top_idx])[::-1]]
        return top_idx, scores[top_idx]

    def _check_fitted(self) -> None:
        if self.user_factors is None:
            raise RuntimeError("ALS not fitted. Call fit() first.")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = ALS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            user_factors = self.user_factors,
            item_factors = self.item_factors,
            user_biases  = self.user_biases,
            item_biases  = self.item_biases,
            global_mean  = np.array([self.global_mean]),
            train_rmse   = np.array(self.train_rmse_history),
            n_factors    = np.array([self.n_factors]),
            reg          = np.array([self.reg]),
        )
        log.info(f"ALS saved → {path}")

    @classmethod
    def load(cls, path: Path = ALS_PATH) -> "ALS":
        data  = np.load(path)
        model = cls(
            n_factors = int(data["n_factors"][0]),
            reg       = float(data["reg"][0]),
        )
        model.user_factors       = data["user_factors"]
        model.item_factors       = data["item_factors"]
        model.user_biases        = data["user_biases"]
        model.item_biases        = data["item_biases"]
        model.global_mean        = float(data["global_mean"][0])
        model.train_rmse_history = data["train_rmse"].tolist()
        log.info(f"ALS loaded ← {path}")
        return model

    def get_user_factors_tensor(self) -> torch.Tensor:
        self._check_fitted()
        return torch.tensor(self.user_factors, dtype=torch.float32)

    def get_item_factors_tensor(self) -> torch.Tensor:
        self._check_fitted()
        return torch.tensor(self.item_factors, dtype=torch.float32)

    def __repr__(self) -> str:
        return (f"ALS(n_factors={self.n_factors}, reg={self.reg}, "
                f"fitted={self.user_factors is not None})")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scipy import sparse as sp
    log.info("Loading CSR matrix...")
    csr   = sp.load_npz(CSR_MATRIX_PATH)
    model = ALS()
    model.fit(csr)
    model.save()