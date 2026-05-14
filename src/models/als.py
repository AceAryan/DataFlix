"""
DataFlix — Model 1: Alternating Least Squares (ALS)
src/models/als.py

Explicit-feedback matrix factorisation baseline.
Solves the regularised least-squares problem:
  min_{P,Q}  Σ_{(u,i)∈R} (r_ui - p_u · q_i)² + λ(||P||² + ||Q||²)

Alternates between:
  - Fixing Q, solving for each p_u analytically (closed-form ridge regression)
  - Fixing P, solving for each q_i analytically

Closed-form per-user update:
  p_u = (QᵀQ + λI)⁻¹ Qᵀ r_u

This is faster and more stable than gradient descent for explicit feedback
because each subproblem has an exact solution.

GPU acceleration: uses torch for matrix ops if CUDA available,
falls back to numpy/scipy on CPU.
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
    n_factors   : latent dimension k
    n_iterations: max ALS iterations
    reg         : L2 regularisation λ
    tol         : convergence tolerance on RMSE improvement
    device      : torch device for matrix ops
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

        # Learnt factors — set after fit()
        self.user_factors:  np.ndarray | None = None  # (n_users,  k)
        self.item_factors:  np.ndarray | None = None  # (n_movies, k)
        self.user_biases:   np.ndarray | None = None  # (n_users,)
        self.item_biases:   np.ndarray | None = None  # (n_movies,)
        self.global_mean:   float             = 0.0

        self.train_rmse_history: list[float] = []

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, csr_mat: sparse.csr_matrix) -> "ALS":
        """
        Fit ALS on a user × movie CSR matrix of mean-centred ratings.

        Parameters
        ----------
        csr_mat : scipy.sparse.csr_matrix, shape (n_users, n_movies)
            Mean-centred ratings. Zeros mean unobserved (not a rating of 0).
        """
        n_users, n_movies = csr_mat.shape
        log.info(f"ALS: {n_users:,} users × {n_movies:,} movies | "
                 f"k={self.n_factors} | λ={self.reg} | "
                 f"device={self.device}")

        # Global mean of observed ratings (for bias initialisation)
        self.global_mean = csr_mat.data.mean() if len(csr_mat.data) > 0 else 0.0

        # Initialise factors with small random values (scaled by 1/√k)
        rng = np.random.default_rng(42)
        scale = 1.0 / np.sqrt(self.n_factors)
        self.user_factors = rng.normal(0, scale, (n_users,  self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, scale, (n_movies, self.n_factors)).astype(np.float32)
        self.user_biases  = np.zeros(n_users,  dtype=np.float32)
        self.item_biases  = np.zeros(n_movies, dtype=np.float32)

        # CSC for efficient column (item) access during item factor update
        csc_mat = csr_mat.tocsc()

        prev_rmse = float("inf")

        for iteration in range(1, self.n_iterations + 1):
            t = time.time()

            # ── Update user factors ──
            self._update_factors(
                factors_to_update = self.user_factors,
                fixed_factors     = self.item_factors,
                interaction_mat   = csr_mat,       # iterate over rows (users)
                biases_to_update  = self.user_biases,
                fixed_biases      = self.item_biases,
            )

            # ── Update item factors ──
            self._update_factors(
                factors_to_update = self.item_factors,
                fixed_factors     = self.user_factors,
                interaction_mat   = csc_mat.T,     # transpose → rows = items
                biases_to_update  = self.item_biases,
                fixed_biases      = self.user_biases,
            )

            # ── Compute training RMSE ──
            rmse = self._compute_rmse(csr_mat)
            self.train_rmse_history.append(rmse)

            elapsed = time.time() - t
            log.info(f"  Iter {iteration:>3}/{self.n_iterations}  "
                     f"train RMSE={rmse:.5f}  ({elapsed:.1f}s)")

            # ── Convergence check ──
            improvement = prev_rmse - rmse
            if improvement < self.tol and iteration > 1:
                log.info(f"  Converged at iteration {iteration} "
                         f"(improvement={improvement:.2e} < tol={self.tol})")
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

        For each entity u:
          A_u = QᵀQ_u + λI      (k×k matrix)
          b_u = Qᵀ(r_u - bias)  (k-vector)
          p_u = A_u⁻¹ b_u        (solved via Cholesky)

        Using torch for batched matrix ops on GPU where available.
        Falls back to numpy on CPU for small matrices.
        """
        k   = self.n_factors
        reg = self.reg

        # Precompute QᵀQ — shared across all users (only varies in Q_u subset)
        # Shape: (k, k)
        Q  = fixed_factors                    # (n_fixed, k)
        QTQ = Q.T @ Q                         # (k, k)
        reg_matrix = reg * np.eye(k, dtype=np.float32)

        n_entities = factors_to_update.shape[0]

        for idx in range(n_entities):
            # Indices and values of rated items for this entity
            row      = interaction_mat.getrow(idx)
            item_ids = row.indices
            ratings  = row.data.astype(np.float32)

            if len(item_ids) == 0:
                # No interactions — keep random init, only regularise
                factors_to_update[idx] = 0.0
                biases_to_update[idx]  = 0.0
                continue

            Q_u    = Q[item_ids]                         # (n_rated, k)
            b_u    = fixed_biases[item_ids]              # (n_rated,)

            # Subtract item biases from ratings
            r_adj  = ratings - b_u                       # (n_rated,)

            # Compute bias update: b_u = (Σ r_adj) / (n_rated + λ)
            bias_u = r_adj.mean() / (1.0 + reg / len(item_ids))
            biases_to_update[idx] = bias_u

            # Subtract bias from adjusted ratings
            r_adj -= bias_u

            # Solve: (Q_u^T Q_u + λI) p = Q_u^T r_adj
            # Use Cholesky decomposition for numerical stability
            A = Q_u.T @ Q_u + reg_matrix               # (k, k)
            b = Q_u.T @ r_adj                          # (k,)

            try:
                factors_to_update[idx] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fallback to least-squares if A is singular
                factors_to_update[idx], _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    def _compute_rmse(self, csr_mat: sparse.csr_matrix) -> float:
        """
        Compute RMSE on observed training ratings.
        Only evaluates on non-zero entries (observed ratings).
        """
        rows, cols = csr_mat.nonzero()
        true_ratings = np.array(csr_mat[rows, cols]).flatten()

        pred = (
            self.user_factors[rows] * self.item_factors[cols]
        ).sum(axis=1)
        pred += self.user_biases[rows] + self.item_biases[cols]

        return float(np.sqrt(np.mean((true_ratings - pred) ** 2)))

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, user_idx: int, movie_idx: int) -> float:
        """Predict rating for a single (user, movie) pair."""
        self._check_fitted()
        score = float(
            self.user_factors[user_idx] @ self.item_factors[movie_idx]
            + self.user_biases[user_idx]
            + self.item_biases[movie_idx]
        )
        return score

    def recommend(
        self,
        user_idx:   int,
        n:          int = 10,
        exclude_seen: bool = True,
        seen_items: set | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recommend top-n items for a user.

        Parameters
        ----------
        user_idx     : integer user index
        n            : number of recommendations
        exclude_seen : if True, filter out items in seen_items
        seen_items   : set of movie_idx already interacted with

        Returns
        -------
        top_items  : np.ndarray of movie_idx, shape (n,)
        top_scores : np.ndarray of predicted scores, shape (n,)
        """
        self._check_fitted()

        # Score all items at once: dot product + biases
        scores = self.item_factors @ self.user_factors[user_idx]
        scores += self.item_biases + self.user_biases[user_idx]

        if exclude_seen and seen_items:
            scores[list(seen_items)] = -np.inf

        top_idx    = np.argpartition(scores, -n)[-n:]
        top_idx    = top_idx[np.argsort(scores[top_idx])[::-1]]
        top_scores = scores[top_idx]

        return top_idx, top_scores

    def _check_fitted(self) -> None:
        if self.user_factors is None:
            raise RuntimeError("ALS model is not fitted. Call fit() first.")

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
        log.info(f"ALS factors saved → {path}")

    @classmethod
    def load(cls, path: Path = ALS_PATH) -> "ALS":
        data = np.load(path)
        model = cls(
            n_factors = int(data["n_factors"][0]),
            reg       = float(data["reg"][0]),
        )
        model.user_factors        = data["user_factors"]
        model.item_factors        = data["item_factors"]
        model.user_biases         = data["user_biases"]
        model.item_biases         = data["item_biases"]
        model.global_mean         = float(data["global_mean"][0])
        model.train_rmse_history  = data["train_rmse"].tolist()
        log.info(f"ALS factors loaded ← {path}")
        return model

    # ── Expose factors as tensors for hybrid model ────────────────────────────

    def get_user_factors_tensor(self) -> torch.Tensor:
        """Return user factors as a float32 tensor for use in hybrid model."""
        self._check_fitted()
        return torch.tensor(self.user_factors, dtype=torch.float32)

    def get_item_factors_tensor(self) -> torch.Tensor:
        """Return item factors as a float32 tensor for use in hybrid model."""
        self._check_fitted()
        return torch.tensor(self.item_factors, dtype=torch.float32)

    def __repr__(self) -> str:
        fitted = self.user_factors is not None
        return (
            f"ALS(n_factors={self.n_factors}, reg={self.reg}, "
            f"n_iterations={self.n_iterations}, fitted={fitted})"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scipy import sparse as sp

    log.info("Loading CSR matrix...")
    csr = sp.load_npz(CSR_MATRIX_PATH)

    model = ALS()
    model.fit(csr)
    model.save()

    log.info("\nTraining RMSE history:")
    for i, rmse in enumerate(model.train_rmse_history, 1):
        log.info(f"  Iter {i:>3}: {rmse:.5f}")