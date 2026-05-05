"""
DataFlix — ALS (Alternating Least Squares) Solver
Closed-form alternating updates for matrix factorisation.
"""

import numpy as np
from scipy import sparse
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    ALS_ITERATIONS, ALS_REG, ALS_CONVERGENCE_TOL, LATENT_DIM_K, SEED
)


class ALSSolver:
    """
    ALS Matrix Factorisation: M ≈ PQ^T
    
    For user u:
        p_u = (Σ_{i ∈ Ω_u} q_i q_i^T + λI)^{-1} Σ_{i ∈ Ω_u} r_{ui} q_i
    
    Convergence: relative decrease in L_MSE < 1e-4 over one sweep.
    """
    
    def __init__(self, n_users: int, n_items: int, k: int = LATENT_DIM_K,
                 reg: float = ALS_REG, n_iterations: int = ALS_ITERATIONS,
                 convergence_tol: float = ALS_CONVERGENCE_TOL, seed: int = SEED):
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.reg = reg
        self.n_iterations = n_iterations
        self.convergence_tol = convergence_tol
        
        np.random.seed(seed)
        self.P = np.random.normal(0, 0.01, (n_users, k)).astype(np.float32)   # User factors
        self.Q = np.random.normal(0, 0.01, (n_items, k)).astype(np.float32)   # Item factors
        self.b_u = np.zeros(n_users, dtype=np.float32)   # User biases
        self.b_i = np.zeros(n_items, dtype=np.float32)   # Item biases
        self.mu = 0.0  # Global mean
        
        self.losses = []
    
    @staticmethod
    def _coo_arrays(R: sparse.csr_matrix):
        """Return (rows, cols, vals) with explicit zeros eliminated."""
        R_coo = R.tocoo()
        mask = R_coo.data != 0
        return R_coo.row[mask], R_coo.col[mask], R_coo.data[mask]

    def _compute_loss(self, R: sparse.csr_matrix) -> float:
        """Compute regularised MSE loss over observed entries."""
        rows, cols, vals = self._coo_arrays(R)
        preds = self.mu + self.b_u[rows] + self.b_i[cols]
        preds += np.sum(self.P[rows] * self.Q[cols], axis=1)

        residuals = vals - preds
        mse = np.mean(residuals ** 2)
        reg = self.reg * (
            np.sum(self.P ** 2) + np.sum(self.Q ** 2) +
            np.sum(self.b_u ** 2) + np.sum(self.b_i ** 2)
        ) / len(rows)

        return mse + reg

    
    def _update_users(self, R: sparse.csr_matrix):
        """Fix Q, solve for P (closed-form per user)."""
        reg_I = self.reg * np.eye(self.k, dtype=np.float32)
        
        for u in range(self.n_users):
            # Get items rated by user u
            rated = R[u].indices
            if len(rated) == 0:
                continue
            
            Q_u = self.Q[rated]  # (n_rated, k)
            r_u = R[u].data - self.mu - self.b_i[rated] - self.b_u[u]
            
            A = Q_u.T @ Q_u + reg_I  # (k, k)
            b = Q_u.T @ r_u           # (k,)
            
            self.P[u] = np.linalg.solve(A, b)
    
    def _update_items(self, R: sparse.csr_matrix):
        """Fix P, solve for Q (closed-form per item)."""
        R_csc = R.tocsc()
        reg_I = self.reg * np.eye(self.k, dtype=np.float32)
        
        for i in range(self.n_items):
            rated = R_csc[:, i].indices
            if len(rated) == 0:
                continue
            
            P_i = self.P[rated]  # (n_rated, k)
            r_i = R_csc[:, i].data - self.mu - self.b_u[rated] - self.b_i[i]
            
            A = P_i.T @ P_i + reg_I
            b = P_i.T @ r_i
            
            self.Q[i] = np.linalg.solve(A, b)
    
    def _update_biases(self, R: sparse.csr_matrix):
        """Update user and item biases (vectorized)."""
        rows, cols, vals = self._coo_arrays(R)
        residuals = vals - self.mu - np.sum(self.P[rows] * self.Q[cols], axis=1)

        # User biases
        user_num = np.bincount(rows, weights=residuals, minlength=self.n_users)
        user_den = np.bincount(rows, minlength=self.n_users) + self.reg
        self.b_u = (user_num / user_den).astype(np.float32)

        # Update residuals with new user biases
        residuals = vals - self.mu - self.b_u[rows] - np.sum(self.P[rows] * self.Q[cols], axis=1)

        # Item biases
        item_num = np.bincount(cols, weights=residuals, minlength=self.n_items)
        item_den = np.bincount(cols, minlength=self.n_items) + self.reg
        self.b_i = (item_num / item_den).astype(np.float32)
    
    def fit(self, R: sparse.csr_matrix, verbose: bool = True):
        """
        Train ALS.
        
        Args:
            R: Sparse user-item rating matrix (centered).
        """
        self.mu = R.data.mean() if len(R.data) > 0 else 0.0
        
        prev_loss = float("inf")
        
        for iteration in range(self.n_iterations):
            t0 = time.time()
            
            # Alternating updates
            self._update_users(R)
            self._update_items(R)
            self._update_biases(R)
            
            # Compute loss
            loss = self._compute_loss(R)
            self.losses.append(loss)
            elapsed = time.time() - t0
            
            # Convergence check (guard against inf on first iter)
            if prev_loss == float("inf"):
                rel_decrease = float("inf")
            else:
                rel_decrease = (prev_loss - loss) / (abs(prev_loss) + 1e-8)
            
            if verbose:
                print(f"  ALS iter {iteration+1}/{self.n_iterations}: "
                      f"loss={loss:.4f}, rel_decrease={rel_decrease:.6f}, "
                      f"time={elapsed:.1f}s")
            
            if 0 < rel_decrease < self.convergence_tol:
                print(f"  Converged at iteration {iteration+1}")
                break
            
            prev_loss = loss
    
    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Predict ratings for given (user, item) pairs."""
        return (self.mu + self.b_u[user_idx] + self.b_i[item_idx] +
                np.sum(self.P[user_idx] * self.Q[item_idx], axis=1))
    
    def predict_user(self, user_idx: int) -> np.ndarray:
        """Predict ratings for all items for a given user."""
        return (self.mu + self.b_u[user_idx] + self.b_i +
                self.P[user_idx] @ self.Q.T)
    
    def get_embeddings(self):
        """Return user and item latent factors."""
        return self.P, self.Q, self.b_u, self.b_i, self.mu
