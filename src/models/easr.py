"""
DataFlix — EASE^R Model
src/models/easr.py

Embarrassingly Shallow AutoEncoders for Sparse Data (Steck, 2019)
Closed-form solution: B = I - P · diag(1/diag(P))^{-1}
where P = (X^T X + λI)^{-1}

No GPU needed. Trains in minutes on ML-32M via numpy/scipy.
"""

import logging
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)

EASR_PATH = None  # Set from config at import time; overridden in train.py / evaluate.py


class EASR:
    """
    EASE^R: Embarrassingly Shallow AutoEncoder for Recommendation.

    Parameters
    ----------
    reg : float
        L2 regularisation λ. Typical sweet-spot for MovieLens: 200–500.
        Higher → smoother / less overfit; lower → sharper but noisier.
    """

    def __init__(self, reg: float = 350.0):
        self.reg = reg
        self.B: np.ndarray | None = None          # item–item weight matrix  (n_items × n_items)
        self.user_factors: np.ndarray | None = None  # X · B  cached after fit (n_users × n_items)

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, X: sp.csr_matrix) -> "EASR":
        """
        Fit EASE^R from a user–item CSR matrix X (implicit: 0/1 or explicit ratings).

        Steps
        -----
        1. Compute G = X^T X  (item–item Gram matrix)
        2. Add λ to diagonal  → G_reg
        3. Invert G_reg  (dense, O(n_items^3) — feasible for ≤100 K items)
        4. B = I – P · diag(1/diag(P))^{-1}  with diagonal zeroed
        5. Cache score matrix  U = X · B

        ML-32M has ~87 K items → inversion is ~87K^3 FP ops.
        Expect 2–5 min on a modern CPU with numpy backed by LAPACK/MKL.
        """
        t0 = time.time()
        n_users, n_items = X.shape
        log.info(f"  EASE^R fit: {n_users:,} users × {n_items:,} items  λ={self.reg}")

        # Step 1+2 — Gram + regularisation
        log.info("  Computing G = X^T X ...")
        G = np.array((X.T @ X).todense(), dtype=np.float64)
        diag_idx = np.arange(n_items)
        G[diag_idx, diag_idx] += self.reg

        # Step 3 — Invert (uses LAPACK dgesv / dpotrf under the hood)
        log.info("  Inverting G ...")
        P = np.linalg.inv(G)
        del G

        # Step 4 — Closed-form weights, zero diagonal
        log.info("  Building B ...")
        B = P / (-np.diag(P))                  # scale columns
        B[diag_idx, diag_idx] = 0.0            # enforce zero self-connections
        self.B = B.astype(np.float32)
        del P

        # Step 5 — Cache user scores  (float32 dense: 200K × 87K ≈ ~66 GB — too large!)
        # We do NOT cache the full score matrix; instead we store X as float32 CSR
        # and compute scores on-the-fly per user in score_all_items().
        # This keeps RAM usage at ~B size (~29 GB for 87K^2 float32).
        # Tip: if RAM is tight, reduce to float16 or chunk B column-wise.
        self._X = X.astype(np.float32)

        elapsed = time.time() - t0
        log.info(f"  EASE^R fit done in {elapsed/60:.1f}m  |  B shape: {self.B.shape}")
        return self

    # ------------------------------------------------------------------ #
    #  Scoring                                                             #
    # ------------------------------------------------------------------ #

    def score_all_items(self, user_idx: int) -> np.ndarray:
        """Return raw scores for all items for a single user. Shape: (n_items,)"""
        # X[u] is a sparse row → convert to dense 1-D array, then dot with B
        x_u = np.asarray(self._X[user_idx].todense(), dtype=np.float32).ravel()  # (n_items,)
        return x_u @ self.B   # (n_items,)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            B=self.B,
            reg=np.array([self.reg]),
        )
        # Save X separately as sparse (.npz)
        sp.save_npz(str(path).replace(".npz", "_X.npz"), self._X)
        log.info(f"  EASE^R saved → {path.name}")

    @classmethod
    def load(cls, path: Path) -> "EASR":
        path = Path(path)
        data = np.load(path)
        obj = cls(reg=float(data["reg"][0]))
        obj.B = data["B"].astype(np.float32)
        x_path = Path(str(path).replace(".npz", "_X.npz"))
        if x_path.exists():
            obj._X = sp.load_npz(x_path).astype(np.float32)
        else:
            log.warning("  EASE^R: X matrix not found — score_all_items will fail.")
        log.info(f"  EASE^R loaded ← {path.name}  λ={obj.reg}")
        return obj


# ── Convenience scorer (matches the pattern used in evaluate.py) ────────────

def score_easr(model: EASR, user_idx: int) -> np.ndarray:
    """Drop-in scorer compatible with evaluate.py's score_fn API."""
    return model.score_all_items(user_idx)