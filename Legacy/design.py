# design.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import numpy as np

from panel import build_response_logpower, build_Z_block     # reuse Z,y builders  :contentReference[oaicite:5]{index=5}
from panel import build_B_block_time_of_day                  # ToD sin/cos + mask  :contentReference[oaicite:6]{index=6}

# ---- A-block with subject-level missing handled by mask OR imputation ----
def build_A_block_with_missing(
    T: int,
    sex_male1: Optional[float],
    age_years: Optional[float],
    task_rest1: Optional[float] = 1.0,
    center_age_at: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Returns:
      A_t: (T, pA), M_A: (T, pA) mask (1=observed/imputed, 0=missing),
      col_idx: column indices for interpretability.
    If sex/age are missing, set zeros and mark mask=0; EM can fill them later.
    """
    # columns: [1, sex, age_centered, task]
    pA = 4
    A = np.zeros((T, pA), dtype=float)
    M = np.zeros((T, pA), dtype=float)

    # intercept
    A[:, 0] = 1.0; M[:, 0] = 1.0
    # sex
    if sex_male1 is not None:
        A[:, 1] = float(sex_male1); M[:, 1] = 1.0
    # age
    if age_years is not None:
        A[:, 2] = float(age_years) - center_age_at; M[:, 2] = 1.0
    # task
    if task_rest1 is not None:
        A[:, 3] = float(task_rest1); M[:, 3] = 1.0

    col_idx = {"intercept": 0, "sex": 1, "age_centered": 2, "task": 3}
    return A, M, col_idx


def build_BZ_blocks(
    X: np.ndarray,
    sessions: List[str],
    Z_posteriors: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    B_t (ToD) and M_t mask from sessions; Z_t from provided vMF posteriors or identity.
    """
    B_t, M_t = build_B_block_time_of_day(sessions)     # (T,2) + (T,2) mask  :contentReference[oaicite:7]{index=7}
    if Z_posteriors is None:
        Z_t = np.ones((len(sessions), 1))              # fallback: single column (no vMF)
    else:
        Z_t = build_Z_block(Z_posteriors)              # (T,K)  :contentReference[oaicite:8]{index=8}
    return B_t, M_t, Z_t
