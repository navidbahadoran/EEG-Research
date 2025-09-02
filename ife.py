from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np

@dataclass
class IFEModel:
    beta: np.ndarray     # (p,)
    F: np.ndarray        # (T,r)
    Lam: np.ndarray      # (N,r)
    loss: float
    rank_r: int
    coef_names: List[str]

def masked_beta_step_fast(y_it, A_t, B_t, Z_t, M_t, Lam, F):
    """
    Vectorized beta-step: design row is identical across channels for each t.
    """
    N, T = y_it.shape
    rows = np.concatenate([A_t, M_t * B_t, Z_t], axis=1)  # (T,p)

    sum_y_t = y_it.sum(axis=0)           # (T,)
    sum_lam = Lam.sum(axis=0)            # (r,)
    proj = F @ sum_lam                   # (T,)
    sum_y_star_t = sum_y_t - proj        # (T,)

    XtX = (rows[:, :, None] * rows[:, None, :]).sum(axis=0) * N
    Xty = (rows * sum_y_star_t[:, None]).sum(axis=0)
    beta = np.linalg.pinv(XtX) @ Xty
    return beta

def factor_step_fast(y_it, beta, A_t, B_t, Z_t, M_t, r):
    rows = np.concatenate([A_t, M_t * B_t, Z_t], axis=1)
    xb = rows @ beta
    R = y_it - xb[None, :]
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    U_r, S_r, Vt_r = U[:, :r], S[:r], Vt[:r, :]
    Tlen = R.shape[1]
    F = np.sqrt(Tlen) * Vt_r.T
    Lam = (S_r[None, :] * U_r) / np.sqrt(Tlen)
    loss = float(np.mean(R**2))
    return R, F, Lam, loss

def ife_fit_masked(y_it, A_t, B_t, Z_t, M_t, r,
                   coef_names: Optional[List[str]] = None,
                   max_iter: int = 30, tol: float = 1e-6) -> IFEModel:
    """ALS: init on [A,Z] then alternate beta/factor steps."""
    N, T = y_it.shape
    # init on [A,Z]
    X_AZ = np.column_stack([A_t, Z_t])
    XtX = np.zeros((X_AZ.shape[1], X_AZ.shape[1]))
    Xty = np.zeros(X_AZ.shape[1])
    for i in range(N):
        XtX += X_AZ.T @ X_AZ
        Xty += X_AZ.T @ y_it[i]
    beta_AZ = np.linalg.pinv(XtX) @ Xty
    xb0 = X_AZ @ beta_AZ
    R0 = y_it - xb0[None, :]
    U0, S0, Vt0 = np.linalg.svd(R0, full_matrices=False)
    U0_r, S0_r, Vt0_r = U0[:, :r], S0[:r], Vt0[:r, :]
    F = np.sqrt(T) * Vt0_r.T
    Lam = (S0_r[None, :] * U0_r) / np.sqrt(T)

    last_loss = np.inf
    for _ in range(max_iter):
        beta = masked_beta_step_fast(y_it, A_t, B_t, Z_t, M_t, Lam, F)
        R, F, Lam, loss = factor_step_fast(y_it, beta, A_t, B_t, Z_t, M_t, r)
        if abs(last_loss - loss) <= tol * (1 + last_loss):
            break
        last_loss = loss

    if coef_names is None:
        pA, pB, pZ = A_t.shape[1], B_t.shape[1], Z_t.shape[1]
        coef_names = (
            [f"A:{i}" for i in range(pA)] +
            [f"B:{i}" for i in range(pB)] +
            [f"Z:{i}" for i in range(pZ)]
        )
    return IFEModel(beta=beta, F=F, Lam=Lam, loss=loss, rank_r=r, coef_names=coef_names)

def ife_ic_bai_ng(y_it, A_t, B_t, Z_t, M_t, r_grid: List[int]) -> Tuple[int, Dict[int, float], Dict[int, IFEModel]]:
    """Select rank via a Bai–Ng-like IC; return best_r, all ICs, all fits."""
    N, T = y_it.shape
    ic_vals: Dict[int, float] = {}
    fits: Dict[int, IFEModel] = {}
    for r in r_grid:
        fit = ife_fit_masked(y_it, A_t, B_t, Z_t, M_t, r=r)
        rows = np.concatenate([A_t, M_t * B_t, Z_t], axis=1)
        R = y_it - (rows @ fit.beta)[None, :] - fit.Lam @ fit.F.T
        mse = float(np.mean(R**2))
        ic = np.log(mse) + r * (N + T) / (N * T) * np.log((N * T) / (N + T))
        fits[r] = fit
        ic_vals[r] = ic
    best_r = min(ic_vals, key=ic_vals.get)
    return best_r, ic_vals, fits
