from __future__ import annotations
import numpy as np
from typing import Tuple
from ife import IFEModel


def posterior_sex_probability(y_it: np.ndarray, fit: IFEModel,
                              A_t: np.ndarray, B_t: np.ndarray, Z_t: np.ndarray, M_t: np.ndarray,
                              sex_col_idx_in_A: int = 1, prior_pi: float = 0.5) -> float:
    """Compute P(sex=1 | data, model) under homoskedastic Gaussian errors."""
    rows = np.concatenate([A_t, M_t * B_t, Z_t], axis=1)
    R = y_it - (rows @ fit.beta)[None, :] - fit.Lam @ fit.F.T
    sigma2 = float(np.var(R) + 1e-12)

    A1, A0 = A_t.copy(), A_t.copy()
    A1[:, sex_col_idx_in_A] = 1.0
    A0[:, sex_col_idx_in_A] = 0.0
    rows1 = np.concatenate([A1, M_t * B_t, Z_t], axis=1)
    rows0 = np.concatenate([A0, M_t * B_t, Z_t], axis=1)

    mu1 = rows1 @ fit.beta + fit.Lam @ fit.F.T
    mu0 = rows0 @ fit.beta + fit.Lam @ fit.F.T
    diff = ((y_it - mu0) ** 2 - (y_it - mu1) ** 2).sum() / (2.0 * sigma2)
    log_odds = np.log(prior_pi) - np.log(1 - prior_pi) + diff
    return float(1.0 / (1.0 + np.exp(-log_odds)))


def estimate_time_of_day(y_it: np.ndarray, fit: IFEModel,
                         A_t: np.ndarray, B_t_other: np.ndarray, Z_t: np.ndarray,
                         beta_idx_sin: int, beta_idx_cos: int,
                         omega: float = 2 * np.pi / 24.0, grid_minutes: int = 10) -> np.ndarray:
    """
    1-D LS inversion when ToD enters as sin/cos. Returns tau_hat (T,) hours.

    IMPORTANT: beta_idx_sin, beta_idx_cos are the positions (in the FULL beta vector)
    of the ToD sin/cos coefficients, assuming the full design order is:
        [ A | B_other | ToD(sin, cos) | Z ]   (or any order you used, as long as indices match).
    We reconstruct a full design with ToD columns set to ZERO to form xb_base.
    """
    T = A_t.shape[0]
    # Construct the "no-ToD" matrix first
    rows_no_tod = np.concatenate([A_t, B_t_other, Z_t], axis=1)  # (T, p_without_ToD)

    # Insert two ZERO columns at the exact ToD positions so cols == len(beta)
    idx_lo, idx_hi = sorted([beta_idx_sin, beta_idx_cos])
    left  = rows_no_tod[:, :idx_lo]
    right = rows_no_tod[:, idx_lo:]
    zeros_tod = np.zeros((T, 2))
    rows_full = np.concatenate([left, zeros_tod, right], axis=1)  # (T, p_full)
    if rows_full.shape[1] != fit.beta.size:
        raise ValueError(
            f"Full design has {rows_full.shape[1]} cols but beta has {fit.beta.size}. "
            "Check beta_idx_sin/beta_idx_cos and the intended column order."
        )

    # Base mean (with ToD turned off)
    xb_base = rows_full @ fit.beta  # (T,)

    # Average across channels and remove factor part to stabilize
    y_bar = y_it.mean(axis=0) - (fit.Lam @ fit.F.T).mean(axis=0)  # (T,)

    # Grid over time-of-day and pick the best match per t
    minute_grid = np.arange(0, 24 * 60, grid_minutes)
    tau_grid = minute_grid / 60.0
    sin_grid = np.sin(omega * tau_grid)
    cos_grid = np.cos(omega * tau_grid)

    b_sin = fit.beta[beta_idx_sin]
    b_cos = fit.beta[beta_idx_cos]
    effect_grid = b_sin * sin_grid + b_cos * cos_grid  # (G,)

    tau_hat = np.zeros(T)
    for t in range(T):
        resid = y_bar[t] - xb_base[t]
        j = int(np.argmin((resid - effect_grid) ** 2))
        tau_hat[t] = float(tau_grid[j])
    return tau_hat
