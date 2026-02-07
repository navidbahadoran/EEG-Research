"""varx.py

Simple, teachable baseline: **ridge-stabilized VARX** with L=1 or 2 lags.

We model a multichannel EEG feature matrix Y (T x C) as

    Y_t = sum_{l=1..L} Y_{t-l} A_l + X_t B + 1 * c' + eps_t

where:
  - A_l is (C x C)
  - B is (p_exog x C)
  - c is (C,)

This is intentionally *simple* (no fixed effects, no latent factors) so
undergrads can understand and extend it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np


@dataclass
class VARXFit:
    lags: int
    A_list: List[np.ndarray]          # each (C,C)
    B: Optional[np.ndarray]           # (p_exog, C) or None
    intercept: np.ndarray             # (C,)
    ridge: float

    def predict(self, Y: np.ndarray, X_exog: Optional[np.ndarray] = None) -> np.ndarray:
        """One-step-ahead fitted values for t >= lags (returns (T, C), with first lags rows NaN)."""
        T, C = Y.shape
        if self.lags < 1:
            raise ValueError("lags must be >= 1")
        if X_exog is not None and X_exog.shape[0] != T:
            raise ValueError("X_exog must have same number of rows as Y")

        Yhat = np.full((T, C), np.nan, dtype=float)
        for t in range(self.lags, T):
            pred = self.intercept.copy()
            for ell, A in enumerate(self.A_list, start=1):
                pred += Y[t - ell] @ A
            if self.B is not None:
                pred += X_exog[t] @ self.B
            Yhat[t] = pred
        return Yhat


def _build_design(Y: np.ndarray, X_exog: Optional[np.ndarray], lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, Y_target) where
    - X is (T-lags, 1 + C*lags + p_exog)
    - Y_target is (T-lags, C)
    """
    T, C = Y.shape
    if lags < 1 or lags >= T:
        raise ValueError("lags must be in [1, T-1]")

    p_exog = 0
    if X_exog is not None:
        if X_exog.shape[0] != T:
            raise ValueError("X_exog must have same number of rows as Y")
        p_exog = X_exog.shape[1]

    X = np.zeros((T - lags, 1 + C * lags + p_exog), dtype=float)
    Yt = Y[lags:, :].astype(float)

    # intercept
    X[:, 0] = 1.0

    # lag blocks (most recent first)
    col = 1
    for ell in range(1, lags + 1):
        X[:, col : col + C] = Y[lags - ell : T - ell, :]
        col += C

    # exogenous
    if X_exog is not None:
        X[:, col : col + p_exog] = X_exog[lags:, :]

    return X, Yt


def fit_varx_ridge(
    Y: np.ndarray,
    X_exog: Optional[np.ndarray] = None,
    lags: int = 1,
    ridge: float = 1e-4,
) -> VARXFit:
    """Fit ridge VARX by closed-form multi-output regression.

    Parameters
    ----------
    Y:
        (T, C) array.
    X_exog:
        (T, p) optional exogenous features.
    lags:
        1 or 2 (or more, but the repo focuses on 1–2).
    ridge:
        L2 penalty strength added to X'X (excluding intercept).
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be 2D (T, C)")
    T, C = Y.shape

    X, Yt = _build_design(Y, X_exog, lags)
    p = X.shape[1]

    XtX = X.T @ X
    reg = np.eye(p)
    reg[0, 0] = 0.0  # do not penalize intercept
    XtX_reg = XtX + ridge * reg
    XtY = X.T @ Yt

    W = np.linalg.solve(XtX_reg, XtY)  # (p, C)

    intercept = W[0, :]
    A_list: List[np.ndarray] = []
    idx = 1
    for _ in range(lags):
        A_list.append(W[idx : idx + C, :])
        idx += C

    B = None
    if X_exog is not None:
        B = W[idx:, :]

    return VARXFit(lags=lags, A_list=A_list, B=B, intercept=intercept, ridge=ridge)


def score_r2(Y_true: np.ndarray, Y_pred: np.ndarray, lags: int) -> float:
    """R^2 on rows t>=lags (ignores NaNs in Y_pred for first lags)."""
    Y_true = np.asarray(Y_true, float)
    Y_pred = np.asarray(Y_pred, float)
    yt = Y_true[lags:]
    yp = Y_pred[lags:]
    mse = np.mean((yt - yp) ** 2)
    var = np.var(yt)
    return float(1.0 - mse / (var + 1e-12))
