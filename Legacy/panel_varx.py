"""panel_varx.py

Pooled ridge VARX for panel data (multiple subjects).

Baseline model (no latent factors yet):
  y_{it} = sum_{l=1..L} A_{l,i} y_{i,t-l} + B_i x_{it} + u_{it}

For a simple, teachable starting point we fit either:
  - pooled coefficients across subjects, or
  - subject-specific coefficients with shared regularization.

This module provides the pooled version.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class PanelVarxResult:
    L: int
    A: np.ndarray  # (L, G, G)
    B: np.ndarray  # (K, G)
    c: np.ndarray  # (G,)
    ridge: float

def _build_design(Y: np.ndarray, X: Optional[np.ndarray], L: int) -> Tuple[np.ndarray, np.ndarray]:
    """Y: (T,G), X: (T,K) -> returns (Phi, Y_target)
       Phi has columns: [1, vec(Y_{t-1}),...,vec(Y_{t-L}), X_t]
    """
    T, G = Y.shape
    if T <= L:
        raise ValueError("T must be > L")
    Yt = Y[L:]  # (T-L, G)
    parts = [np.ones((T-L, 1))]
    for l in range(1, L+1):
        parts.append(Y[L-l:T-l])  # (T-L, G)
    if X is not None:
        parts.append(X[L:])       # (T-L, K)
    Phi = np.concatenate(parts, axis=1)  # (T-L, 1+L*G+K)
    return Phi, Yt

def fit_pooled_varx_ridge(Y_list: list[np.ndarray], X_list: Optional[list[np.ndarray]], L: int = 1, ridge: float = 1.0) -> PanelVarxResult:
    """Fit pooled VARX across subjects by stacking time."""
    if X_list is None:
        X_list = [None]*len(Y_list)
    Phis=[]
    Ys=[]
    for Y, X in zip(Y_list, X_list):
        Phi, Yt = _build_design(np.asarray(Y,float), None if X is None else np.asarray(X,float), L)
        Phis.append(Phi); Ys.append(Yt)
    Phi_all = np.vstack(Phis)
    Y_all = np.vstack(Ys)  # (n_obs, G)

    # Ridge on multivariate regression: W = (Phi'Phi+lam I)^-1 Phi'Y
    XtX = Phi_all.T @ Phi_all
    lamI = ridge * np.eye(XtX.shape[0])
    W = np.linalg.solve(XtX + lamI, Phi_all.T @ Y_all)  # (p, G)

    # unpack
    idx=0
    c = W[idx]; idx += 1  # (G,)
    A_mats=[]
    idx=1
    for l in range(L):
        block = W[idx:idx+Y_list[0].shape[1], :]  # (G, G)
        # prediction: Y_{t-l} (1xG) @ block -> (1xG)
        A_mats.append(block.T)  # store as (G,G)
        idx += Y_list[0].shape[1]
    A = np.stack(A_mats, axis=0)  # (L,G,G)
    B = W[idx:, :]  # (K,G) or (0,G)
    return PanelVarxResult(L=L, A=A, B=B, c=c, ridge=ridge)