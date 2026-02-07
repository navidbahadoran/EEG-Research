# =========================
# vmf_features.py
# =========================
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, 1.0))


def extract_vmf_features_from_P(
    P: np.ndarray,
    *,
    state_axis: int = 1,
    max_T: Optional[int] = None,
    stride: int = 1,
) -> Dict[str, float]:
    """
    Compute dynamic features from vMF posterior probabilities P (T,K).

    Parameters
    ----------
    P : np.ndarray
        Array shape (T,K).
    max_T : Optional[int]
        If provided, truncate to first max_T rows (useful if extremely long).
    stride : int
        Subsample time by taking every stride-th row.

    Returns
    -------
    features : dict
        Flat dict of numeric features.
    """
    if P.ndim != 2:
        raise ValueError(f"P must be 2D (T,K), got {P.shape}")

    if max_T is not None and P.shape[0] > max_T:
        P = P[:max_T, :]

    if stride > 1:
        P = P[::stride, :]

    T, K = P.shape
    if T < 3:
        raise ValueError(f"Need T>=3 to compute dynamic features; got T={T}")

    # Occupancy (mean prob per state)
    occ = P.mean(axis=0)  # (K,)

    # Entropy time series
    H_t = -np.sum(P * _safe_log(P), axis=1)  # (T,)
    H_mean = float(np.mean(H_t))
    H_std = float(np.std(H_t))

    # Argmax state and switching rate
    s = np.argmax(P, axis=1)  # (T,)
    switches = np.mean(s[1:] != s[:-1])
    switches = float(switches)

    # Volatility: average step-to-step change in probabilities
    dP = P[1:] - P[:-1]  # (T-1, K)
    vol_l1 = float(np.mean(np.sum(np.abs(dP), axis=1)))
    vol_l2 = float(np.mean(np.sqrt(np.sum(dP**2, axis=1))))

    # Max confidence and its dynamics
    maxp_t = np.max(P, axis=1)
    maxp_mean = float(np.mean(maxp_t))
    maxp_std = float(np.std(maxp_t))

    # Optional: simple transition matrix features (argmax-based)
    # Build KxK transition counts
    trans = np.zeros((K, K), dtype=float)
    for a, b in zip(s[:-1], s[1:]):
        trans[a, b] += 1.0
    row_sums = trans.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        trans_prob = np.divide(trans, row_sums, where=row_sums > 0)

    # Transition entropy per state (average)
    trans_ent = []
    for k in range(K):
        pk = trans_prob[k]
        if np.sum(pk) <= 0:
            continue
        ent_k = -np.sum(pk[pk > 0] * np.log(pk[pk > 0]))
        trans_ent.append(ent_k)
    trans_entropy_mean = float(np.mean(trans_ent)) if trans_ent else 0.0

    # Self-transition probability (staying in same argmax state)
    self_trans = float(np.trace(trans) / max(1.0, trans.sum()))

    feats: Dict[str, float] = {
        "T_used": float(T),
        "entropy_mean": H_mean,
        "entropy_std": H_std,
        "switch_rate": switches,
        "vol_l1": vol_l1,
        "vol_l2": vol_l2,
        "maxp_mean": maxp_mean,
        "maxp_std": maxp_std,
        "trans_entropy_mean": trans_entropy_mean,
        "self_transition_rate": self_trans,
    }

    # Add occupancy features
    for k in range(K):
        feats[f"occ_{k}"] = float(occ[k])

    return feats
