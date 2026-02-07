"""vmf_npz.py

Utilities to work with vMF **time series** stored in `.npz` files.

The goal is to standardize the representation of Z_{it} so it can be
plugged into the model as part of X_{it} (covariates) or used for
prediction of traits (p_factor, attention, etc.).

We try to be robust to unknown key names inside the .npz by:
  - inspecting arrays in the file
  - preferring a 2D array with shape (T, K) or (K, T)
  - otherwise raising a clear error listing available keys/shapes.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

@dataclass
class VmfSeries:
    """Standardized vMF time series."""
    Z: np.ndarray           # (T, K) posterior probs or features
    meta: Dict[str, Any]    # other arrays/values in npz

def _pick_time_by_k(arr: np.ndarray, K: Optional[int]) -> Optional[np.ndarray]:
    if arr.ndim != 2:
        return None
    T0, T1 = arr.shape
    # if K is known, use it
    if K is not None:
        if T1 == K:
            return arr
        if T0 == K:
            return arr.T
    # otherwise, choose whichever dimension is "small" (<=64) as K
    if T1 <= 64 and T0 > T1:
        return arr
    if T0 <= 64 and T1 > T0:
        return arr.T
    return None

def load_vmf_npz(npz_path: str | Path, K: Optional[int] = None) -> VmfSeries:
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"vMF npz not found: {p}")
    data = np.load(p, allow_pickle=True)
    meta: Dict[str, Any] = {k: data[k] for k in data.files}

    # Try preferred keys first
    preferred = [
        "probabilities", "probs", "posterior", "posteriors", "gamma",
        "responsibilities", "resp", "Z", "z"
    ]
    for k in preferred:
        if k in meta:
            cand = _pick_time_by_k(np.asarray(meta[k]), K)
            if cand is not None:
                return VmfSeries(Z=cand.astype(float), meta=meta)

    # Otherwise search all arrays
    for k, v in meta.items():
        cand = _pick_time_by_k(np.asarray(v), K)
        if cand is not None:
            return VmfSeries(Z=cand.astype(float), meta=meta)

    shapes = {k: np.asarray(meta[k]).shape for k in meta.keys()}
    raise ValueError(
        "Could not infer a (T,K) vMF time series from the npz. " 
        f"Available keys/shapes: {shapes}. "
        "If you know K, pass K=... to load_vmf_npz."
    )

def vmf_dynamic_features(Z: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    """Compute simple, interpretable dynamic features from (T,K) posterior probs.

    These are designed for **trait prediction** (attention, p_factor) and for
    later inclusion as covariates in the EEG dynamic model.
    """
    Z = np.asarray(Z, float)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D (T,K); got {Z.shape}")
    T, K = Z.shape
    # Normalize just in case
    row_sum = Z.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    P = Z / row_sum

    # Occupancy
    occ = P.mean(axis=0)  # (K,)
    # Entropy over time
    Ht = -(P * np.log(P + eps)).sum(axis=1)  # (T,)
    # Switching and volatility
    hard = P.argmax(axis=1)
    switches = float(np.mean(hard[1:] != hard[:-1])) if T > 1 else 0.0
    vol = float(np.mean(np.linalg.norm(P[1:] - P[:-1], axis=1))) if T > 1 else 0.0

    feats: Dict[str, float] = {
        "T": float(T),
        "K": float(K),
        "switch_rate": switches,
        "volatility": vol,
        "entropy_mean": float(Ht.mean()),
        "entropy_std": float(Ht.std(ddof=0)),
    }
    for k in range(K):
        feats[f"occ_{k}"] = float(occ[k])
    return feats

def downsample_Z(Z: np.ndarray, step: int) -> np.ndarray:
    """Downsample Z by taking every `step`-th time point."""
    if step <= 1:
        return Z
    return np.asarray(Z)[::step]
