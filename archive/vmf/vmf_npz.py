# =========================
# vmf_npz.py
# =========================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np


@dataclass
class VmfRecord:
    """Container for one (subject, task) vMF time series + metadata."""
    P: np.ndarray              # shape (T, K)
    subject: str
    task: str
    sfreq: Optional[float]
    ch_names: Optional[np.ndarray]
    mus_fixed: Optional[np.ndarray]  # shape (K, 32) typically
    filename: str


def _to_py_scalar(x: Any) -> Any:
    """Safely convert numpy scalar/0-d array to python scalar."""
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    if hasattr(x, "item") and not isinstance(x, (list, tuple, dict, np.ndarray)):
        # numpy scalar
        try:
            return x.item()
        except Exception:
            return x
    return x


def load_vmf_npz(npz_path: Path, K: int) -> VmfRecord:
    """
    Load a vMF npz file that contains keys:
      - P: (T, K) posterior probabilities
      - kappa: (K,)
      - logalpha: (K,)
      - mus_fixed: (K, 32)
      - subject: scalar
      - task: scalar
      - sfreq: scalar
      - ch_names: (32,)
      - template_path: scalar
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"vMF .npz not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)

    if "P" not in d.files:
        raise KeyError(f"Expected key 'P' in {npz_path.name}, found: {d.files}")

    P = d["P"]
    if P.ndim != 2:
        raise ValueError(f"'P' must be 2D (T,K). Got shape {P.shape} in {npz_path.name}")

    # Ensure shape is (T, K)
    if P.shape[1] != K and P.shape[0] == K:
        P = P.T

    if P.shape[1] != K:
        raise ValueError(
            f"Expected P to have K={K} columns. Got shape {P.shape} in {npz_path.name}"
        )

    # Basic sanity checks (non-fatal but helpful)
    if np.nanmin(P) < -1e-8 or np.nanmax(P) > 1 + 1e-8:
        raise ValueError(f"P has values outside [0,1] in {npz_path.name}")

    # Sometimes probabilities don't sum exactly to 1 due to float error; allow tolerance
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3, rtol=0):
        # Not fatal, but warn via print
        print(f"[WARN] P rows not summing to 1 (tolerance 1e-3) in {npz_path.name}")

    subject = str(_to_py_scalar(d["subject"])) if "subject" in d.files else "NA"
    task = str(_to_py_scalar(d["task"])) if "task" in d.files else "NA"

    sfreq = float(_to_py_scalar(d["sfreq"])) if "sfreq" in d.files else None
    ch_names = d["ch_names"] if "ch_names" in d.files else None
    mus_fixed = d["mus_fixed"] if "mus_fixed" in d.files else None

    return VmfRecord(
        P=P.astype(float),
        subject=subject,
        task=task,
        sfreq=sfreq,
        ch_names=ch_names,
        mus_fixed=mus_fixed,
        filename=npz_path.name,
    )
