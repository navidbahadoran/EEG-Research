"""eeg_data.py

Thin abstraction for EEG windowed features.

We intentionally do NOT assume raw EEG formats (EDF/BDF/FIF) here.
Instead, we recommend a preprocessing step that outputs windowed features:

  - y_{it} in R^G : e.g., log-bandpower per channel per window

Recommended storage for each (subject, task/session) file:
  `.npz` with keys:
    - 'Y' : array (T, G)  windowed EEG features
    - 'sfreq' (optional)
    - 'times' (optional)
    - any other metadata

Once you have raw EEG, preprocess into this format so the panel VAR code
stays lightweight and reproducible.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class EegWindows:
    Y: np.ndarray  # (T, G)
    meta: Dict[str, Any]

def load_eeg_windows_npz(path: str | Path) -> EegWindows:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"EEG windows npz not found: {p}")
    d = np.load(p, allow_pickle=True)
    if "Y" not in d.files:
        raise ValueError(f"Expected key 'Y' in {p}. Found keys: {d.files}")
    meta = {k: d[k] for k in d.files if k != "Y"}
    return EegWindows(Y=np.asarray(d["Y"], float), meta=meta)
