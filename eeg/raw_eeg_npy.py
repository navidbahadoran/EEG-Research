# eeg/raw_eeg_npy.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


@dataclass
class RawEEG:
    data: np.ndarray           # shape (T, G) = (n_samples, n_channels)
    sfreq: float               # sampling frequency
    ch_names: Optional[list[str]] = None
    source_path: Optional[Path] = None


def _standardize_shape(x: np.ndarray) -> np.ndarray:
    """
    Return array shaped (T, G) where T is time samples and G channels.
    Supports common formats:
      - (T, G)
      - (G, T)
      - (T,) or (G,) invalid (need 2D)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")

    T, G = x.shape
    # Heuristic: time dimension is usually much larger than channels
    if T >= G:
        return x.astype(float)
    else:
        return x.T.astype(float)


def load_raw_eeg_npy(path: Path, sfreq_fallback: float) -> RawEEG:
    """
    Load .npy raw EEG file. Since .npy has no embedded metadata, we use fallback sfreq unless
    you later add an index/sidecar metadata file.
    """
    arr = np.load(path, allow_pickle=False)
    data = _standardize_shape(arr)

    # Basic sanity checks
    if not np.isfinite(data).all():
        # Replace inf/nan with column means (safe fallback)
        col_mean = np.nanmean(np.where(np.isfinite(data), data, np.nan), axis=0)
        inds = ~np.isfinite(data)
        data[inds] = np.take(col_mean, np.where(inds)[1])

    return RawEEG(data=data, sfreq=float(sfreq_fallback), ch_names=None, source_path=path)
