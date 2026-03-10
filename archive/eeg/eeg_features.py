# eeg/eeg_features.py
from __future__ import annotations

from typing import Dict, Tuple, List
import numpy as np


def bandpower_features(window: np.ndarray, sfreq: float, bands: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    Compute bandpower per channel for each frequency band using FFT (no scipy dependency).
    window: shape (win_samples, G)
    Returns: shape (G * n_bands,)
    """
    win, G = window.shape
    x = window - window.mean(axis=0, keepdims=True)

    # FFT
    freqs = np.fft.rfftfreq(win, d=1.0 / sfreq)
    fft = np.fft.rfft(x, axis=0)
    psd = (np.abs(fft) ** 2) / win  # simple periodogram PSD estimate

    feats = []
    for _, (f_lo, f_hi) in bands.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        bp = psd[mask].sum(axis=0)  # sum over freq bins -> per channel
        feats.append(bp)

    feat = np.concatenate(feats, axis=0)  # (n_bands*G,)
    # log-transform for stability
    return np.log1p(feat)


def make_feature_names(ch_names: List[str] | None, bands: Dict[str, Tuple[float, float]], G: int) -> List[str]:
    if ch_names is None:
        ch_names = [f"ch{j:02d}" for j in range(G)]
    names = []
    for band_name in bands.keys():
        for ch in ch_names:
            names.append(f"{band_name}_{ch}")
    return names
