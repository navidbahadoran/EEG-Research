# eeg/panel_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, RAW_EEG_DIR, VMF_DIR, SFREQ_FALLBACK, WIN_SEC, STEP_SEC, BANDS
from eeg.raw_eeg_npy import load_raw_eeg_npy
from eeg.windowing import sliding_windows
from eeg.eeg_features import bandpower_features


@dataclass
class PanelRecord:
    subject: str
    task: str
    Y: np.ndarray               # (T_windows, d_y)
    Z: Optional[np.ndarray]     # (T_windows, K) or None
    labels: Dict[str, float]    # attention, p_factor, ...
    raw_file: str
    vmf_file: Optional[str]


def _window_features_from_raw(raw_path: Path) -> Tuple[np.ndarray, float]:
    rec = load_raw_eeg_npy(raw_path, sfreq_fallback=SFREQ_FALLBACK)
    x = rec.data
    sfreq = rec.sfreq

    win = int(round(WIN_SEC * sfreq))
    step = int(round(STEP_SEC * sfreq))
    feats = []
    for _, w in sliding_windows(x, win=win, step=step):
        feats.append(bandpower_features(w, sfreq=sfreq, bands=BANDS))
    if not feats:
        raise ValueError(f"Raw EEG too short for windowing: {raw_path.name}")
    return np.vstack(feats), sfreq


def _window_vmf_from_npz(vmf_path: Path, n_windows: int) -> np.ndarray:
    d = np.load(vmf_path, allow_pickle=True)
    P = d["P"]  # shape (T, K)
    if P.ndim != 2:
        raise ValueError(f"Expected P to be 2D, got {P.shape} in {vmf_path.name}")

    # Aligning raw windows with vMF time steps requires knowing how P was produced.
    # For now (baseline), we do a simple resampling to match number of windows:
    T = P.shape[0]
    idx = np.linspace(0, T - 1, num=n_windows).round().astype(int)
    Z = P[idx, :]
    # normalize just in case
    Z = Z / np.clip(Z.sum(axis=1, keepdims=True), 1e-12, None)
    return Z


def load_panel_record(subject: str, task: str, index_csv: Path | None = None) -> PanelRecord:
    """
    Load one (subject, task) record using outputs/data_index.csv.
    Returns aligned window-level features Y and vMF probabilities Z (if available).
    """
    if index_csv is None:
        index_csv = OUTPUT_DIR / "data_index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(f"Missing {index_csv}. Run: python data_index.py")

    idx = pd.read_csv(index_csv)
    row = idx[(idx["subject"] == subject) & (idx["task"] == task)]
    if row.empty:
        raise KeyError(f"No record for subject={subject}, task={task} in {index_csv}")

    r = row.iloc[0]
    raw_file = r["raw_file"]
    vmf_file = r["vmf_file"] if "vmf_file" in r and pd.notna(r["vmf_file"]) else None

    labels = {}
    for c in ["attention", "p_factor", "internalizing", "externalizing"]:
        if c in r and pd.notna(r[c]):
            labels[c] = float(r[c])

    Y, sfreq = _window_features_from_raw(Path(RAW_EEG_DIR) / raw_file)

    Z = None
    if vmf_file is not None and isinstance(vmf_file, str) and len(vmf_file) > 0:
        Z = _window_vmf_from_npz(Path(VMF_DIR) / vmf_file, n_windows=Y.shape[0])

    return PanelRecord(
        subject=subject,
        task=task,
        Y=Y,
        Z=Z,
        labels=labels,
        raw_file=raw_file,
        vmf_file=vmf_file,
    )
