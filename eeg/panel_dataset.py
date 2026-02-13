# panel_dataset.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, RAW_EEG_DIR, VMF_DIR, TIME_STRIDE, K_VMF
from .raw_eeg_npy import load_raw_eeg_npy
from vmf_npz import load_vmf_npz

def load_task_panel(task: str):
    idx_path = OUTPUT_DIR / "data_index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing {idx_path}. Run: python data_index.py")

    idx = pd.read_csv(idx_path)
    sub = idx[(idx["task"] == task) & (idx["match_status"] == "ok")].copy()
    if sub.empty:
        raise ValueError(f"No matched rows for task={task}. Check outputs/data_index.csv")

    Y_list, Z_list, subjects = [], [], []
    lengths = []

    for _, r in sub.iterrows():
        raw_path = Path(RAW_EEG_DIR) / r["raw_file"]
        vmf_path = Path(VMF_DIR) / r["vmf_file"]

        raw = load_raw_eeg_npy(raw_path, sfreq_fallback=1.0)
        Y = raw.data          # (T,G)
        vmf = load_vmf_npz(vmf_path, K=K_VMF)
        Z = vmf.P # (T,K)

        T = min(Y.shape[0], Z.shape[0])
        Y = Y[:T:TIME_STRIDE]
        Z = Z[:T:TIME_STRIDE]

        Y_list.append(Y)
        Z_list.append(Z)
        subjects.append(r["subject"])
        lengths.append(Y.shape[0])

    # Trim to common T across subjects for this task
    T_min = min(lengths)
    Y_list = [Y[:T_min] for Y in Y_list]
    Z_list = [Z[:T_min] for Z in Z_list]

    return Y_list, Z_list, subjects, T_min

