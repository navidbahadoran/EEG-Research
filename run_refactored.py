# run_refactored.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from config import SUBJECT_META, SESSION_NAMES                   # labels, male, age  :contentReference[oaicite:21]{index=21}
from dataset import load_12_subject_array, concat_sessions_timewise_from_npy  # loaders  :contentReference[oaicite:22]{index=22}
from model import EEGPanelIFEMI

def main():
    arr = load_12_subject_array("clean_EC.npy")  # shape (12, 4, 32, 90000)  :contentReference[oaicite:23]{index=23}
    assert arr.shape[0] == len(SUBJECT_META), "Subject meta must match npy subjects."

    model = EEGPanelIFEMI(K_vmf=4, r_grid=[1,2,3], random_state=42)
    rows = []
    for s in range(arr.shape[0]):
        label, male1, age = SUBJECT_META[s]
        X, sessions = concat_sessions_timewise_from_npy(arr[s], SESSION_NAMES)  # (T,C), session tags  :contentReference[oaicite:24]{index=24}
        rep = model.fit(
            X=X,
            sessions=sessions,
            sex_male1=float(male1),       # or None to test EM sex-imputation
            age_years=float(age),         # or None to test EM age-imputation
            task_rest1=1.0,
            em_iters=2
        )
        rows.append({
            "subject": label,
            "rank": rep.best_r,
            "train_mse": rep.train_metrics["mse"],
            "train_r2": rep.train_metrics["r2"],
            "test_mse": rep.test_metrics["mse"],
            "test_r2": rep.test_metrics["r2"],
        })
        print(f"[Done] {label}: r={rep.best_r}, train={rep.train_metrics}, test={rep.test_metrics}")

    df = pd.DataFrame(rows).sort_values("subject")
    df.to_csv("summary_refactored.csv", index=False)
    print("\n=== Summary (refactored) ===")
    print(df)

if __name__ == "__main__":
    main()
