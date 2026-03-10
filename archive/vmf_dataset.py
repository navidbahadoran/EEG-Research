# =========================
# vmf_dataset.py
# =========================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DATA_DIR, CSV_PATH, VMF_K
from vmf_npz import load_vmf_npz, VmfRecord
from vmf_features import extract_vmf_features_from_P


TARGET_COLS_DEFAULT = ["p_factor", "attention", "internalizing", "externalizing"]


def _filename_from_csv_path(p: str) -> str:
    """CSV stores absolute paths; we only need the filename."""
    try:
        return Path(p).name
    except Exception:
        # fallback: split by slash/backslash
        return str(p).replace("\\", "/").split("/")[-1]


def build_vmf_feature_table(
    csv_path: Path = CSV_PATH,
    vmf_dir: Path = DATA_DIR,
    K: int = VMF_K,
    targets: Optional[List[str]] = None,
    *,
    stride: int = 1,
    max_T: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns a table with one row per (subject, task) with extracted features + targets.

    Parameters
    ----------
    stride : int
        Subsample time axis for speed. (e.g., stride=10 uses every 10th time point)
    max_T : Optional[int]
        Truncate long series (optional).
    """
    if targets is None:
        targets = TARGET_COLS_DEFAULT

    df = pd.read_csv(csv_path)

    required = ["probabilities_file"] + [c for c in targets if c in df.columns]
    if "probabilities_file" not in df.columns:
        raise KeyError("CSV must contain column 'probabilities_file'.")

    rows = []
    missing_files = 0

    for _, r in df.iterrows():
        fname = _filename_from_csv_path(r["probabilities_file"])
        npz_path = vmf_dir / fname

        try:
            rec: VmfRecord = load_vmf_npz(npz_path, K=K)
        except FileNotFoundError:
            missing_files += 1
            continue

        feats = extract_vmf_features_from_P(rec.P, stride=stride, max_T=max_T)

        row = {
            "subject": rec.subject,
            "task": rec.task,
            "filename": rec.filename,
        }
        row.update(feats)

        # Attach targets (many are subject-level but repeated per task; that's ok)
        for c in targets:
            if c in df.columns:
                row[c] = r[c]

        rows.append(row)

    out = pd.DataFrame(rows)
    if missing_files > 0:
        print(f"[WARN] Missing {missing_files} .npz files (not found in {vmf_dir})")

    if out.empty:
        raise RuntimeError("No vMF records loaded. Check DATA_DIR and CSV_PATH.")

    return out


def aggregate_to_subject_level(
    task_df: pd.DataFrame,
    *,
    targets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate (subject, task) rows into one row per subject.
    Features are averaged across tasks.
    Targets are taken as the first non-missing value within subject.
    """
    if targets is None:
        targets = TARGET_COLS_DEFAULT

    # Identify feature columns: numeric, excluding targets
    exclude = {"subject", "task", "filename"} | set(targets)
    feat_cols = [c for c in task_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(task_df[c])]

    def first_non_missing(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    agg_dict = {c: "mean" for c in feat_cols}
    for t in targets:
        if t in task_df.columns:
            agg_dict[t] = first_non_missing

    subj = (
        task_df
        .groupby("subject", as_index=False)
        .agg(agg_dict)
    )

    return subj
