# dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import re
import numpy as np
import pandas as pd

from io_utils import load_subject_sessions, pick_channel_columns  # CSV helpers  (uses pandas)  # :contentReference[oaicite:5]{index=5}
from config import SESSION_NAMES, CSV_SUBJECT_FILES

def concat_sessions_timewise_from_npy(X_s: np.ndarray, session_names: List[str]) -> tuple[np.ndarray, list[str]]:
    """
    X_s: (Sesh=4, C, L) -> returns:
      X: (T=4*L, C), sessions: list[str] of length T with per-t session name
    """
    Sesh, C, L = X_s.shape
    assert Sesh == len(session_names), f"Expected {len(session_names)} sessions, got {Sesh}"
    blocks = [X_s[j].T for j in range(Sesh)]  # each (L, C)
    X = np.vstack(blocks)                     # (4L, C)
    sessions = sum(([session_names[j]] * L for j in range(Sesh)), [])
    return X, sessions

def load_12_subject_array(npy_path: Path | str) -> np.ndarray:
    """
    Load the 12-subject tensor: expected shape (12, 4, 32, 90000).
    Returns np.ndarray with ndim=4.
    """
    arr = np.load(str(npy_path))
    if arr.ndim != 4 or arr.shape[1] != 4:
        raise ValueError(f"Unexpected npy shape {arr.shape}; expected (S=12,4,C,L).")
    return arr


def _extract_session_tag(stem: str) -> str | None:
    """
    Convert 'subject_0_day1_morning' -> 'day1_morning'.
    Returns None if no match.
    """
    m = re.search(r"(day\d+_(?:morning|afternoon))", stem.lower())
    return m.group(1) if m else None


def load_extra_subject_from_csv(csv_dir: Path, channel_count: int = 32) -> tuple[np.ndarray, list[str]]:
    """
    Load the 4 CSVs for the extra subject and stack in SESSION_NAMES order.
    Matches by normalized 'session_tag' (day1_morning, ...), not full stem.
    """
    paths = [csv_dir / f for f in CSV_SUBJECT_FILES]
    df = load_subject_sessions(paths)  # adds 'session' with full stem  :contentReference[oaicite:2]{index=2}
    # derive a normalized tag column
    df["session_tag"] = df["session"].astype(str).apply(_extract_session_tag)
    if df["session_tag"].isna().any():
        bad = df[df["session_tag"].isna()]["session"].unique().tolist()
        raise ValueError(f"Could not parse session tag from stems: {bad}")

    ch_cols = pick_channel_columns(df, n_channels=channel_count)  # :contentReference[oaicite:3]{index=3}

    blocks, sessions = [], []
    for tag in SESSION_NAMES:
        dfi = df[df["session_tag"] == tag]
        if dfi.empty:
            avail = sorted(df["session_tag"].unique().tolist())
            raise ValueError(
                f"No rows for session '{tag}'. "
                f"Available session_tag values: {avail}. "
                f"Check filenames or SESSION_NAMES."
            )
        Xi = dfi[ch_cols].to_numpy(float)  # (Li, C)
        blocks.append(Xi)
        sessions.extend([tag] * Xi.shape[0])

    X = np.vstack(blocks)  # (T, C)
    if X.shape[0] == 0:
        raise ValueError("After stacking CSV sessions, T==0. Verify CSV content and tags.")
    return X, sessions