from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

META_LIKE = {"time", "timestamp", "fs", "sampling_rate"}

def load_subject_sessions(paths: List[Path]) -> pd.DataFrame:
    """Load multiple CSVs; add a 'session' column from filename stem."""
    dfs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        df = pd.read_csv(p)
        df["session"] = p.stem
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def pick_channel_columns(df: pd.DataFrame, n_channels: int = 32) -> list[str]:
    """Pick the first n numeric columns (excluding typical metadata columns)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    ch_cols = [c for c in num_cols if c.lower() not in META_LIKE][:n_channels]
    if len(ch_cols) < n_channels:
        print(f"[warn] only {len(ch_cols)} numeric channels found; proceeding.")
    return ch_cols
