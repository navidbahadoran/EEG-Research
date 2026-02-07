"""vmf_dataset.py

Build a clean dataset from:
  - the summary CSV (metadata + traits + npz pointer)
  - local `.npz` files (vMF time series)

Output:
  - a pandas DataFrame with one row per (subject, task) and extracted features
  - optional aggregation to one row per subject.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd

from vmf_npz import load_vmf_npz, vmf_dynamic_features
from config import Paths

@dataclass
class VmfRow:
    subject_ID: str
    task: str
    npz_path: Path

def _local_npz_path_from_csv(probabilities_file: str, vmf_npz_dir: Path) -> Path:
    # Use filename only (robust across machines)
    name = Path(probabilities_file).name
    return vmf_npz_dir / name

def build_vmf_feature_table(csv_path: str | Path, paths: Optional[Paths] = None, K: Optional[int] = None) -> pd.DataFrame:
    paths = paths or Paths.default()
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    rows = []
    missing = 0

    for _, r in df.iterrows():
        local = _local_npz_path_from_csv(str(r["probabilities_file"]), paths.vmf_npz_dir)
        if not local.exists():
            missing += 1
            continue
        series = load_vmf_npz(local, K=K)
        feats = vmf_dynamic_features(series.Z)
        out = {**r.to_dict(), **feats}
        out["local_npz"] = str(local)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    if missing > 0:
        out_df.attrs["missing_npz"] = missing
    return out_df

def aggregate_subject_level(df_task: pd.DataFrame, outcomes=("p_factor","attention","internalizing","externalizing")) -> pd.DataFrame:
    """Aggregate (subject, task) rows to subject-level by averaging feature columns across tasks."""
    # Keep first outcome values per subject (usually identical across tasks); average features across tasks
    feature_cols = [c for c in df_task.columns if c.startswith("occ_") or c in ("switch_rate","volatility","entropy_mean","entropy_std")]
    meta_cols = ["subject_ID","age","sex","handedness"]
    agg = df_task.groupby("subject_ID").agg({**{c:"mean" for c in feature_cols}, **{c:"first" for c in outcomes if c in df_task.columns}, **{c:"first" for c in meta_cols if c in df_task.columns}})
    agg = agg.reset_index()
    return agg
