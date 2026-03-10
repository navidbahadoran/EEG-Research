from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from eeg.vmf_baselines import (
    MarkovTransitionBaseline,
    MultinomialDominantStateBaseline,
    PersistenceBaseline,
    PooledVARXBaseline,
    PreparedPanelData,
    save_baseline_result,
)
from scripts.model_common import load_panel_data, standardize_from_train


def prepare_panel_data() -> PreparedPanelData:
    data = load_panel_data()

    Y_list = data["Y_list"]
    X_list = data["X_list"]
    units = data["units"]
    tasks = data["tasks"]
    feature_names = data["feature_names"]
    T = int(data["T"])
    task_levels = data.get("task_levels", [])
    baseline_task = data.get("baseline_task", None)

    train_end = max(4, min(T - 1, int(np.floor(float(config.TRAIN_FRAC) * T))))
    X_list, _, _ = standardize_from_train(X_list, train_end)

    return PreparedPanelData(
        units=units,
        tasks=tasks,
        Y_list=Y_list,
        X_list=X_list,
        feature_names=feature_names,
        train_end=train_end,
        common_T=T,
        baseline_task=baseline_task,
        task_levels=task_levels,
    )


def run_vmf_baselines() -> tuple[str, str]:
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_panel_data()
    K = data.Y_list[0].shape[1]
    p = data.X_list[0].shape[1]

    metadata = {
        "N_units": len(data.units),
        "common_T": data.common_T,
        "K": K,
        "p": p,
        "train_frac": float(config.TRAIN_FRAC),
        "train_end": int(data.train_end),
        "baseline_task": data.baseline_task,
        "task_levels": "|".join(data.task_levels),
    }

    baselines = [
        PersistenceBaseline(),
        PooledVARXBaseline(ridge=1.0),
        MarkovTransitionBaseline(),
        MultinomialDominantStateBaseline(C=1.0, max_iter=1000),
    ]

    metrics_frames: list[pd.DataFrame] = []
    prediction_files: list[str] = []

    for baseline in baselines:
        print(f"[BASELINE] running {baseline.mode}")
        result = baseline.fit_predict(data)
        metrics_path, preds_path = save_baseline_result(result, out_dir, metadata)
        metrics_frames.append(pd.read_csv(metrics_path))
        prediction_files.append(str(preds_path))
        print(f"[BASELINE] wrote {metrics_path.name} and {preds_path.name}")

    combined = pd.concat(metrics_frames, ignore_index=True)

    factor_metrics_path = out_dir / "vmf_pvar_pooled_metrics.csv"
    if factor_metrics_path.exists():
        factor_metrics = pd.read_csv(factor_metrics_path)
        keep_cols = [c for c in combined.columns if c in factor_metrics.columns]
        extra = [c for c in factor_metrics.columns if c not in keep_cols]
        combined = pd.concat([factor_metrics[keep_cols + extra], combined], ignore_index=True, sort=False)

    comparison_path = out_dir / "vmf_baseline_comparison.csv"
    combined.to_csv(comparison_path, index=False)
    print(f"[BASELINE] comparison table -> {comparison_path}")

    manifest_path = out_dir / "vmf_baseline_manifest.csv"
    pd.DataFrame({"prediction_file": prediction_files}).to_csv(manifest_path, index=False)

    return str(comparison_path), str(manifest_path)


if __name__ == "__main__":
    run_vmf_baselines()