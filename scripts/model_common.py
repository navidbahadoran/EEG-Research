from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config
from vmf.vmf_panel_builder_pooled import make_vmf_panel_pooled


def project_rows_to_simplex(Y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    Y = np.clip(Y, eps, None)
    Y /= np.sum(Y, axis=1, keepdims=True)
    return Y


def score_probability_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

    true_state = np.argmax(y_true, axis=1)
    pred_state = np.argmax(y_pred, axis=1)
    accuracy = float(np.mean(true_state == pred_state))

    y_true_clip = project_rows_to_simplex(y_true)
    y_pred_clip = project_rows_to_simplex(y_pred)

    kl = float(np.mean(np.sum(y_true_clip * (np.log(y_true_clip) - np.log(y_pred_clip)), axis=1)))
    cross_entropy = float(-np.mean(np.sum(y_true_clip * np.log(y_pred_clip), axis=1)))

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "accuracy": accuracy,
        "kl": kl,
        "cross_entropy": cross_entropy,
    }


def save_metrics_row(path: str | Path, row: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(path, index=False)


def fit_standardizer_from_train(
    X_list: List[np.ndarray],
    train_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit feature standardization using training periods only.
    """
    X_train = np.vstack([np.asarray(X[:train_end], dtype=float) for X in X_list])
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    return x_mean, x_std


def apply_standardizer(
    X_list: List[np.ndarray],
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> List[np.ndarray]:
    """
    Apply a previously fitted standardizer to all periods.
    """
    return [(np.asarray(X, dtype=float) - x_mean) / x_std for X in X_list]


def standardize_from_train(
    X_list: List[np.ndarray],
    train_end: int,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Convenience wrapper: fit on train, apply to full list.
    """
    x_mean, x_std = fit_standardizer_from_train(X_list, train_end)
    X_std = apply_standardizer(X_list, x_mean, x_std)
    return X_std, x_mean, x_std


def load_panel_data() -> Dict:
    """
    Load raw pooled panel data only.
    No standardization is performed here.
    """
    panel = make_vmf_panel_pooled(
        csv_path=str(config.VMF_CSV_PATH),
        vmf_dir=str(config.VMF_DIR),
        id_col="subject_ID",
        task_col="task",
        npz_col="probabilities_file",
        age_col="age",
        sex_col="sex",
        stride=int(config.TIME_STRIDE),
        drop_baseline_task=None,
        summary_window=25,
        targets=("attention", "p_factor"),
    )

    if isinstance(panel, dict):
        Y_list = panel["Y_list"]
        X_list = panel["X_list"]
        units = panel.get("units", np.arange(len(Y_list)))
        tasks = panel.get("tasks", np.array(["unknown"] * len(Y_list), dtype=object))
        feature_names = panel.get(
            "feature_names",
            np.array([f"x_{j}" for j in range(X_list[0].shape[1])], dtype=object),
        )
        y_targets = panel.get("y_targets", None)
        task_levels = panel.get("task_levels", [])
        baseline_task = panel.get("baseline_task", None)
        unit_meta = panel.get("unit_meta", None)
        T_common = panel.get("T", min(Y.shape[0] for Y in Y_list))
    else:
        (
            units,
            tasks,
            Y_list,
            X_list,
            y_targets,
            T_common,
            task_levels,
            baseline_task,
            feature_names,
            unit_meta,
        ) = panel

    Y_list = [np.asarray(Y, dtype=float) for Y in Y_list]
    X_list = [np.asarray(X, dtype=float) for X in X_list]
    units = np.asarray(units, dtype=object)
    tasks = np.asarray(tasks, dtype=object)
    feature_names = np.asarray(feature_names, dtype=object)

    return {
        "Y_list": Y_list,
        "X_list": X_list,
        "units": units,
        "tasks": tasks,
        "feature_names": feature_names,
        "T": int(T_common),
        "y_targets": y_targets,
        "task_levels": task_levels,
        "baseline_task": baseline_task,
        "unit_meta": unit_meta,
    }