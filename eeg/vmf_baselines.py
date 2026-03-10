from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-12


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = float(np.sum((y_true - y_pred) ** 2))
    y_bar = np.mean(y_true, axis=0, keepdims=True)
    sst = float(np.sum((y_true - y_bar) ** 2))
    if sst <= EPS:
        return np.nan
    return 1.0 - sse / sst


def project_rows_to_simplex(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    M = np.clip(M, EPS, None)
    row_sums = np.sum(M, axis=1, keepdims=True)
    row_sums = np.where(row_sums <= EPS, 1.0, row_sums)
    return M / row_sums


def score_probability_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = project_rows_to_simplex(np.asarray(y_pred, dtype=float))

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    r2 = float(safe_r2(y_true, y_pred))

    true_state = np.argmax(y_true, axis=1)
    pred_state = np.argmax(y_pred, axis=1)
    accuracy = float(np.mean(true_state == pred_state))

    p = project_rows_to_simplex(np.clip(y_true, EPS, 1.0))
    q = project_rows_to_simplex(np.clip(y_pred, EPS, 1.0))
    kl = float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))
    cross_entropy = float(-np.mean(np.sum(p * np.log(q), axis=1)))

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "accuracy": accuracy,
        "kl": kl,
        "cross_entropy": cross_entropy,
    }


@dataclass
class PreparedPanelData:
    units: list[str]
    tasks: list[str]
    Y_list: list[np.ndarray]
    X_list: list[np.ndarray]
    feature_names: list[str]
    train_end: int
    common_T: int
    baseline_task: str
    task_levels: list[str]


@dataclass
class BaselineResult:
    mode: str
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]
    extras: dict[str, Any]


def build_supervised_arrays(
    Y_list: list[np.ndarray],
    X_list: list[np.ndarray],
    train_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    X_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    X_test_parts: list[np.ndarray] = []
    y_test_parts: list[np.ndarray] = []
    test_index: list[tuple[int, int]] = []

    for i, (Y, X) in enumerate(zip(Y_list, X_list)):
        feats = np.hstack([Y[:-1], X[:-1]])
        targets = Y[1:]

        X_train_parts.append(feats[: train_end - 1])
        y_train_parts.append(targets[: train_end - 1])
        X_test_parts.append(feats[train_end - 1 :])
        y_test_parts.append(targets[train_end - 1 :])

        for t in range(train_end - 1, Y.shape[0] - 1):
            test_index.append((i, t))

    return (
        np.vstack(X_train_parts),
        np.vstack(y_train_parts),
        np.vstack(X_test_parts),
        np.vstack(y_test_parts),
        test_index,
    )


class PersistenceBaseline:
    mode = "baseline_persistence"

    def fit_predict(self, data: PreparedPanelData) -> BaselineResult:
        y_true_parts: list[np.ndarray] = []
        y_pred_parts: list[np.ndarray] = []

        for Y in data.Y_list:
            y_true_parts.append(Y[data.train_end:])
            y_pred_parts.append(Y[data.train_end - 1 : -1])

        y_true = np.vstack(y_true_parts)
        y_pred = np.vstack(y_pred_parts)
        metrics = score_probability_forecasts(y_true, y_pred)
        return BaselineResult(self.mode, y_true, y_pred, metrics, extras={})


class PooledVARXBaseline:
    mode = "baseline_pooled_varx_ridge"

    def __init__(self, ridge: float = 1.0):
        self.ridge = float(ridge)
        self.coef_: np.ndarray | None = None

    def fit_predict(self, data: PreparedPanelData) -> BaselineResult:
        X_train, y_train, X_test, y_test, _ = build_supervised_arrays(
            data.Y_list, data.X_list, data.train_end
        )
        X_train_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        X_test_aug = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

        d = X_train_aug.shape[1]
        penalty = self.ridge * np.eye(d)
        penalty[0, 0] = 0.0  # do not penalize intercept

        self.coef_ = np.linalg.solve(X_train_aug.T @ X_train_aug + penalty, X_train_aug.T @ y_train)
        y_pred = X_test_aug @ self.coef_
        y_pred = project_rows_to_simplex(y_pred)
        metrics = score_probability_forecasts(y_test, y_pred)
        extras = {"ridge": self.ridge, "coef_shape": tuple(self.coef_.shape)}
        return BaselineResult(self.mode, y_test, y_pred, metrics, extras=extras)


class MarkovTransitionBaseline:
    mode = "baseline_markov_transition"

    def fit_predict(self, data: PreparedPanelData) -> BaselineResult:
        K = data.Y_list[0].shape[1]
        counts = np.zeros((K, K), dtype=float)

        for Y in data.Y_list:
            states = np.argmax(Y[: data.train_end], axis=1)
            for t in range(len(states) - 1):
                counts[states[t], states[t + 1]] += 1.0

        row_sums = counts.sum(axis=1, keepdims=True)
        P = np.divide(counts, row_sums, out=np.full_like(counts, 1.0 / K), where=row_sums > 0)

        y_true_parts: list[np.ndarray] = []
        y_pred_parts: list[np.ndarray] = []
        for Y in data.Y_list:
            y_true_parts.append(Y[data.train_end:])
            y_pred_parts.append(project_rows_to_simplex(Y[data.train_end - 1 : -1] @ P))

        y_true = np.vstack(y_true_parts)
        y_pred = np.vstack(y_pred_parts)
        metrics = score_probability_forecasts(y_true, y_pred)
        extras = {"transition_matrix": P}
        return BaselineResult(self.mode, y_true, y_pred, metrics, extras=extras)


class MultinomialDominantStateBaseline:
    mode = "baseline_multinomial_logit"

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.model_: LogisticRegression | None = None

    def fit_predict(self, data: PreparedPanelData) -> BaselineResult:
        K = data.Y_list[0].shape[1]
        X_train, y_train_prob, X_test, y_test_prob, _ = build_supervised_arrays(
            data.Y_list, data.X_list, data.train_end
        )
        y_train_cls = np.argmax(y_train_prob, axis=1)

        self.model_ = LogisticRegression(
            solver="lbfgs",
            C=self.C,
            max_iter=self.max_iter,
            n_jobs=None,
            random_state=123,
        )
        self.model_.fit(X_train, y_train_cls)
        prob_small = self.model_.predict_proba(X_test)

        y_pred = np.full((X_test.shape[0], K), EPS, dtype=float)
        y_pred[:, self.model_.classes_] = prob_small
        y_pred = project_rows_to_simplex(y_pred)
        metrics = score_probability_forecasts(y_test_prob, y_pred)
        extras = {
            "classes": np.asarray(self.model_.classes_, dtype=int),
            "coef_shape": tuple(self.model_.coef_.shape),
            "intercept_shape": tuple(self.model_.intercept_.shape),
        }
        return BaselineResult(self.mode, y_test_prob, y_pred, metrics, extras=extras)


def save_baseline_result(
    result: BaselineResult,
    out_dir: Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{result.mode}_metrics.csv"
    preds_path = out_dir / f"{result.mode}_predictions.npz"

    row = {"mode": result.mode, **metadata, **result.metrics}
    pd.DataFrame([row]).to_csv(metrics_path, index=False)

    save_dict: dict[str, Any] = {
        "y_true_oof": result.y_true,
        "y_pred_oof": result.y_pred,
    }
    for k, v in result.extras.items():
        save_dict[k] = v
    np.savez_compressed(preds_path, **save_dict)
    return metrics_path, preds_path
