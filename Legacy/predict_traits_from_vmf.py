"""predict_traits_from_vmf.py

Supervised prediction for outcomes like `p_factor` or `attention` using a
summary CSV that includes vMF parameters per subject/task.

Expected columns (your provided CSV has these):
  - subject_ID, task, age, sex, ...
  - p_factor, attention, internalizing, externalizing (targets)
  - kappa: Python list string of length K
  - logalpha: Python list string of length K

This script:
  - parses list columns
  - builds simple aggregate features (mean/std/max, entropy, ...)
  - does group-aware CV by subject_ID to avoid leakage

Example:
  python predict_traits_from_vmf.py --csv vmf_fixedmus_summary_K7.csv --target attention
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def parse_list_col(s: str) -> np.ndarray:
    if isinstance(s, list):
        return np.asarray(s, dtype=float)
    return np.asarray(ast.literal_eval(s), dtype=float)


def row_features(kappa: np.ndarray, logalpha: np.ndarray) -> np.ndarray:
    probs = _softmax(logalpha)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    top1 = float(np.max(probs))
    gap = float(np.sort(probs)[-1] - np.sort(probs)[-2]) if probs.size >= 2 else 0.0
    feats = np.array([
        float(kappa.mean()), float(kappa.std()), float(kappa.max()), float(kappa.min()),
        float(np.median(kappa)), float(entropy), float(top1), float(gap),
    ], dtype=float)
    return feats


def build_design(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    kappa_list = df["kappa"].apply(parse_list_col)
    loga_list = df["logalpha"].apply(parse_list_col)

    X = np.vstack([row_features(k, a) for k, a in zip(kappa_list, loga_list)])
    names = [
        "kappa_mean", "kappa_std", "kappa_max", "kappa_min", "kappa_median",
        "alpha_entropy", "alpha_top1", "alpha_top1_gap",
    ]

    # Add a couple of plain covariates if available
    extra_cols = []
    for col in ["duration_sec", "n_times", "sfreq", "age"]:
        if col in df.columns:
            extra_cols.append(col)

    if extra_cols:
        extra = df[extra_cols].astype(float).to_numpy()
        X = np.column_stack([X, extra])
        names += extra_cols

    # Sex as indicator if available
    if "sex" in df.columns:
        sex = (df["sex"].astype(str).str.upper() == "M").astype(float).to_numpy()[:, None]
        X = np.column_stack([X, sex])
        names += ["sex_male1"]

    return X, names


def group_kfold_indices(groups: np.ndarray, n_splits: int = 5, seed: int = 0):
    """Simple group K-fold without sklearn dependency."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_splits)
    for k in range(n_splits):
        test_g = set(folds[k])
        test_idx = np.array([g in test_g for g in groups], dtype=bool)
        train_idx = ~test_idx
        yield np.where(train_idx)[0], np.where(test_idx)[0]


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    # add intercept
    X1 = np.column_stack([np.ones((X.shape[0], 1)), X])
    p = X1.shape[1]
    XtX = X1.T @ X1
    reg = np.eye(p)
    reg[0, 0] = 0.0
    w = np.linalg.solve(XtX + lam * reg, X1.T @ y)
    return w


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones((X.shape[0], 1)), X])
    return X1 @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True, help="e.g., p_factor, attention")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--lambda", dest="lam", type=float, default=10.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not found in CSV columns.")
    df = df.dropna(subset=[args.target, "kappa", "logalpha", "subject_ID"]).copy()

    X, names = build_design(df)
    y = df[args.target].astype(float).to_numpy()
    groups = df["subject_ID"].astype(str).to_numpy()

    r2s = []
    mses = []
    for tr, te in group_kfold_indices(groups, n_splits=args.splits, seed=0):
        w = ridge_fit(X[tr], y[tr], lam=args.lam)
        pred = ridge_predict(X[te], w)
        mse = float(np.mean((y[te] - pred) ** 2))
        var = float(np.var(y[te]))
        r2 = float(1.0 - mse / (var + 1e-12))
        r2s.append(r2)
        mses.append(mse)

    print(f"Rows used: {len(df)}")
    print(f"Features: {len(names)} -> {', '.join(names)}")
    print(f"Group-CV (by subject_ID): splits={args.splits}")
    print(f"R2:  mean={np.mean(r2s):.3f}, std={np.std(r2s):.3f}")
    print(f"MSE: mean={np.mean(mses):.4g}, std={np.std(mses):.4g}")


if __name__ == "__main__":
    main()
