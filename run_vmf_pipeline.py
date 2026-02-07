# =========================
# run_vmf_pipeline.py
# =========================
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from config import CSV_PATH, DATA_DIR, OUTPUT_DIR, VMF_K, RANDOM_SEED, N_FOLDS
from vmf_dataset import build_vmf_feature_table, aggregate_to_subject_level


# ----------- Small ML utilities (no sklearn) -----------

def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardize using train stats; return standardized + mean/std."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    """
    Closed-form ridge regression with intercept.
    Minimizes ||y - (b0 + Xb)||^2 + alpha ||b||^2
    """
    # Add intercept
    Xtr = np.column_stack([np.ones(X_train.shape[0]), X_train])
    Xte = np.column_stack([np.ones(X_test.shape[0]), X_test])

    # Do not penalize intercept
    p = Xtr.shape[1]
    I = np.eye(p)
    I[0, 0] = 0.0

    A = Xtr.T @ Xtr + alpha * I
    b = Xtr.T @ y_train
    coef = np.linalg.solve(A, b)
    return Xte @ coef


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def grouped_kfold(subjects: np.ndarray, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Simple grouped CV by subject. Splits unique subjects into folds,
    returns list of (train_idx, test_idx) indices into the full row array.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects)
    rng.shuffle(uniq)

    folds = np.array_split(uniq, n_splits)
    splits = []
    for k in range(n_splits):
        test_subj = set(folds[k].tolist())
        test_mask = np.array([s in test_subj for s in subjects], dtype=bool)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        splits.append((train_idx, test_idx))
    return splits


def run_prediction(
    df: pd.DataFrame,
    target: str,
    *,
    alpha: float = 10.0,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """
    Subject-wise CV for one target using ridge regression.
    """
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in dataframe columns.")

    # Build feature matrix
    exclude = {"subject", target}
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    # Drop rows missing target
    sub = df.dropna(subset=[target]).copy()
    y = sub[target].to_numpy(dtype=float)
    subjects = sub["subject"].to_numpy()

    X = sub[feat_cols].to_numpy(dtype=float)

    # Subject-wise splits
    splits = grouped_kfold(subjects, n_splits=n_folds, seed=seed)

    y_pred_all = np.zeros_like(y)
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        ytr = y[tr]

        # Standardize
        Xtr_s, Xte_s, _, _ = standardize_train_test(Xtr, Xte)

        # Fit/predict
        yhat = ridge_fit_predict(Xtr_s, ytr, Xte_s, alpha=alpha)
        y_pred_all[te] = yhat

    return {
        "n": float(len(y)),
        "r2": r2_score(y, y_pred_all),
        "mse": mse(y, y_pred_all),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(CSV_PATH), help="Path to vmf_fixedmus_summary_K7.csv")
    ap.add_argument("--vmf_dir", type=str, default=str(DATA_DIR), help="Directory containing .npz files")
    ap.add_argument("--K", type=int, default=VMF_K, help="Number of vMF components (K=7)")
    ap.add_argument("--stride", type=int, default=10, help="Subsample time series by taking every stride-th row (speed)")
    ap.add_argument("--max_T", type=int, default=0, help="If >0, truncate P to first max_T rows")
    ap.add_argument("--alpha", type=float, default=10.0, help="Ridge penalty")
    args = ap.parse_args()

    max_T = args.max_T if args.max_T > 0 else None

    # 1) Build task-level feature table
    task_df = build_vmf_feature_table(
        csv_path=pd.Path(args.csv) if hasattr(pd, "Path") else args.csv,  # compatibility
        vmf_dir=__import__("pathlib").Path(args.vmf_dir),
        K=args.K,
        stride=args.stride,
        max_T=max_T,
    )

    # 2) Aggregate to subject level (recommended for p_factor/attention)
    subj_df = aggregate_to_subject_level(task_df)

    # 3) Save features
    out_path = OUTPUT_DIR / "vmf_subject_features.csv"
    subj_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote subject-level features to: {out_path}")

    # 4) Predict targets (if present)
    for target in ["p_factor", "attention", "internalizing", "externalizing"]:
        if target in subj_df.columns:
            res = run_prediction(subj_df[["subject"] + [c for c in subj_df.columns if c != "subject"]], target, alpha=args.alpha)
            print(f"\nTarget: {target}")
            print(f"  n   = {int(res['n'])}")
            print(f"  R2  = {res['r2']:.4f}")
            print(f"  MSE = {res['mse']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
