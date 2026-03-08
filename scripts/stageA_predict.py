# stageA_predict.py
# Stage A ONLY: Train/evaluate subject-level prediction from vmf_subject_features.csv
# Safe for future stages: consumes feature table, produces predictions + metrics.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PredictConfig:
    features_csv: Path = Path("outputs") / "vmf_subject_features.csv"
    out_pred_csv: Path = Path("outputs") / "stageA_oof_predictions.csv"
    out_metrics_csv: Path = Path("outputs") / "stageA_metrics.csv"
    out_info_json: Path = Path("outputs") / "stageA_model_info.json"

    # Targets to try (only those present in the CSV will be used)
    candidate_targets: Tuple[str, ...] = ("attention", "p_factor", "internalizing", "externalizing")

    # Cross-validation
    n_folds: int = 5
    seed: int = 123

    # Ridge strength (can later be tuned; keep fixed for pedagogy)
    ridge_alpha: float = 10.0


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def standardize_train_apply(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float) -> np.ndarray:
    """
    Ridge with intercept, implemented without sklearn.
    """
    # Add intercept
    Xtr = np.column_stack([np.ones(X_train.shape[0]), X_train])
    Xte = np.column_stack([np.ones(X_test.shape[0]), X_test])

    p = Xtr.shape[1]
    I = np.eye(p)
    I[0, 0] = 0.0  # don't penalize intercept

    coef = np.linalg.solve(Xtr.T @ Xtr + alpha * I, Xtr.T @ y_train)
    return Xte @ coef


def make_folds(n: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds


def main(cfg: PredictConfig) -> None:
    if not cfg.features_csv.exists():
        raise FileNotFoundError(
            f"Missing {cfg.features_csv}. Run Stage A feature pipeline first:\n"
            f"  python run_vmf_pipeline.py"
        )

    df = pd.read_csv(cfg.features_csv)
    if "subject" not in df.columns:
        raise ValueError("Expected a 'subject' column in features CSV.")

    # Determine targets present
    targets = [t for t in cfg.candidate_targets if t in df.columns]
    if not targets:
        raise ValueError(f"No targets found. Expected one of: {cfg.candidate_targets}")

    # Feature columns: numeric, excluding subject + targets
    exclude = set(["subject"]) | set(targets)
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found for prediction.")

    out_rows_metrics = []
    out_rows_preds = []

    for target in targets:
        sub = df[["subject", target] + feature_cols].dropna(subset=[target]).copy()
        # if features have NaNs, fill with column means (simple, stable for students)
        sub[feature_cols] = sub[feature_cols].apply(lambda s: s.fillna(s.mean()), axis=0)

        y = sub[target].to_numpy(dtype=float)
        X = sub[feature_cols].to_numpy(dtype=float)
        subjects = sub["subject"].to_numpy()

        n = len(y)
        if n < cfg.n_folds:
            print(f"[WARN] target={target}: n={n} < n_folds={cfg.n_folds}. Skipping.")
            continue

        folds = make_folds(n, cfg.n_folds, cfg.seed)
        oof_pred = np.full(n, np.nan, dtype=float)

        for fold_id, te_idx in enumerate(folds):
            tr_idx = np.setdiff1d(np.arange(n), te_idx)

            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr = y[tr_idx]

            Xtr_z, Xte_z, _, _ = standardize_train_apply(Xtr, Xte)
            pred = ridge_fit_predict(Xtr_z, ytr, Xte_z, alpha=cfg.ridge_alpha)

            oof_pred[te_idx] = pred

        # Metrics on OOF predictions
        mask = ~np.isnan(oof_pred)
        y_true = y[mask]
        y_hat = oof_pred[mask]

        r2 = r2_score(y_true, y_hat)
        out_rows_metrics.append({
            "target": target,
            "n": int(len(y_true)),
            "r2_oof": r2,
            "rmse_oof": rmse(y_true, y_hat),
            "mae_oof": mae(y_true, y_hat),
            "model": "ridge",
            "ridge_alpha": cfg.ridge_alpha,
            "n_folds": cfg.n_folds,
            "seed": cfg.seed,
            "n_features": len(feature_cols),
        })

        # Save per-subject predictions
        for s, yt, yh in zip(subjects, y, oof_pred):
            out_rows_preds.append({
                "subject": s,
                "target": target,
                "y_true": float(yt),
                "y_pred_oof": float(yh) if np.isfinite(yh) else np.nan,
            })

        print(f"[OK] {target}: OOF R2={r2:.3f}  RMSE={rmse(y_true,y_hat):.3f}  MAE={mae(y_true,y_hat):.3f}  n={len(y_true)}")

    # Write outputs
    cfg.out_pred_csv.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(out_rows_metrics).to_csv(cfg.out_metrics_csv, index=False)
    pd.DataFrame(out_rows_preds).to_csv(cfg.out_pred_csv, index=False)

    info = {
        "features_csv": str(cfg.features_csv),
        "feature_cols_count": len(feature_cols),
        "targets_used": targets,
        "n_folds": cfg.n_folds,
        "ridge_alpha": cfg.ridge_alpha,
        "seed": cfg.seed,
    }
    cfg.out_info_json.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"\nSaved:\n- {cfg.out_metrics_csv}\n- {cfg.out_pred_csv}\n- {cfg.out_info_json}")


if __name__ == "__main__":
    main(PredictConfig())
