# stageB_predict_varx.py
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, RANDOM_SEED


def ridge_var1_oos(Y: np.ndarray, alpha: float = 10.0, train_frac: float = 0.7):
    """
    Simple VAR(1) ridge:
      Y_t = A Y_{t-1} + e_t
    where Y is (T, d).
    Returns predictions for test region and metrics.
    """
    T, d = Y.shape
    if T < 10:
        return None

    split = int(np.floor(train_frac * (T - 1)))  # number of usable transitions for training
    # Build lagged design
    X = Y[:-1, :]     # (T-1, d)
    Y1 = Y[1:, :]     # (T-1, d)

    Xtr, Ytr = X[:split], Y1[:split]
    Xte, Yte = X[split:], Y1[split:]

    # Standardize X for stability
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)

    Xtrz = (Xtr - mu) / sd
    Xtez = (Xte - mu) / sd

    # Ridge with multi-output: solve for each output dim together
    # coef shape: (d, d) mapping X -> Y
    XtX = Xtrz.T @ Xtrz
    XtY = Xtrz.T @ Ytr
    coef = np.linalg.solve(XtX + alpha * np.eye(d), XtY)  # (d, d)

    Yhat = Xtez @ coef  # (n_test, d)

    mse = float(np.mean((Yte - Yhat) ** 2))
    rmse = float(np.sqrt(mse))

    # R2 across all dims pooled
    ss_res = float(np.sum((Yte - Yhat) ** 2))
    ss_tot = float(np.sum((Yte - Yte.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
        "n_test": int(Yte.shape[0]),
        "d": int(d),
        "Yhat": Yhat,
        "Ytrue": Yte,
    }


def main():
    panel_path = OUTPUT_DIR / "stageB_panel.npz"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing {panel_path}. Run:\n  python stageB_build_panel.py")

    data = np.load(panel_path, allow_pickle=True)
    Y_list = data["Y_list"]  # object array of (T_i, d)
    feature_names = list(data["feature_names"])

    rng = np.random.default_rng(RANDOM_SEED)

    metrics_rows = []
    pred_rows = []

    alpha = 10.0
    train_frac = 0.7

    for i, Y in enumerate(Y_list):
        out = ridge_var1_oos(np.asarray(Y, dtype=float), alpha=alpha, train_frac=train_frac)
        if out is None:
            continue

        metrics_rows.append({
            "unit": i,
            "rmse": out["rmse"],
            "mse": out["mse"],
            "r2": out["r2"],
            "n_test": out["n_test"],
            "d": out["d"],
            "alpha": alpha,
            "train_frac": train_frac,
        })

        # Save a small sample of predictions for inspection (first 50 test steps)
        Yhat = out["Yhat"][:50]
        Ytrue = out["Ytrue"][:50]
        for t in range(Yhat.shape[0]):
            pred_rows.append({
                "unit": i,
                "t_test_index": t,
                "ytrue_norm": float(np.linalg.norm(Ytrue[t])),
                "yhat_norm": float(np.linalg.norm(Yhat[t])),
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(OUTPUT_DIR / "stageB_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUTPUT_DIR / "stageB_pred_samples.csv", index=False)

    info = {
        "model": "ridge_VAR1_baseline",
        "alpha": alpha,
        "train_frac": train_frac,
        "feature_dim": len(feature_names),
        "notes": "Baseline Stage B: predicts window-level EEG feature vectors from one lag.",
    }
    (OUTPUT_DIR / "stageB_model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"[OK] Saved:\n- {OUTPUT_DIR/'stageB_metrics.csv'}\n- {OUTPUT_DIR/'stageB_pred_samples.csv'}\n- {OUTPUT_DIR/'stageB_model_info.json'}")


if __name__ == "__main__":
    main()
