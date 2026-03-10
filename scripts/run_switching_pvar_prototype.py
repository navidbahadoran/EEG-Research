from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from eeg.switching_pvar import SwitchingPVARPrototype
from scripts.model_common import load_panel_data, save_metrics_row, standardize_from_train


def run_switching_pvar_prototype(K_regimes: int = 3, ridge: float = 1.0):
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_panel_data()

    Y_list = data["Y_list"]
    X_list = data["X_list"]
    units = np.asarray(data["units"], dtype=object)
    tasks = np.asarray(data["tasks"], dtype=object)
    feature_names = np.asarray(data["feature_names"], dtype=object)
    T = int(data["T"])

    train_end = max(4, min(T - 1, int(np.floor(float(config.TRAIN_FRAC) * T))))
    X_list, x_mean, x_std = standardize_from_train(X_list, train_end)

    G = Y_list[0].shape[1]
    p = X_list[0].shape[1]

    print(
        f"[INFO] switching pooled VARX | units={len(units)} | common_T={T} | "
        f"G={G} | p={p} | K_regimes={K_regimes}"
    )

    model = SwitchingPVARPrototype(
        K=K_regimes,
        G=G,
        p=p,
        ridge=ridge,
        max_iter=25,
        tol=1e-4,
        seed=int(config.RANDOM_SEED),
        verbose=True,
    )

    out = model.forecast_and_score(Y_list, X_list, train_frac=float(config.TRAIN_FRAC))

    metrics_path = out_dir / f"switching_pvar_k{K_regimes}_metrics.csv"
    save_metrics_row(metrics_path, {
        "mode": f"switching_pvar_k{K_regimes}",
        "N_units": len(units),
        "common_T": T,
        "K": G,
        "p": p,
        "n_regimes": K_regimes,
        "ridge": ridge,
        "train_frac": float(config.TRAIN_FRAC),
        "train_end": int(out["train_end"]),
        "mse": float(out["mse"]),
        "rmse": float(out["rmse"]),
        "r2": float(out["r2"]),
        "accuracy": float(out["accuracy"]),
        "kl": float(out["kl"]),
        "cross_entropy": float(out["cross_entropy"]),
    })

    artifacts_path = out_dir / f"switching_pvar_k{K_regimes}_artifacts.npz"
    np.savez_compressed(
        artifacts_path,
        units=units,
        tasks=tasks,
        feature_names=feature_names,
        x_mean=x_mean,
        x_std=x_std,
        y_true_oof=out["Ytrue"],
        y_pred_oof=out["Ypred"],
        regime_prob_pred=out["regime_prob_pred"],
        regime_prob_filt=out["regime_prob_filt"],
        W=model.W_,
        sigma2=model.sigma2_,
        Pi=model.Pi_,
        pi0=model.pi0_,
        loss_history=np.array(model.loss_history_, dtype=float),
        train_end=np.array([int(out["train_end"])]),
    )

    print("[OK] wrote:")
    print("-", metrics_path)
    print("-", artifacts_path)
    return str(metrics_path), str(artifacts_path)


if __name__ == "__main__":
    run_switching_pvar_prototype(K_regimes=3, ridge=1.0)