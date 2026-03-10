from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from eeg.slds_panel import PanelSLDSPrototype
from scripts.model_common import load_panel_data, save_metrics_row, standardize_from_train


def run_slds_panel_prototype(
    K: int = 3,
    latent_dim: int | None = None,
    subset_units: int | None = None,
):
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pooled vMF panel ...")
    data = load_panel_data()

    Y_list = data["Y_list"]
    X_list = data["X_list"]
    units = np.asarray(data["units"], dtype=object)
    tasks = np.asarray(data["tasks"], dtype=object)
    feature_names = np.asarray(data["feature_names"], dtype=object)
    common_T = int(data["T"])

    if subset_units is not None:
        Y_list = Y_list[:subset_units]
        X_list = X_list[:subset_units]
        units = units[:subset_units]
        tasks = tasks[:subset_units]
        print(f"Using subset of units: {subset_units}")

    if len(Y_list) == 0:
        raise ValueError("Y_list is empty.")
    if len(X_list) != len(Y_list):
        raise ValueError("X_list and Y_list must have the same length.")

    train_end = max(4, min(common_T - 1, int(np.floor(float(config.TRAIN_FRAC) * common_T))))
    X_list, x_mean, x_std = standardize_from_train(X_list, train_end)

    d = Y_list[0].shape[1]
    p = X_list[0].shape[1]

    if latent_dim is None:
        latent_dim = int(getattr(config, "RF", 3))

    print(f"N units        : {len(Y_list)}")
    print(f"Common T       : {common_T}")
    print(f"Response dim d : {d}")
    print(f"Covariates p   : {p}")
    print(f"K regimes      : {K}")
    print(f"Latent dim     : {latent_dim}")
    print(f"Train fraction : {float(config.TRAIN_FRAC):.3f}")

    model = PanelSLDSPrototype(
        latent_dim=latent_dim,
        K=K,
        ridge=1e-4,
        max_iter=max(10, int(config.MAX_ITER)),
        tol=float(config.TOL),
        random_state=int(config.RANDOM_SEED),
        verbose=False,
    )

    print("Fitting SLDS-style panel prototype ...")
    result = model.forecast_and_score(
        Y_list,
        X_list,
        train_frac=float(config.TRAIN_FRAC),
    )

    print("\nFinished.")
    print("Metrics:")
    for k, v in result.metrics.items():
        print(f"  {k:>14s}: {v:.6f}")

    metrics_path = out_dir / f"slds_panel_k{K}_r{latent_dim}_metrics.csv"
    save_metrics_row(metrics_path, {
        "mode": f"slds_panel_k{K}_r{latent_dim}",
        "N_units": len(units),
        "common_T": common_T,
        "K": K,
        "latent_dim": latent_dim,
        "p": p,
        "mse": result.metrics["mse"],
        "rmse": result.metrics["rmse"],
        "r2": result.metrics["r2"],
        "accuracy": result.metrics["accuracy"],
        "kl": result.metrics["kl"],
        "cross_entropy": result.metrics["cross_entropy"],
        "train_frac": float(config.TRAIN_FRAC),
        "train_end": int(result.train_end),
        "max_iter_used": int(len(result.loss_history)),
    })

    artifacts_path = out_dir / f"slds_panel_k{K}_r{latent_dim}_artifacts.npz"
    np.savez_compressed(
        artifacts_path,
        units=units,
        tasks=tasks,
        feature_names=feature_names,
        x_mean=x_mean,
        x_std=x_std,
        y_true_oof=result.y_true_oof,
        y_pred_oof=result.y_pred_oof,
        x_true_oof=result.x_true_oof,
        x_pred_oof=result.x_pred_oof,
        regime_prob_pred=result.regime_prob_pred,
        regime_prob_filt=result.regime_prob_filt,
        C=result.C,
        d=result.d,
        A_latent=result.A_latent,
        B_latent=result.B_latent,
        c_latent=result.c_latent,
        sigma2=result.sigma2,
        Pi=result.Pi,
        pi0=result.pi0,
        loss_history=result.loss_history,
        train_end=np.array([int(result.train_end)]),
    )

    print(f"\nSaved metrics   : {metrics_path}")
    print(f"Saved artifacts : {artifacts_path}")

    print("\nTransition matrix Pi:")
    print(pd.DataFrame(
        result.Pi,
        index=[f"from_regime_{j+1}" for j in range(result.Pi.shape[0])],
        columns=[f"to_regime_{j+1}" for j in range(result.Pi.shape[1])],
    ))

    return metrics_path, artifacts_path


if __name__ == "__main__":
    run_slds_panel_prototype(
        K=3,
        latent_dim=int(getattr(config, "RF", 3)),
        subset_units=None,
    )