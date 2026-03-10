from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from eeg.pvar_full_model_parallel_adaptive import PVARFactorALSParallelAdaptive
from scripts.model_common import load_panel_data, standardize_from_train


def _build_latent_subject_summary(
    units: list[str],
    y_targets: pd.DataFrame,
    model: PVARFactorALSParallelAdaptive,
) -> pd.DataFrame:
    rows = []
    f_bar = np.mean(model.f_, axis=0)

    for i, unit in enumerate(units):
        Lambda_i = model.params_[i].Lambda
        latent_bar = Lambda_i @ f_bar

        row = {
            "unit_id": unit,
            "lambda_fro": float(np.linalg.norm(Lambda_i)),
            "lambda_abs_mean": float(np.mean(np.abs(Lambda_i))),
            "latent_bar_norm": float(np.linalg.norm(latent_bar)),
            "latent_bar_mean": float(np.mean(latent_bar)),
            "latent_bar_std": float(np.std(latent_bar)),
        }

        for r in range(Lambda_i.shape[1]):
            row[f"lambda_colnorm_{r}"] = float(np.linalg.norm(Lambda_i[:, r]))

        rows.append(row)

    out = pd.DataFrame(rows).set_index("unit_id")

    if y_targets is not None and len(y_targets) > 0:
        overlap = out.columns.intersection(y_targets.columns)
        if len(overlap) > 0:
            y_targets_use = y_targets.drop(columns=list(overlap))
        else:
            y_targets_use = y_targets
        out = out.join(y_targets_use, how="left")

    return out.reset_index()


def run_vmf_pvar_pooled():
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_panel_data()

    units = list(data["units"])
    tasks = list(data["tasks"])
    Y_list = data["Y_list"]
    X_list = data["X_list"]
    feature_names = list(data["feature_names"])
    y_targets = data.get("y_targets", None)
    T = int(data["T"])

    train_end = max(4, min(T - 1, int(np.floor(float(config.TRAIN_FRAC) * T))))
    X_list, x_means, x_stds = standardize_from_train(X_list, train_end)

    K = Y_list[0].shape[1]
    p = X_list[0].shape[1]

    print(f"[INFO] pooled vMF-PVAR | units={len(units)} | common_T={T} | K={K} | p={p}")
    print(f"[INFO] train_end={train_end} | train_frac={float(config.TRAIN_FRAC):.3f}")

    model = PVARFactorALSParallelAdaptive(
        G=K,
        p=p,
        rf=int(config.RF),
        rg=int(config.RG),
        rh=int(config.RH),
        lam_A=float(config.LAM_D),
        lam_B=float(config.LAM_C),
        lam_L=float(config.LAM_L),
        lam_f=float(config.LAM_F),
        lam_g=float(config.LAM_G),
        lam_h=float(config.LAM_H),
        max_iter=int(config.MAX_ITER),
        tol=float(config.TOL),
        seed=int(config.RANDOM_SEED),
        verbose=True,
    )

    out = model.forecast_and_score(Y_list, X_list, train_frac=float(config.TRAIN_FRAC))

    metrics_path = out_dir / "vmf_pvar_pooled_metrics.csv"
    artifacts_path = out_dir / "vmf_pvar_pooled_artifacts.npz"
    latent_summary_path = out_dir / "vmf_pvar_pooled_latent_subject_summary.csv"
    feature_info_path = out_dir / "vmf_pvar_pooled_feature_info.csv"

    metrics_df = pd.DataFrame([
        {
            "mode": "vmf_pooled_factor_pvar",
            "N_units": len(units),
            "common_T": T,
            "K": K,
            "p": p,
            "rf": int(config.RF),
            "rg": int(config.RG),
            "rh": int(config.RH),
            "max_iter": int(config.MAX_ITER),
            "tol": float(config.TOL),
            "train_frac": float(config.TRAIN_FRAC),
            "train_end": int(out["train_end"]),
            "mse": float(out["mse"]),
            "rmse": float(out["rmse"]),
            "r2": float(out["r2"]),
            "accuracy": float(out["accuracy"]),
            "kl": float(out["kl"]),
            "cross_entropy": float(out["cross_entropy"]),
        }
    ])
    metrics_df.to_csv(metrics_path, index=False)

    np.savez_compressed(
        artifacts_path,
        units=np.array(units, dtype=object),
        tasks=np.array(tasks, dtype=object),
        feature_names=np.array(feature_names, dtype=object),
        x_mean=x_means,
        x_std=x_stds,
        y_true_oof=out["Ytrue"],
        y_pred_oof=out["Ypred"],
        f_t=model.f_,
        g_t=model.g_,
        h_t=model.h_,
        Lambda=model.Lambda_,
        loss_history=np.array(model.loss_history_, dtype=float),
        train_end=np.array([int(out["train_end"])]),
    )

    if y_targets is None:
        y_targets_df = pd.DataFrame(index=np.array(units, dtype=object))
    else:
        y_targets_df = y_targets.copy()
        if "unit_id" in y_targets_df.columns:
            y_targets_df = y_targets_df.set_index("unit_id")

    latent_summary_df = _build_latent_subject_summary(
        units=units,
        y_targets=y_targets_df,
        model=model,
    )
    latent_summary_df.to_csv(latent_summary_path, index=False)

    feature_info_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "mean_train": x_means,
            "std_train": x_stds,
        }
    )
    feature_info_df.to_csv(feature_info_path, index=False)

    print("[OK] wrote:")
    print("-", metrics_path)
    print("-", artifacts_path)
    print("-", latent_summary_path)
    print("-", feature_info_path)

    return str(metrics_path), str(artifacts_path)


if __name__ == "__main__":
    run_vmf_pvar_pooled()