from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import config
from eeg.pvar_full_model_parallel_adaptive import PVARFactorALSParallelAdaptive
from vmf.vmf_panel_builder_pooled import make_vmf_panel_pooled


def _fit_standardizer_from_train(
    X_list: list[np.ndarray],
    feature_names: list[str],
    train_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit train-only means/stds.

    We keep sex and task dummies unscaled.
    Everything else is mean-imputed and standardized.
    """
    X_train = np.vstack([X[:train_end - 1] for X in X_list])  # causal alignment uses X_t for Y_{t+1}

    means = np.nanmean(X_train, axis=0)
    means = np.where(np.isnan(means), 0.0, means)

    stds = np.nanstd(X_train, axis=0)
    stds = np.where((stds < 1e-8) | np.isnan(stds), 1.0, stds)

    for j, name in enumerate(feature_names):
        if name == "sex" or name.startswith("task_"):
            means[j] = 0.0
            stds[j] = 1.0

    return means, stds


def _apply_standardizer(
    X_list: list[np.ndarray],
    means: np.ndarray,
    stds: np.ndarray,
    feature_names: list[str],
) -> list[np.ndarray]:
    X_out = []

    for X in X_list:
        X2 = np.asarray(X, dtype=float).copy()

        for j, name in enumerate(feature_names):
            col = X2[:, j]
            fill_val = 0.0 if (name == "sex" or name.startswith("task_")) else means[j]
            col = np.where(np.isnan(col), fill_val, col)

            if not (name == "sex" or name.startswith("task_")):
                col = (col - means[j]) / stds[j]

            X2[:, j] = col

        X_out.append(X2)

    return X_out


def _build_latent_subject_summary(
    units: list[str],
    tasks: list[str],
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

    (
        units,
        tasks,
        Y_list,
        X_list,
        y_targets,
        T,
        task_levels,
        baseline,
        feature_names,
        unit_meta,
    ) = make_vmf_panel_pooled(
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

    train_end = max(4, min(T - 1, int(np.floor(float(config.TRAIN_FRAC) * T))))
    x_means, x_stds = _fit_standardizer_from_train(X_list, feature_names, train_end=train_end)
    X_list_std = _apply_standardizer(X_list, x_means, x_stds, feature_names)

    K = Y_list[0].shape[1]
    p = X_list_std[0].shape[1]

    print(f"[INFO] pooled vMF-PVAR | units={len(units)} | common_T={T} | K={K} | p={p}")
    print(f"[INFO] tasks={task_levels} | baseline(dummy dropped)={baseline}")
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

    out = model.forecast_and_score(Y_list, X_list_std, train_frac=float(config.TRAIN_FRAC))

    metrics_path = out_dir / "vmf_pvar_pooled_metrics.csv"
    artifacts_path = out_dir / "vmf_pvar_pooled_artifacts.npz"
    targets_path = out_dir / "vmf_pvar_pooled_targets.csv"
    latent_summary_path = out_dir / "vmf_pvar_pooled_latent_subject_summary.csv"
    unit_meta_path = out_dir / "vmf_pvar_pooled_unit_meta.csv"
    feature_info_path = out_dir / "vmf_pvar_pooled_feature_info.csv"

    metrics_df = pd.DataFrame(
        [
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
                "baseline_task": baseline,
                "task_levels": "|".join(task_levels),
            }
        ]
    )
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

    y_targets.to_csv(targets_path)
    unit_meta.to_csv(unit_meta_path)

    latent_summary_df = _build_latent_subject_summary(
        units=units,
        tasks=tasks,
        y_targets=y_targets,
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
    print("-", targets_path)
    print("-", latent_summary_path)
    print("-", unit_meta_path)
    print("-", feature_info_path)

    return str(metrics_path), str(artifacts_path), str(targets_path)


if __name__ == "__main__":
    run_vmf_pvar_pooled()