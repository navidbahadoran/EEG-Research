# run_vmf_factor_pvar_pooled.py
import numpy as np
import pandas as pd

from vmf_panel_builder_pooled import make_vmf_panel_pooled

# Use your existing model class (adjust import if your path differs)
from eeg.pvar_full_model_parallel_adaptive import FullPVARFactorALS

def run_pooled(
    csv_path: str,
    vmf_dir: str,
    out_dir: str = "outputs",
    stride: int = 10,
    rf: int = 2,
    rg: int = 2,
    rh: int = 2,
    max_iter: int = 15,
    train_frac: float = 0.7,
    ridge: float = 1e-3,
    drop_baseline_task: str | None = None,
):
    units, tasks, Y_list, X_list, y_targets, T, task_levels, baseline = make_vmf_panel_pooled(
        csv_path=csv_path,
        vmf_dir=vmf_dir,
        stride=stride,
        drop_baseline_task=drop_baseline_task,
        targets=("attention", "p_factor"),
    )

    K = Y_list[0].shape[1]
    p = X_list[0].shape[1]
    print(f"[INFO] pooled vMF-PVAR | units={len(units)} | common_T={T} | K={K} | p={p}")
    print(f"[INFO] tasks={task_levels} | baseline(dummy dropped)={baseline}")

    model = FullPVARFactorALS(G=K, p=p, rf=rf, rg=rg, rh=rh, ridge=ridge, max_iter=max_iter)

    out = model.forecast_and_score(Y_list, X_list, train_frac=train_frac)

    # Metrics
    metrics = {
        "mode": "vmf_pooled",
        "N_units": len(units),
        "common_T": T,
        "K": K,
        "p": p,
        "rf": rf, "rg": rg, "rh": rh,
        "max_iter": max_iter,
        "train_frac": train_frac,
        "rmse": float(out["rmse"]),
        "mse": float(out["mse"]),
        "r2": float(out["r2"]),
        "baseline_task": baseline,
        "task_levels": "|".join(task_levels),
    }
    metrics_path = f"{out_dir}/vmf_pvar_pooled_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    # Save artifacts: predictions + factors + loadings + target table
    # NOTE: adjust attribute names if your model uses different ones.
    npz_path = f"{out_dir}/vmf_pvar_pooled_artifacts.npz"
    np.savez(
        npz_path,
        units=np.array(units, dtype=object),
        tasks=np.array(tasks, dtype=object),
        y_true_oof=out["y_true_oof"],   # (N, T_test, K) if your forecast_and_score returns this
        y_pred_oof=out["y_pred_oof"],   # (N, T_test, K)
        f_t=getattr(model, "f_t", None),
        Lambda=getattr(model, "Lambda", None),
    )

    targets_path = f"{out_dir}/vmf_pvar_pooled_targets.csv"
    y_targets.to_csv(targets_path)

    print("[OK] wrote:")
    print("-", metrics_path)
    print("-", npz_path)
    print("-", targets_path)

    return metrics_path, npz_path, targets_path

if __name__ == "__main__":
    # Example usage: fill your paths via config.py in your repo
    raise SystemExit("Run this via a small wrapper that passes csv_path and vmf_dir from config.")