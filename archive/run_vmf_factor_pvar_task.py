# run_vmf_factor_pvar_task.py
import numpy as np
import pandas as pd
from vmf_panel_builder import make_vmf_panel_from_csv
from eeg.pvar_full_model_parallel_adaptive import FullPVARFactorALS  # use your class

def run_one_task(
    task: str,
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
):
    subjects, Y_list, X_list, y_targets, common_T = make_vmf_panel_from_csv(
        csv_path=csv_path,
        vmf_dir=vmf_dir,
        task=task,
        stride=stride,
        targets=("attention", "p_factor"),
    )
    K = Y_list[0].shape[1]
    p = X_list[0].shape[1]

    print(f"[INFO] vMF-PVAR task={task} | N={len(subjects)} | T={common_T} | K={K} | p={p}")

    model = FullPVARFactorALS(
        G=K, p=p, rf=rf, rg=rg, rh=rh,
        ridge=ridge, max_iter=max_iter
    )

    out = model.forecast_and_score(Y_list, X_list, train_frac=train_frac)

    # Save prediction metrics
    metrics_path = f"{out_dir}/vmf_pvar_{task}_metrics.csv"
    pd.DataFrame([{
        "task": task,
        "N_units": len(subjects),
        "common_T": common_T,
        "K": K,
        "p": p,
        "rf": rf, "rg": rg, "rh": rh,
        "max_iter": max_iter,
        "train_frac": train_frac,
        "rmse": out["rmse"],
        "mse": out["mse"],
        "r2": out["r2"],
    }]).to_csv(metrics_path, index=False)

    # Save predictions + factors + loadings for diagnostics
    npz_path = f"{out_dir}/vmf_pvar_{task}_artifacts.npz"
    np.savez(
        npz_path,
        subjects=np.array(subjects, dtype=object),
        y_true_oof=out["y_true_oof"],    # (N, T_test, K)
        y_pred_oof=out["y_pred_oof"],    # (N, T_test, K)
        f_t=model.f_t,                   # (T, rf) after fit on full train portion inside method (depends on your implementation)
        Lambda=model.Lambda,             # (N, K, rf)
    )

    print("[OK] wrote:")
    print("-", metrics_path)
    print("-", npz_path)

    return metrics_path, npz_path, y_targets