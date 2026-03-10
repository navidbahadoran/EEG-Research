# run_full_pvar_task_parallel.py
import os

# IMPORTANT: set BEFORE importing numpy / scipy / joblib / sklearn
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import time

from config import (
    OUTPUT_DIR,
    TRAIN_FRAC,
    G_CHANNELS,
    K_VMF,
    RF,
    RG,
    RH,
    MAX_ITER,
)
from eeg.panel_dataset import load_task_panel
from eeg.pvar_full_model_parallel_adaptive import PVARFactorALSParallelAdaptive


def main(task: str, n_jobs: int | None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    Y_list, Z_list, subjects, common_T = load_task_panel(task)

    print(f"[INFO] task={task} | N={len(subjects)} | common_T={common_T} | G={G_CHANNELS} | p={K_VMF}")

    model = PVARFactorALSParallelAdaptive(
        G=G_CHANNELS,
        p=K_VMF,
        rf=RF,
        rg=RG,
        rh=RH,
        max_iter=MAX_ITER,
        n_jobs=n_jobs,      # None => auto
        verbose=True,
    )

    t0 = time.time()
    model.fit(Y_list, Z_list)
    out = model.predict_one_step(Y_list, Z_list, train_frac=TRAIN_FRAC)
    dt = time.time() - t0

    metrics = pd.DataFrame([{
        "task": task,
        "N_units": len(subjects),
        "common_T": int(common_T),
        "G": int(G_CHANNELS),
        "p": int(K_VMF),
        "rf": int(RF),
        "rg": int(RG),
        "rh": int(RH),
        "max_iter": int(MAX_ITER),
        "train_frac": float(TRAIN_FRAC),
        "rmse": float(out["rmse"]),
        "mse": float(out["mse"]),
        "r2": float(out["r2"]),
        "elapsed_sec": float(dt),
        "n_jobs_used": int(model.n_jobs),
        "blas_threads_forced": 1,
    }])

    metrics_path = OUTPUT_DIR / f"full_pvar_{task}_metrics_parallel.csv"
    metrics.to_csv(metrics_path, index=False)

    preds_path = OUTPUT_DIR / f"full_pvar_{task}_predictions_parallel.npz"
    np.savez(
        preds_path,
        Ytrue=out["Ytrue"],
        Ypred=out["Ypred"],
        subjects=np.array(subjects, dtype=object),
    )

    print(f"[OK] wrote:\n- {metrics_path}\n- {preds_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="DespicableMe")
    parser.add_argument("--n_jobs", type=int, default=None, help="Override adaptive n_jobs. Default: auto.")
    args = parser.parse_args()

    main(task=args.task, n_jobs=args.n_jobs)
