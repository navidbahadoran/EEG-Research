# run_full_pvar_all_tasks_parallel.py
import os

# Keep BLAS from oversubscribing when using joblib processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import time
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, TRAIN_FRAC, G_CHANNELS, K_VMF, RF, RG, RH, MAX_ITER
from eeg.panel_dataset import load_task_panel
from eeg.pvar_full_model_parallel_adaptive import PVARFactorALSParallelAdaptive


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index_path = OUTPUT_DIR / "data_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}. Run: python data_index.py")

    df = pd.read_csv(index_path)
    tasks = sorted(df["task"].dropna().unique().tolist())

    print(f"[INFO] Found {len(tasks)} tasks: {tasks}")

    all_metrics = []

    for task in tasks:
        print("\n" + "=" * 80)
        print(f"[TASK] {task}")
        print("=" * 80)

        Y_list, Z_list, subjects, common_T = load_task_panel(task)
        print(f"[INFO] task={task} | N={len(subjects)} | common_T={common_T} | G={G_CHANNELS} | p={K_VMF}")

        model = PVARFactorALSParallelAdaptive(
            G=G_CHANNELS,
            p=K_VMF,
            rf=RF,
            rg=RG,
            rh=RH,
            max_iter=MAX_ITER,
            n_jobs=None,     # adaptive n_jobs per machine
            verbose=True,
        )

        t0 = time.time()
        model.fit(Y_list, Z_list)
        out = model.predict_one_step(Y_list, Z_list, train_frac=TRAIN_FRAC)
        elapsed = time.time() - t0

        # save per-task predictions
        preds_path = OUTPUT_DIR / f"full_pvar_{task}_predictions_parallel.npz"
        np.savez(
            preds_path,
            Ytrue=out["Ytrue"],
            Ypred=out["Ypred"],
            subjects=np.array(subjects, dtype=object),
        )

        # collect metrics
        row = {
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
            "elapsed_sec": float(elapsed),
            "n_jobs_used": int(model.n_jobs),
        }
        all_metrics.append(row)

        # save per-task metrics
        metrics_path = OUTPUT_DIR / f"full_pvar_{task}_metrics_parallel.csv"
        pd.DataFrame([row]).to_csv(metrics_path, index=False)

        print(f"[OK] wrote:\n- {metrics_path}\n- {preds_path}")

    # save combined metrics
    all_path = OUTPUT_DIR / "full_pvar_all_tasks_metrics_parallel.csv"
    pd.DataFrame(all_metrics).to_csv(all_path, index=False)
    print("\n" + "=" * 80)
    print(f"[DONE] wrote combined metrics: {all_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
