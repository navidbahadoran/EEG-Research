# run_full_pvar_task.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR, G_CHANNELS, K_VMF, RF, RG, RH,
    LAM_D, LAM_C, LAM_L, LAM_F, LAM_G, LAM_H,
    MAX_ITER, TOL, RANDOM_SEED, TRAIN_FRAC
)
from eeg.panel_dataset import load_task_panel
from eeg.pvar_full_model import FullPVARFactorALS

def main(task: str = "DespicableMe"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    Y_list, Z_list, subjects, T = load_task_panel(task)
    print(f"[INFO] task={task} | N={len(subjects)} | common_T={T} | G={Y_list[0].shape[1]} | p={Z_list[0].shape[1]}")

    model = FullPVARFactorALS(
        G=G_CHANNELS, p=K_VMF,
        rf=RF, rg=RG, rh=RH,
        lam_D=LAM_D, lam_C=LAM_C, lam_L=LAM_L,
        lam_f=LAM_F, lam_g=LAM_G, lam_h=LAM_H,
        max_iter=MAX_ITER, tol=TOL, seed=RANDOM_SEED
    )

    out = model.forecast_and_score(Y_list, Z_list, train_frac=TRAIN_FRAC)

    metrics = pd.DataFrame([{
        "task": task,
        "N_units": len(subjects),
        "common_T": T,
        "rf": RF, "rg": RG, "rh": RH,
        "lam_D": LAM_D, "lam_C": LAM_C, "lam_L": LAM_L,
        "lam_f": LAM_F, "lam_g": LAM_G, "lam_h": LAM_H,
        "train_frac": TRAIN_FRAC,
        "rmse": out["rmse"],
        "mse": out["mse"],
        "r2": out["r2"],
        "n_test_points": out["n_test_points"],
    }])

    mpath = OUTPUT_DIR / f"full_pvar_{task}_metrics.csv"
    metrics.to_csv(mpath, index=False)

    ppath = OUTPUT_DIR / f"full_pvar_{task}_predictions.npz"
    np.savez(ppath, Ytrue=out["Ytrue"], Ypred=out["Ypred"], subjects=np.array(subjects, dtype=object))

    print(f"[OK] wrote:\n- {mpath}\n- {ppath}")

if __name__ == "__main__":
    # change this if needed
    main(task="DespicableMe")
