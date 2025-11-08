# evaluation.py
from __future__ import annotations
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd

from directional import row_normalize, spherical_kmeans, estimate_vmf_params   # vMF tools  # :contentReference[oaicite:8]{index=8}
from panel import build_response_logpower, build_A_block, build_B_block_time_of_day, build_Z_block  # builders  # :contentReference[oaicite:9]{index=9}
from ife import ife_ic_bai_ng                                                  # IFE/IC      # :contentReference[oaicite:10]{index=10}
from config import K_VMF, RANK_GRID, SESSION_NAMES

def make_train_test_masks(sessions: List[str], train_sessions: List[str]) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(sessions, dtype=object)
    train_mask = np.isin(s, np.asarray(train_sessions, dtype=object))
    test_mask = ~train_mask
    return train_mask, test_mask

def vmf_block(X: np.ndarray) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute vMF posterior features Z_t from directionalized topographies.
    Returns (Z_t: (T,K), report).
    """
    Y_dir = row_normalize(X)                                    # unit rows (T,C)  # :contentReference[oaicite:11]{index=11}
    mu, labels = spherical_kmeans(Y_dir, K=K_VMF, iters=30, seed=42)  # :contentReference[oaicite:12]{index=12}
    kappas, post = estimate_vmf_params(Y_dir, mu, labels)       # :contentReference[oaicite:13]{index=13}
    Z_t = build_Z_block(post)                                   # maps to regressors  # :contentReference[oaicite:14]{index=14}
    rep = {"kappas": kappas, "counts": np.bincount(labels, minlength=K_VMF)}
    return Z_t, rep

def fit_ife_train(y_it_tr, A_tr, B_tr, Z_tr, M_tr):
    """Rank selection + fit on train window only."""
    best_r, ic_vals, fits = ife_ic_bai_ng(y_it_tr, A_tr, B_tr, Z_tr, M_tr, r_grid=RANK_GRID)  # :contentReference[oaicite:15]{index=15}
    return fits[best_r], best_r, ic_vals

def predict_oos(y_it_ts: np.ndarray, beta: np.ndarray, Lam: np.ndarray,
                A_ts: np.ndarray, B_ts: np.ndarray, Z_ts: np.ndarray, M_ts: np.ndarray) -> np.ndarray:
    """
    Out-of-sample prediction for test times:
      f_t = (Lam'Lam)^(-1) Lam' (y_:t - d_t' beta),   yhat = d_t' beta + Lam f_t.
    """
    rows_ts = np.concatenate([A_ts, M_ts * B_ts, Z_ts], axis=1)  # time-shared design
    N, r = Lam.shape
    # precompute pseudo-inverse
    G = np.linalg.pinv(Lam.T @ Lam) @ Lam.T  # (r,N)
    Tt = rows_ts.shape[0]
    yhat = np.zeros((N, Tt))
    for t in range(Tt):
        mu_t = rows_ts[t] @ beta
        y_t  = y_it_ts[:, t]
        f_t  = G @ (y_t - mu_t)
        yhat[:, t] = mu_t + Lam @ f_t
    return yhat

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    resid = y_true - y_pred
    mse = float(np.mean(resid**2))
    var = float(np.var(y_true))
    r2  = 1.0 - (np.mean(resid**2) / (var + 1e-12))
    return {"mse": mse, "r2": r2}

def run_subject_pipeline(X: np.ndarray, sessions: List[str], sex_male1: float, age_years: float,
                         task_rest1: float = 1.0,
                         train_sessions: List[str] | None = None) -> dict:
    """
    Full subject flow: vMF -> y_it/A/B/Z/M -> split -> fit on train -> OOS test.
    """
    if train_sessions is None:
        train_sessions = SESSION_NAMES[:3]   # train on first 3 sessions, test on last

    # Build B/M and split by sessions
    B_t, M_t = build_B_block_time_of_day(sessions)   # :contentReference[oaicite:16]{index=16}
    train_mask, test_mask = make_train_test_masks(sessions, train_sessions)

    # vMF-based Z_t
    Z_t, vmf_rep = vmf_block(X)                      # :contentReference[oaicite:17]{index=17} :contentReference[oaicite:18]{index=18}

    # Response and A-block
    y_it = build_response_logpower(X)                # (N=C, T)  # :contentReference[oaicite:19]{index=19}
    A_t  = build_A_block(T=X.shape[0], sex_male1=sex_male1, age_years=age_years, task_rest1=task_rest1)  # :contentReference[oaicite:20]{index=20}

    # Split all time-shared rows and panel response
    def split_rows(arr): return arr[train_mask], arr[~train_mask]
    A_tr, A_ts = split_rows(A_t); B_tr, B_ts = split_rows(B_t); M_tr, M_ts = split_rows(M_t); Z_tr, Z_ts = split_rows(Z_t)
    y_it_tr, y_it_ts = y_it[:, train_mask], y_it[:, ~train_mask]

    # Fit on train only
    fit, best_r, ic_vals = fit_ife_train(y_it_tr, A_tr, B_tr, Z_tr, M_tr)

    # In-sample report
    rows_tr = np.concatenate([A_tr, M_tr * B_tr, Z_tr], axis=1)
    yhat_tr = (rows_tr @ fit.beta)[None, :] + fit.Lam @ fit.F.T
    tr_metrics = evaluate(y_it_tr, yhat_tr)

    # Out-of-sample (estimate f_t on test given fixed beta, Lam)
    yhat_ts = predict_oos(y_it_ts, fit.beta, fit.Lam, A_ts, B_ts, Z_ts, M_ts)
    ts_metrics = evaluate(y_it_ts, yhat_ts)

    # Coeff table
    coef_names = (
        [f"A:{n}" for n in ["intercept", "sex(male=1)", "age(centered)", "task(rest)"]]
        + [f"B:ToD_{n}" for n in ["sin", "cos"]]
        + [f"Z:vMF_p{k+1}" for k in range(Z_t.shape[1])]
    )
    beta_df = pd.DataFrame({"coef": coef_names, "estimate": fit.beta})

    return {
        "fit": fit,
        "best_r": best_r,
        "ic_vals": {k: float(v) for k, v in ic_vals.items()},
        "vmf": vmf_rep,
        "train_metrics": tr_metrics,
        "test_metrics": ts_metrics,
        "beta": beta_df,
    }
