"""run_varx_demo.py

Small demo to sanity-check VARX code without requiring the EEG tensor.

It simulates a stable VAR process and fits a ridge VARX. The goal is just to
verify the pipeline runs end-to-end.
"""

from __future__ import annotations

import argparse
import numpy as np

from varx import fit_varx_ridge, score_r2


def simulate_var(T: int = 2000, C: int = 8, lags: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Random stable-ish coefficients by shrinking spectral radius
    A_list = []
    for _ in range(lags):
        A = rng.normal(scale=0.15, size=(C, C))
        A_list.append(A)
    Y = np.zeros((T, C), dtype=float)
    eps = rng.normal(scale=1.0, size=(T, C))
    for t in range(lags, T):
        y = np.zeros(C)
        for ell, A in enumerate(A_list, start=1):
            y += Y[t - ell] @ A
        Y[t] = y + eps[t]
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lags", type=int, default=2)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--C", type=int, default=8)
    args = ap.parse_args()

    Y = simulate_var(T=args.T, C=args.C, lags=max(1, args.lags), seed=0)
    fit = fit_varx_ridge(Y, X_exog=None, lags=args.lags, ridge=1e-3)
    Yhat = fit.predict(Y)
    r2 = score_r2(Y, Yhat, lags=args.lags)
    print(f"VARX demo finished. lags={args.lags}, R2={r2:.3f}")


if __name__ == "__main__":
    main()
