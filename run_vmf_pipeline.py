"""run_vmf_pipeline.py

End-to-end pipeline for current vMF time series data:
  1) Read summary CSV
  2) Map `probabilities_file` -> local `data/vmf_npz/<filename>.npz`
  3) Load npz time series Z_{it}
  4) Extract dynamic features
  5) Fit simple ridge baselines to predict traits (p_factor / attention)
     with a simple 5-fold split over subjects.

Usage:
  python run_vmf_pipeline.py --csv vmf_fixedmus_summary_K7.csv --K 7
"""

from __future__ import annotations
import argparse
import numpy as np

from vmf_dataset import build_vmf_feature_table, aggregate_subject_level

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    # beta = (X'X + lam I)^-1 X'y
    XtX = X.T @ X
    beta = np.linalg.solve(XtX + lam*np.eye(X.shape[1]), X.T @ y)
    return beta

def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = y.astype(float); yhat = yhat.astype(float)
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean((y - yhat)**2))

def kfold_indices(n: int, k: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for j in range(k):
        te = folds[j]
        tr = np.concatenate([folds[m] for m in range(k) if m != j])
        yield tr, te

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to vmf summary CSV")
    ap.add_argument("--K", type=int, default=None, help="Number of vMF components (optional)")
    ap.add_argument("--lam", type=float, default=10.0, help="Ridge penalty")
    args = ap.parse_args()

    df_task = build_vmf_feature_table(args.csv, K=args.K)
    missing = df_task.attrs.get("missing_npz", 0)
    print(f"Loaded rows: {len(df_task)}  (missing local npz: {missing})")

    df_subj = aggregate_subject_level(df_task)
    print(f"Subject-level rows: {len(df_subj)}")

    feat_cols = [c for c in df_subj.columns if c.startswith("occ_") or c in ("switch_rate","volatility","entropy_mean","entropy_std")]
    X0 = df_subj[feat_cols].to_numpy(float)

    # Standardize within each fold (avoid leakage)
    n = X0.shape[0]
    X = np.column_stack([np.ones(n), X0])  # intercept + features

    for target in ["p_factor","attention","internalizing","externalizing"]:
        if target not in df_subj.columns:
            continue
        y = df_subj[target].to_numpy(float)
        preds = np.zeros_like(y)
        for tr, te in kfold_indices(n, k=5, seed=0):
            mu = X0[tr].mean(axis=0)
            sd = X0[tr].std(axis=0)
            sd[sd == 0] = 1.0
            Xtr = np.column_stack([np.ones(len(tr)), (X0[tr]-mu)/sd])
            Xte = np.column_stack([np.ones(len(te)), (X0[te]-mu)/sd])
            beta = ridge_fit(Xtr, y[tr], lam=args.lam)
            preds[te] = Xte @ beta

        print(f"{target}: R2={r2_score(y,preds):.3f}, MSE={mse(y,preds):.3f}")

    df_subj.to_csv("vmf_subject_features.csv", index=False)
    print("Wrote vmf_subject_features.csv")

if __name__ == "__main__":
    main()
