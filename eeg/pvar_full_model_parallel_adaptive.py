# eeg/pvar_full_model_parallel_adaptive.py
from __future__ import annotations

import os
import time
import numpy as np
from joblib import Parallel, delayed


def _env_int(name: str):
    v = os.environ.get(name)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def detect_blas_threads(default_if_unknown: int = 4) -> int:
    """
    Best-effort detection from environment variables.
    If unset, assume a conservative >1 (default_if_unknown).
    """
    for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        val = _env_int(k)
        if val is not None and val > 0:
            return val
    return default_if_unknown


def auto_n_jobs(max_cap: int = 12, frac: float = 0.6, assume_blas_threads_if_unknown: int = 4) -> int:
    """
    Adaptive heuristic:
      - start from logical cores
      - divide by BLAS threads to avoid oversubscription
      - take frac of that budget
      - cap at max_cap (Windows overhead, memory)
    """
    logical = os.cpu_count() or 4
    blas_threads = detect_blas_threads(default_if_unknown=assume_blas_threads_if_unknown)
    budget = max(1, logical // max(1, blas_threads))
    n_jobs = int(round(frac * budget))
    return max(1, min(max_cap, n_jobs))


def ridge_solve(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve min_B ||Y - X B||^2 + lam ||B||^2
    X: (n, d), Y: (n, q) => B: (d, q)
    """
    XtX = X.T @ X
    XtY = X.T @ Y
    d = XtX.shape[0]
    return np.linalg.solve(XtX + lam * np.eye(d), XtY)


def _update_subject_block(
    i: int,
    Ylag_i: np.ndarray,   # (T1, G)
    Ycur_i: np.ndarray,   # (T1, G)
    Zcur_i: np.ndarray,   # (T1, p)
    f: np.ndarray,        # (T1, rf)
    g: np.ndarray,        # (T1, rg)
    h: np.ndarray,        # (T1, rh)
    D_i: list[np.ndarray],      # rh of (G,G)
    C_i: list[np.ndarray],      # rg of (G,p)
    Lambda_i: np.ndarray,       # (G, rf)
    lam_A: float,
    lam_B: float,
    lam_L: float,
):
    """
    Update subject-specific parameters given factors:
      - Lambda_i
      - D_i[m] for m=1..rh
      - C_i[m] for m=1..rg

    Returns updated (i, D_i, C_i, Lambda_i).
    """
    T1, G = Ycur_i.shape
    p = Zcur_i.shape[1]
    rf = f.shape[1]
    rg = g.shape[1]
    rh = h.shape[1]

    # ---- Update Lambda_i via regression: residual(t) ≈ Lambda_i f_t
    # residual = y - A_t ylag - B_t z
    X = f                           # (T1, rf)
    Y = np.zeros((T1, G), dtype=float)

    for t in range(T1):
        # A_t = sum_m h[t,m] D_i[m]
        A_t = np.zeros((G, G), dtype=float)
        for m in range(rh):
            A_t += h[t, m] * D_i[m]

        # B_t = sum_m g[t,m] C_i[m]
        B_t = np.zeros((G, p), dtype=float)
        for m in range(rg):
            B_t += g[t, m] * C_i[m]

        Y[t] = Ycur_i[t] - (A_t @ Ylag_i[t] + B_t @ Zcur_i[t])

    # Solve Y ≈ X @ (Lambda_i^T)  => Lambda_i = (solve)^T
    Lambda_i = ridge_solve(X, Y, lam_L).T   # (G, rf)

    # ---- Precompute factor residual part for speed
    # y_tilde = y - Lambda f
    Ytilde = Ycur_i - (f @ Lambda_i.T)      # (T1, G)

    # ---- Update D_i[m]: Ytilde[t] ≈ (sum_m h[t,m] D_i[m]) ylag[t]
    # We update each D_i[m] via a ridge regression:
    #   target: Ytilde
    #   predictors: h[t,m] * ylag[t]
    # Solve: Ytilde ≈ Xd @ (D_m^T), with Xd: (T1, G)
    for m in range(rh):
        Xd = (h[:, [m]] * Ylag_i)           # (T1, G) broadcast
        D_i[m] = ridge_solve(Xd, Ytilde, lam_A).T   # (G,G)

    # ---- Update C_i[m]: Ytilde[t] ≈ (sum_m g[t,m] C_i[m]) z[t]
    # Update each C_i[m] via:
    #   target: Ytilde
    #   predictors: g[t,m] * z[t]
    # Solve: Ytilde ≈ Xc @ (C_m^T), Xc: (T1, p)
    for m in range(rg):
        Xc = (g[:, [m]] * Zcur_i)           # (T1, p)
        C_i[m] = ridge_solve(Xc, Ytilde, lam_B).T   # (G,p)

    return i, D_i, C_i, Lambda_i


class PVARFactorALSParallelAdaptive:
    """
    Parallel ALS (per-subject updates) for:
      y_it = A_it y_i,t-1 + B_it z_it + Lambda_i f_t + u_it
    with factor structures:
      A_it = sum_m h_t[m] D_i[m]
      B_it = sum_m g_t[m] C_i[m]

    Notes:
    - Parallelization is across subjects (i).
    - Factor updates (f) are updated serially for stability.
    - g and h are currently held fixed (can be extended later).
    """

    def __init__(
        self,
        G: int,
        p: int,
        rf: int = 1,
        rg: int = 1,
        rh: int = 1,
        lam_A: float = 10.0,
        lam_B: float = 10.0,
        lam_L: float = 10.0,
        lam_f: float = 1.0,
        max_iter: int = 5,
        seed: int = 123,
        n_jobs: int | None = None,
        n_jobs_cap: int = 12,
        n_jobs_frac: float = 0.6,
        verbose: bool = True,
    ):
        self.G = G
        self.p = p
        self.rf = rf
        self.rg = rg
        self.rh = rh

        self.lam_A = lam_A
        self.lam_B = lam_B
        self.lam_L = lam_L
        self.lam_f = lam_f

        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

        if n_jobs is None:
            self.n_jobs = auto_n_jobs(max_cap=n_jobs_cap, frac=n_jobs_frac)
        else:
            self.n_jobs = int(n_jobs)

    def fit(self, Y_list: list[np.ndarray], Z_list: list[np.ndarray]):
        N = len(Y_list)
        T, G = Y_list[0].shape
        T1 = T - 1

        # lagged and current
        Ylag = [Y[:-1].astype(float, copy=False) for Y in Y_list]
        Ycur = [Y[1:].astype(float, copy=False) for Y in Y_list]
        Zcur = [Z[1:].astype(float, copy=False) for Z in Z_list]

        # init factors
        f = self.rng.standard_normal((T1, self.rf))
        g = self.rng.standard_normal((T1, self.rg))
        h = self.rng.standard_normal((T1, self.rh))

        # init subject params
        Lambda = [self.rng.standard_normal((G, self.rf)) * 0.05 for _ in range(N)]
        D = [[self.rng.standard_normal((G, G)) * 0.05 for _ in range(self.rh)] for _ in range(N)]
        C = [[self.rng.standard_normal((G, self.p)) * 0.05 for _ in range(self.rg)] for _ in range(N)]

        if self.verbose:
            logical = os.cpu_count() or -1
            blas_threads = detect_blas_threads()
            print(f"[ALS-PAR] logical_cores={logical} | detected_blas_threads={blas_threads} | n_jobs={self.n_jobs}")

        for it in range(self.max_iter):
            t0 = time.time()
            if self.verbose:
                print(f"[ALS-PAR] Iter {it+1}/{self.max_iter} ...")

            # parallel subject updates
            results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(_update_subject_block)(
                    i, Ylag[i], Ycur[i], Zcur[i],
                    f, g, h,
                    D[i], C[i], Lambda[i],
                    self.lam_A, self.lam_B, self.lam_L,
                )
                for i in range(N)
            )

            for (i, Di, Ci, Li) in results:
                D[i] = Di
                C[i] = Ci
                Lambda[i] = Li

            # update f_t (serial, stable)
            # Stack across subjects: resid_it ≈ Lambda_i f_t
            for t in range(T1):
                X_blocks = []
                y_blocks = []
                for i in range(N):
                    # compute A_t ylag + B_t z
                    A_t = np.zeros((G, G), dtype=float)
                    for m in range(self.rh):
                        A_t += h[t, m] * D[i][m]
                    B_t = np.zeros((G, self.p), dtype=float)
                    for m in range(self.rg):
                        B_t += g[t, m] * C[i][m]

                    resid = Ycur[i][t] - (A_t @ Ylag[i][t] + B_t @ Zcur[i][t])

                    # resid (G,) ≈ Lambda_i (G,rf) f_t (rf,)
                    X_blocks.append(Lambda[i])        # (G,rf)
                    y_blocks.append(resid.reshape(-1, 1))  # (G,1)

                X = np.vstack(X_blocks)              # (N*G, rf)
                y = np.vstack(y_blocks)              # (N*G, 1)
                f[t] = ridge_solve(X, y, self.lam_f).ravel()

            if self.verbose:
                dt = time.time() - t0
                print(f"[ALS-PAR] Iter {it+1} done in {dt:.2f}s")

        self.D = D
        self.C = C
        self.Lambda = Lambda
        self.f = f
        self.g = g
        self.h = h
        return self

    def predict_one_step(self, Y_list: list[np.ndarray], Z_list: list[np.ndarray], train_frac: float = 0.7):
        N = len(Y_list)
        T, G = Y_list[0].shape
        split = int(train_frac * (T - 1))

        Ytrue, Ypred = [], []
        for i in range(N):
            for t in range(split, T - 1):
                A_t = np.zeros((G, G), dtype=float)
                for m in range(self.rh):
                    A_t += self.h[t, m] * self.D[i][m]
                B_t = np.zeros((G, self.p), dtype=float)
                for m in range(self.rg):
                    B_t += self.g[t, m] * self.C[i][m]

                pred = A_t @ Y_list[i][t] + B_t @ Z_list[i][t + 1] + self.Lambda[i] @ self.f[t]
                Ytrue.append(Y_list[i][t + 1])
                Ypred.append(pred)

        Ytrue = np.vstack(Ytrue)
        Ypred = np.vstack(Ypred)

        mse = float(np.mean((Ytrue - Ypred) ** 2))
        rmse = float(np.sqrt(mse))
        ss_res = float(np.sum((Ytrue - Ypred) ** 2))
        ss_tot = float(np.sum((Ytrue - Ytrue.mean(axis=0)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        return {"rmse": rmse, "mse": mse, "r2": r2, "Ytrue": Ytrue, "Ypred": Ypred}
