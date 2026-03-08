from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed


EPS = 1e-12


def _env_int(name: str):
    v = os.environ.get(name)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def detect_blas_threads(default_if_unknown: int = 4) -> int:
    for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        val = _env_int(k)
        if val is not None and val > 0:
            return val
    return default_if_unknown


def auto_n_jobs(max_cap: int = 12, frac: float = 0.6, assume_blas_threads_if_unknown: int = 4) -> int:
    logical = os.cpu_count() or 4
    blas_threads = detect_blas_threads(default_if_unknown=assume_blas_threads_if_unknown)
    budget = max(1, logical // max(1, blas_threads))
    n_jobs = int(round(frac * budget))
    return max(1, min(max_cap, n_jobs))


def ridge_solve(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    XtX = X.T @ X
    XtY = X.T @ Y
    d = XtX.shape[0]
    return np.linalg.solve(XtX + lam * np.eye(d), XtY)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = float(np.sum((y_true - y_pred) ** 2))
    y_bar = np.mean(y_true, axis=0, keepdims=True)
    sst = float(np.sum((y_true - y_bar) ** 2))
    if sst <= EPS:
        return np.nan
    return 1.0 - sse / sst


def project_rows_to_simplex(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    M = np.clip(M, EPS, None)
    row_sums = np.sum(M, axis=1, keepdims=True)
    row_sums = np.where(row_sums <= EPS, 1.0, row_sums)
    return M / row_sums


@dataclass
class SubjectParams:
    D: list[np.ndarray]       # rh of (G,G)
    C: list[np.ndarray]       # rg of (G,p)
    Lambda: np.ndarray        # (G, rf)


def _compute_A_t(h_t: np.ndarray, D_i: list[np.ndarray]) -> np.ndarray:
    G = D_i[0].shape[0]
    A_t = np.zeros((G, G), dtype=float)
    for m, Dm in enumerate(D_i):
        A_t += h_t[m] * Dm
    return A_t


def _compute_B_t(g_t: np.ndarray, C_i: list[np.ndarray]) -> np.ndarray:
    G, p = C_i[0].shape
    B_t = np.zeros((G, p), dtype=float)
    for m, Cm in enumerate(C_i):
        B_t += g_t[m] * Cm
    return B_t


def _update_subject_block(
    i: int,
    Ylag_i: np.ndarray,      # (T1, G)
    Ycur_i: np.ndarray,      # (T1, G)
    Xcur_i: np.ndarray,      # (T1, p)
    f: np.ndarray,           # (T1, rf)
    g: np.ndarray,           # (T1, rg)
    h: np.ndarray,           # (T1, rh)
    params_i: SubjectParams,
    lam_A: float,
    lam_B: float,
    lam_L: float,
) -> tuple[int, SubjectParams]:
    T1, G = Ycur_i.shape
    rf = f.shape[1]
    rg = g.shape[1]
    rh = h.shape[1]

    D_i = [d.copy() for d in params_i.D]
    C_i = [c.copy() for c in params_i.C]
    Lambda_i = params_i.Lambda.copy()

    # ---- Update Lambda_i from residual excluding dynamic and covariate pieces
    R_lambda = np.zeros((T1, G), dtype=float)
    for t in range(T1):
        A_t = _compute_A_t(h[t], D_i)
        B_t = _compute_B_t(g[t], C_i)
        R_lambda[t] = Ycur_i[t] - (A_t @ Ylag_i[t] + B_t @ Xcur_i[t])

    Lambda_i = ridge_solve(f, R_lambda, lam_L).T  # (G, rf)

    # ---- Update D_i[m] using partial residuals
    for m in range(rh):
        Rm = np.zeros((T1, G), dtype=float)
        Xm = np.zeros((T1, G), dtype=float)

        for t in range(T1):
            contrib_other_D = np.zeros(G, dtype=float)
            for ell in range(rh):
                if ell == m:
                    continue
                contrib_other_D += h[t, ell] * (D_i[ell] @ Ylag_i[t])

            B_t = _compute_B_t(g[t], C_i)
            lambda_part = Lambda_i @ f[t]

            Rm[t] = Ycur_i[t] - (contrib_other_D + B_t @ Xcur_i[t] + lambda_part)
            Xm[t] = h[t, m] * Ylag_i[t]

        D_i[m] = ridge_solve(Xm, Rm, lam_A).T

    # ---- Update C_i[m] using partial residuals
    for m in range(rg):
        Rm = np.zeros((T1, G), dtype=float)
        Xm = np.zeros((T1, Xcur_i.shape[1]), dtype=float)

        for t in range(T1):
            A_t = _compute_A_t(h[t], D_i)

            contrib_other_C = np.zeros(G, dtype=float)
            for ell in range(rg):
                if ell == m:
                    continue
                contrib_other_C += g[t, ell] * (C_i[ell] @ Xcur_i[t])

            lambda_part = Lambda_i @ f[t]

            Rm[t] = Ycur_i[t] - (A_t @ Ylag_i[t] + contrib_other_C + lambda_part)
            Xm[t] = g[t, m] * Xcur_i[t]

        C_i[m] = ridge_solve(Xm, Rm, lam_B).T

    return i, SubjectParams(D=D_i, C=C_i, Lambda=Lambda_i)


class PVARFactorALSParallelAdaptive:
    """
    Factor-augmented pooled dynamic model for vMF probabilities.

    Model:
        Y_{i,t+1} = A_{i,t} Y_{i,t} + B_{i,t} X_{i,t} + Lambda_i f_t + u_{i,t+1}

    with
        A_{i,t} = sum_m h_t[m] D_i[m]
        B_{i,t} = sum_m g_t[m] C_i[m]

    Notes
    -----
    - causal alignment: predict t+1 from information at t
    - fit() expects full series and uses:
        Ylag = Y[:, :-1]
        Ycur = Y[:, 1:]
        Xcur = X[:, :-1]
    """

    def __init__(
        self,
        G: int,
        p: int,
        rf: int = 2,
        rg: int = 2,
        rh: int = 2,
        lam_A: float = 10.0,
        lam_B: float = 10.0,
        lam_L: float = 10.0,
        lam_f: float = 1.0,
        lam_g: float = 1.0,
        lam_h: float = 1.0,
        max_iter: int = 20,
        tol: float = 1e-4,
        seed: int = 123,
        n_jobs: int | None = None,
        n_jobs_cap: int = 12,
        n_jobs_frac: float = 0.6,
        verbose: bool = True,
    ):
        self.G = int(G)
        self.p = int(p)
        self.rf = int(rf)
        self.rg = int(rg)
        self.rh = int(rh)

        self.lam_A = float(lam_A)
        self.lam_B = float(lam_B)
        self.lam_L = float(lam_L)
        self.lam_f = float(lam_f)
        self.lam_g = float(lam_g)
        self.lam_h = float(lam_h)

        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rng = np.random.default_rng(seed)
        self.verbose = bool(verbose)

        if n_jobs is None:
            self.n_jobs = auto_n_jobs(max_cap=n_jobs_cap, frac=n_jobs_frac)
        else:
            self.n_jobs = int(n_jobs)

        self.loss_history_: list[float] = []

    def _initialize_subject_params(self, N: int) -> list[SubjectParams]:
        params = []
        for _ in range(N):
            D_i = [self.rng.standard_normal((self.G, self.G)) * 0.01 for _ in range(self.rh)]
            C_i = [self.rng.standard_normal((self.G, self.p)) * 0.01 for _ in range(self.rg)]
            Lambda_i = self.rng.standard_normal((self.G, self.rf)) * 0.01
            params.append(SubjectParams(D=D_i, C=C_i, Lambda=Lambda_i))
        return params

    def _compute_loss(
        self,
        Ylag: list[np.ndarray],
        Ycur: list[np.ndarray],
        Xcur: list[np.ndarray],
        params: list[SubjectParams],
        f: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
    ) -> float:
        losses = []
        T1 = Ycur[0].shape[0]

        for i in range(len(Ycur)):
            resid = np.zeros_like(Ycur[i])
            for t in range(T1):
                A_t = _compute_A_t(h[t], params[i].D)
                B_t = _compute_B_t(g[t], params[i].C)
                mu_t = A_t @ Ylag[i][t] + B_t @ Xcur[i][t] + params[i].Lambda @ f[t]
                resid[t] = Ycur[i][t] - mu_t
            losses.append(np.mean(resid**2))

        return float(np.mean(losses))

    def fit(self, Y_list: list[np.ndarray], X_list: list[np.ndarray]):
        N = len(Y_list)
        if N == 0:
            raise ValueError("Y_list is empty")

        T = Y_list[0].shape[0]
        G = Y_list[0].shape[1]
        if G != self.G:
            raise ValueError(f"G mismatch: model G={self.G}, data G={G}")

        T1 = T - 1
        if T1 < 2:
            raise ValueError("Need at least 3 time points to fit the model")

        Ylag = [np.asarray(Y[:-1], dtype=float) for Y in Y_list]
        Ycur = [np.asarray(Y[1:], dtype=float) for Y in Y_list]
        Xcur = [np.asarray(X[:-1], dtype=float) for X in X_list]

        f = self.rng.standard_normal((T1, self.rf)) * 0.05
        g = self.rng.standard_normal((T1, self.rg)) * 0.05
        h = self.rng.standard_normal((T1, self.rh)) * 0.05
        params = self._initialize_subject_params(N)

        if self.verbose:
            logical = os.cpu_count() or -1
            blas_threads = detect_blas_threads()
            print(f"[ALS-PAR] logical_cores={logical} | detected_blas_threads={blas_threads} | n_jobs={self.n_jobs}")

        prev_loss = np.inf

        for it in range(self.max_iter):
            t0 = time.time()
            if self.verbose:
                print(f"[ALS-PAR] Iter {it + 1}/{self.max_iter} ...")

            # ---- subject updates in parallel
            results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(_update_subject_block)(
                    i=i,
                    Ylag_i=Ylag[i],
                    Ycur_i=Ycur[i],
                    Xcur_i=Xcur[i],
                    f=f,
                    g=g,
                    h=h,
                    params_i=params[i],
                    lam_A=self.lam_A,
                    lam_B=self.lam_B,
                    lam_L=self.lam_L,
                )
                for i in range(N)
            )

            for i, params_i in results:
                params[i] = params_i

            # ---- update f_t
            for t in range(T1):
                X_blocks = []
                y_blocks = []
                for i in range(N):
                    A_t = _compute_A_t(h[t], params[i].D)
                    B_t = _compute_B_t(g[t], params[i].C)
                    resid = Ycur[i][t] - (A_t @ Ylag[i][t] + B_t @ Xcur[i][t])
                    X_blocks.append(params[i].Lambda)
                    y_blocks.append(resid.reshape(-1, 1))

                X_stack = np.vstack(X_blocks)
                y_stack = np.vstack(y_blocks)
                f[t] = ridge_solve(X_stack, y_stack, self.lam_f).ravel()

            # ---- update g_t
            for t in range(T1):
                W = np.zeros((N * self.G, self.rg), dtype=float)
                y_vec = np.zeros((N * self.G, 1), dtype=float)

                for i in range(N):
                    A_t = _compute_A_t(h[t], params[i].D)
                    lambda_part = params[i].Lambda @ f[t]
                    resid = Ycur[i][t] - (A_t @ Ylag[i][t] + lambda_part)

                    row0 = i * self.G
                    row1 = (i + 1) * self.G
                    y_vec[row0:row1, 0] = resid

                    for m in range(self.rg):
                        W[row0:row1, m] = params[i].C[m] @ Xcur[i][t]

                g[t] = ridge_solve(W, y_vec, self.lam_g).ravel()

            # ---- update h_t
            for t in range(T1):
                W = np.zeros((N * self.G, self.rh), dtype=float)
                y_vec = np.zeros((N * self.G, 1), dtype=float)

                for i in range(N):
                    B_t = _compute_B_t(g[t], params[i].C)
                    lambda_part = params[i].Lambda @ f[t]
                    resid = Ycur[i][t] - (B_t @ Xcur[i][t] + lambda_part)

                    row0 = i * self.G
                    row1 = (i + 1) * self.G
                    y_vec[row0:row1, 0] = resid

                    for m in range(self.rh):
                        W[row0:row1, m] = params[i].D[m] @ Ylag[i][t]

                h[t] = ridge_solve(W, y_vec, self.lam_h).ravel()

            cur_loss = self._compute_loss(Ylag, Ycur, Xcur, params, f, g, h)
            self.loss_history_.append(cur_loss)

            if np.isfinite(prev_loss):
                rel_improve = (prev_loss - cur_loss) / max(abs(prev_loss), 1.0)
            else:
                rel_improve = np.nan
            if self.verbose:
                dt = time.time() - t0
                print(f"[ALS-PAR] loss={cur_loss:.8f} | rel_improve={rel_improve:.6e} | dt={dt:.2f}s")

            if np.isfinite(prev_loss) and rel_improve < self.tol:
                if self.verbose:
                    print(f"[ALS-PAR] converged at iter {it + 1}")
                break

            prev_loss = cur_loss

        self.params_ = params
        self.f_ = f
        self.g_ = g
        self.h_ = h

        self.Lambda_ = np.stack([p.Lambda for p in params], axis=0)  # (N, G, rf)
        return self

    def _average_train_effects(self):
        h_bar = np.mean(self.h_, axis=0)
        g_bar = np.mean(self.g_, axis=0)
        f_bar = np.mean(self.f_, axis=0)
        return h_bar, g_bar, f_bar

    def predict_test_one_step(
        self,
        Y_list: list[np.ndarray],
        X_list: list[np.ndarray],
        train_end: int,
    ) -> dict[str, np.ndarray | float]:
        """
        Predict Y_{t+1} for t >= train_end-1 using frozen train-period averages.

        This is a simple, causal test-time rule:
            pred_{i,t+1} = Abar_i Y_{i,t} + Bbar_i X_{i,t} + Lambda_i fbar

        where Abar_i/Bbar_i/fbar are estimated from the training fit.
        """
        h_bar, g_bar, f_bar = self._average_train_effects()

        Ytrue = []
        Ypred = []

        for i in range(len(Y_list)):
            A_bar = _compute_A_t(h_bar, self.params_[i].D)
            B_bar = _compute_B_t(g_bar, self.params_[i].C)
            latent_bar = self.params_[i].Lambda @ f_bar

            for t in range(train_end - 1, Y_list[i].shape[0] - 1):
                pred = A_bar @ Y_list[i][t] + B_bar @ X_list[i][t] + latent_bar
                pred = project_rows_to_simplex(pred.reshape(1, -1)).ravel()

                Ypred.append(pred)
                Ytrue.append(Y_list[i][t + 1])

        Ytrue = np.asarray(Ytrue, dtype=float)
        Ypred = np.asarray(Ypred, dtype=float)

        mse = float(np.mean((Ytrue - Ypred) ** 2))
        rmse = float(np.sqrt(mse))
        r2 = float(safe_r2(Ytrue, Ypred))

        true_state = np.argmax(Ytrue, axis=1)
        pred_state = np.argmax(Ypred, axis=1)
        acc = float(np.mean(true_state == pred_state))

        p = np.clip(Ytrue, EPS, 1.0)
        q = np.clip(Ypred, EPS, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)
        q = q / np.sum(q, axis=1, keepdims=True)

        kl = float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))
        cross_entropy = float(-np.mean(np.sum(p * np.log(q), axis=1)))

        return {
            "Ytrue": Ytrue,
            "Ypred": Ypred,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "accuracy": acc,
            "kl": kl,
            "cross_entropy": cross_entropy,
        }

    def forecast_and_score(
        self,
        Y_list: list[np.ndarray],
        X_list: list[np.ndarray],
        train_frac: float = 0.7,
    ) -> dict[str, np.ndarray | float | int]:
        T = Y_list[0].shape[0]
        train_end = max(4, min(T - 1, int(np.floor(train_frac * T))))

        Y_train = [Y[:train_end].copy() for Y in Y_list]
        X_train = [X[:train_end].copy() for X in X_list]

        self.fit(Y_train, X_train)
        out = self.predict_test_one_step(Y_list, X_list, train_end=train_end)
        out["train_end"] = train_end
        out["train_frac"] = float(train_frac)
        return out