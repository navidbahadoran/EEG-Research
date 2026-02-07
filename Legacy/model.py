# model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np

from directional import row_normalize, spherical_kmeans, estimate_vmf_params   # vMF  :contentReference[oaicite:9]{index=9}
from panel import build_response_logpower                                       # y_it :contentReference[oaicite:10]{index=10}
from ife import ife_ic_bai_ng, IFEModel                                        # IFE/IC + dataclass  :contentReference[oaicite:11]{index=11}
from impute import posterior_sex_probability, estimate_time_of_day             # posterior & ToD  :contentReference[oaicite:12]{index=12}

from design import build_A_block_with_missing, build_BZ_blocks

@dataclass
class FitReport:
    fit: IFEModel
    best_r: int
    ic_vals: Dict[int, float]
    vmf: Dict[str, np.ndarray]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    beta_table: Dict[str, float]

class EEGPanelIFEMI:
    """
    Factor-augmented panel model with EM-style imputations for missing A (sex/age) and missing B (ToD).
    - Uses your existing vMF pipeline for Z (directional features).
    - Uses your IFE + Bai–Ng IC for rank selection.
    """
    def __init__(self, K_vmf: int = 4, r_grid: List[int] = [1,2,3], random_state: int = 42):
        self.K_vmf = K_vmf
        self.r_grid = r_grid
        self.random_state = random_state
        self.fit_: Optional[IFEModel] = None
        self.beta_names_: Optional[List[str]] = None
        self.col_idx_: Optional[Dict[str, int]] = None

    # ---------- Feature builders ----------
    def _vmf_features(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        Y_dir = row_normalize(X)                                         # (T,C) unit rows  :contentReference[oaicite:13]{index=13}
        mu, labels = spherical_kmeans(Y_dir, K=self.K_vmf, iters=30, seed=self.random_state)  # :contentReference[oaicite:14]{index=14}
        kappas, post = estimate_vmf_params(Y_dir, mu, labels)            # :contentReference[oaicite:15]{index=15}
        rep = {"kappas": kappas, "counts": np.bincount(labels, minlength=self.K_vmf)}
        return post, rep

    # ---------- EM-style imputations for A (sex/age) and B (ToD) ----------
    def _impute_sex_probability(self, y_it, fit, A_t, B_t, Z_t, M_t, sex_col=1, prior_pi=0.5) -> float:
        return posterior_sex_probability(y_it, fit, A_t, B_t, Z_t, M_t, sex_col_idx_in_A=sex_col, prior_pi=prior_pi)  # :contentReference[oaicite:16]{index=16}

    def _estimate_tod(self, y_it, fit, A_t, Z_t, idx_tod_sin: int, idx_tod_cos: int, grid_minutes: int = 10) -> np.ndarray:
        # No “other B” columns in this example; pass zeros as instructed by your function’s API
        B_other = np.zeros((A_t.shape[0], 0))
        tau_hat = estimate_time_of_day(
            y_it, fit, A_t, B_t_other=B_other, Z_t=Z_t,
            beta_idx_sin=idx_tod_sin, beta_idx_cos=idx_tod_cos,
            omega=2*np.pi/24.0, grid_minutes=grid_minutes
        )  # :contentReference[oaicite:17]{index=17}
        return tau_hat

    # ---------- Fit with optional EM iterations ----------
    def fit(
        self,
        X: np.ndarray,
        sessions: List[str],
        sex_male1: Optional[float],
        age_years: Optional[float],
        task_rest1: float = 1.0,
        em_iters: int = 2,
        train_sessions: Optional[List[str]] = None,
    ) -> FitReport:
        """
        X: (T,C) raw features (e.g., EEG channels)
        sessions: list[str] per time
        sex_male1, age_years: may be None (missing), to be imputed
        """
        T, C = X.shape
        y_it = build_response_logpower(X)                                 # (N=C, T)  :contentReference[oaicite:18]{index=18}
        post, vmf_rep = self._vmf_features(X)
        B_t, M_t, Z_t = build_BZ_blocks(X, sessions, Z_posteriors=post)

        # A with subject-level missing handled via mask; EM will fill if missing
        A_t, M_A, col_idx = build_A_block_with_missing(
            T, sex_male1=sex_male1, age_years=age_years, task_rest1=task_rest1
        )
        self.col_idx_ = col_idx

        # train/test split: default = first 3 sessions train, last test
        if train_sessions is None:
            from config import SESSION_NAMES                                   # reuse your split convention  :contentReference[oaicite:19]{index=19}
            train_sessions = SESSION_NAMES[:3]
        s = np.asarray(sessions, dtype=object)
        tr_mask = np.isin(s, np.asarray(train_sessions, dtype=object))
        ts_mask = ~tr_mask

        def split_rows(arr): return arr[tr_mask], arr[ts_mask]
        A_tr, A_ts = split_rows(A_t); B_tr, B_ts = split_rows(B_t); M_tr, M_ts = split_rows(M_t); Z_tr, Z_ts = split_rows(Z_t)
        y_it_tr, y_it_ts = y_it[:, tr_mask], y_it[:, ts_mask]

        # --- EM outer loop (very light: use rank selection each time; 1–2 iterations is usually enough) ---
        fit = None
        for em in range(max(1, em_iters)):
            # M-step: fit IFE with current A/B/Z/M
            best_r, ic_vals, fits = ife_ic_bai_ng(y_it_tr, A_tr, B_tr, Z_tr, M_tr, r_grid=self.r_grid)  # :contentReference[oaicite:20]{index=20}
            fit = fits[best_r]

            # E-step: if sex missing, update to posterior mean (soft)
            if sex_male1 is None:
                p_male = self._impute_sex_probability(y_it_tr, fit, A_tr, B_tr, Z_tr, M_tr, sex_col=col_idx["sex"])
                A_tr[:, col_idx["sex"]] = p_male; A_ts[:, col_idx["sex"]] = p_male  # apply to all time points

            # E-step: if age missing, do a simple ridge-style projection (or keep prior)
            if age_years is None:
                # Simple heuristic: regress the time-average over channels on A columns to back out age_centered
                xb = (np.concatenate([A_tr, M_tr * B_tr, Z_tr], axis=1) @ fit.beta)  # (T_tr,)
                ybar = y_it_tr.mean(axis=0) - (fit.Lam @ fit.F.T).mean(axis=0)
                age_effect = np.clip(np.dot(ybar - xb, np.ones_like(xb)) / (len(xb)+1e-9), -10.0, 10.0)
                A_tr[:, col_idx["age_centered"]] = age_effect
                A_ts[:, col_idx["age_centered"]] = age_effect

            # E-step: if portions of ToD are structurally missing, optionally update ToD estimates
            # Identify ToD columns in FULL beta order: A | B | Z
            idx_tod_sin = A_tr.shape[1]           # immediately after A
            idx_tod_cos = idx_tod_sin + 1
            if np.any(M_tr[:, 0] == 0) or np.any(M_tr[:, 1] == 0):
                tau_hat_tr = self._estimate_tod(y_it_tr, fit, A_tr, Z_tr, idx_tod_sin, idx_tod_cos)
                omega = 2*np.pi/24.0
                B_tr = np.column_stack([np.sin(omega*tau_hat_tr), np.cos(omega*tau_hat_tr)])
                M_tr = np.ones_like(B_tr)
                # keep test B as originally observed unless you want OOS ToD inference:
                # tau_hat_ts = self._estimate_tod(y_it_ts, fit, A_ts, Z_ts, idx_tod_sin, idx_tod_cos)

        # Final in-sample report
        rows_tr = np.concatenate([A_tr, M_tr * B_tr, Z_tr], axis=1)
        yhat_tr = (rows_tr @ fit.beta)[None, :] + fit.Lam @ fit.F.T
        tr_resid = y_it_tr - yhat_tr
        tr_mse = float(np.mean(tr_resid**2)); tr_r2 = 1.0 - tr_mse / (float(np.var(y_it_tr)) + 1e-12)

        # Out-of-sample prediction (estimate factors on test)
        N, r = fit.Lam.shape
        G = np.linalg.pinv(fit.Lam.T @ fit.Lam) @ fit.Lam.T  # (r,N)
        rows_ts = np.concatenate([A_ts, M_ts * B_ts, Z_ts], axis=1)
        yhat_ts = np.zeros((N, rows_ts.shape[0]))
        for t in range(rows_ts.shape[0]):
            mu_t = rows_ts[t] @ fit.beta
            y_t  = y_it_ts[:, t]
            f_t  = G @ (y_t - mu_t)
            yhat_ts[:, t] = mu_t + fit.Lam @ f_t

        ts_resid = y_it_ts - yhat_ts
        ts_mse = float(np.mean(ts_resid**2)); ts_r2 = 1.0 - ts_mse / (float(np.var(y_it_ts)) + 1e-12)

        # Coeff table
        self.beta_names_ = (
            [f"A:{n}" for n in ["intercept", "sex(male=1)", "age(centered)", "task(rest)"]]
            + [f"B:ToD_{n}" for n in ["sin", "cos"]]
            + [f"Z:vMF_p{k+1}" for k in range(Z_t.shape[1])]
        )
        beta_table = dict(zip(self.beta_names_, map(float, fit.beta)))

        self.fit_ = fit
        return FitReport(
            fit=fit, best_r=fit.rank_r, ic_vals={int(k): float(v) for k, v in self._as_ic(best_r, fit, self.r_grid).items()},
            vmf=vmf_rep,
            train_metrics={"mse": tr_mse, "r2": tr_r2},
            test_metrics={"mse": ts_mse, "r2": ts_r2},
            beta_table=beta_table,
        )

    def _as_ic(self, best_r: int, fit: IFEModel, r_grid: List[int]) -> Dict[int, float]:
        # Provide a minimal IC dict for reporting compatibility
        return {fit.rank_r: 0.0}

    def predict(self, X: np.ndarray, sessions: List[str]) -> np.ndarray:
        """Convenience OOS prediction with fixed beta/Lam and per-time factor estimation."""
        assert self.fit_ is not None, "Call fit() first."
        post, _ = self._vmf_features(X)
        B_t, M_t, Z_t = build_BZ_blocks(X, sessions, Z_posteriors=post)
        T = X.shape[0]
        # For prediction, caller should pass imputed subject-level A (or reuse from training subject)
        raise NotImplementedError("Provide subject-level A (sex/age/task). This can mirror your training A.")
