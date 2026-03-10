from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _safe_normalize_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, eps, None)
    return P / row_sums


def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def _ridge_solve(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve min_B ||Y - X B||^2 + lam ||B||^2
    X: (n, q)
    Y: (n, d)
    returns B: (q, d)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    q = X.shape[1]
    XtX = X.T @ X
    XtY = X.T @ Y
    return np.linalg.solve(XtX + lam * np.eye(q), XtY)


def _weighted_diag_var(resid: np.ndarray, w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    resid: (n, d)
    w: (n,)
    returns diagonal variance: (d,)
    """
    resid = np.asarray(resid, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    w = np.clip(w, 1e-12, None)
    w = w / np.sum(w)
    var = np.sum((resid ** 2) * w[:, None], axis=0)
    return np.clip(var, eps, None)


@dataclass
class UnitSwitchingPVARResult:
    metrics: Dict[str, float]
    y_true_oof: np.ndarray
    y_pred_oof: np.ndarray
    regime_prob_pred: np.ndarray
    regime_prob_filt: np.ndarray
    W: np.ndarray
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    sigma2: np.ndarray
    Pi: np.ndarray
    pi0: np.ndarray
    loss_history: np.ndarray
    train_end: int


class UnitSwitchingPVAR:
    """
    Unit-specific switching PVAR with shared regime-specific coefficients.

    Model:
        y_{i,t} = c_{z_{i,t}} + A_{z_{i,t}} y_{i,t-1} + B_{z_{i,t}} x_{i,t} + u_{i,t}

    where:
        z_{i,t} is unit-specific and follows a Markov chain with shared transition matrix Pi.

    Input:
        Y_list[j]: (T_j, d)
        X_list[j]: (T_j, p)

    Internal aligned tensors after trimming to common length:
        Y:    (N, T, d)
        X:    (N, T, p)
        Ylag: (N, T-1, d)
        Ycur: (N, T-1, d)
        Xcur: (N, T-1, p)

    EM latent objects:
        gamma: (N, T-1, K)
        xi:    (N, T-2, K, K)
    """

    def __init__(
        self,
        K: int = 3,
        ridge: float = 1e-4,
        max_iter: int = 25,
        tol: float = 1e-4,
        random_state: int = 0,
    ) -> None:
        self.K = int(K)
        self.ridge = float(ridge)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.rng_ = np.random.default_rng(self.random_state)

        self.c_: Optional[np.ndarray] = None       # (K, d)
        self.A_: Optional[np.ndarray] = None       # (K, d, d)
        self.B_: Optional[np.ndarray] = None       # (K, p, d)
        self.sigma2_: Optional[np.ndarray] = None  # (K, d)
        self.Pi_: Optional[np.ndarray] = None      # (K, K)
        self.pi0_: Optional[np.ndarray] = None     # (K,)
        self.loss_history_: List[float] = []

        self.d_: Optional[int] = None
        self.p_: Optional[int] = None
        self.N_: Optional[int] = None
        self.T_: Optional[int] = None

    # ------------------------------------------------------------------
    # data utilities
    # ------------------------------------------------------------------
    def _stack_to_common_panel(
        self,
        Y_list: List[np.ndarray],
        X_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(Y_list) != len(X_list):
            raise ValueError("Y_list and X_list must have same length.")
        if len(Y_list) == 0:
            raise ValueError("Empty Y_list.")

        N = len(Y_list)
        T_common = min(y.shape[0] for y in Y_list)
        d = Y_list[0].shape[1]
        p = X_list[0].shape[1]

        Y = np.zeros((N, T_common, d), dtype=float)
        X = np.zeros((N, T_common, p), dtype=float)

        for i, (y, x) in enumerate(zip(Y_list, X_list)):
            Y[i] = np.asarray(y[:T_common], dtype=float)
            X[i] = np.asarray(x[:T_common], dtype=float)

        return Y, X

    def _prepare_tensors(
        self,
        Y_list: List[np.ndarray],
        X_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Y, X = self._stack_to_common_panel(Y_list, X_list)
        Ylag = Y[:, :-1, :]
        Ycur = Y[:, 1:, :]
        Xcur = X[:, 1:, :]
        return Y, X, Ylag, Ycur, Xcur

    def _build_design(self, Ylag: np.ndarray, Xcur: np.ndarray) -> np.ndarray:
        """
        Returns Z with shape (N, Tm, 1+d+p)
        """
        N, Tm, d = Ylag.shape
        intercept = np.ones((N, Tm, 1), dtype=float)
        return np.concatenate([intercept, Ylag, Xcur], axis=2)

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------
    def _initialize_params(self, Ylag: np.ndarray, Ycur: np.ndarray, Xcur: np.ndarray) -> None:
        N, Tm, d = Ycur.shape
        p = Xcur.shape[2]

        self.N_ = N
        self.T_ = Tm
        self.d_ = d
        self.p_ = p

        self.c_ = np.zeros((self.K, d))
        self.A_ = np.zeros((self.K, d, d))
        self.B_ = np.zeros((self.K, p, d))
        self.sigma2_ = np.ones((self.K, d), dtype=float) * 0.05

        self.Pi_ = np.full((self.K, self.K), 1.0 / self.K)
        self.Pi_ += 0.25 * np.eye(self.K)
        self.Pi_ = _safe_normalize_rows(self.Pi_)
        self.pi0_ = np.full(self.K, 1.0 / self.K)

        # pooled initialization
        Z = self._build_design(Ylag, Xcur)
        Z2 = Z.reshape(-1, Z.shape[-1])
        Y2 = Ycur.reshape(-1, d)
        W_pooled = _ridge_solve(Z2, Y2, self.ridge)

        self.c_[:] = W_pooled[0:1, :]
        A0 = W_pooled[1:1 + d, :]
        B0 = W_pooled[1 + d:, :]

        for k in range(self.K):
            self.A_[k] = A0.T + 0.01 * self.rng_.standard_normal((d, d))
            self.B_[k] = B0 + 0.01 * self.rng_.standard_normal((p, d))

    # ------------------------------------------------------------------
    # mean / likelihood
    # ------------------------------------------------------------------
    def _compute_mu_all_regimes(
        self,
        Ylag: np.ndarray,
        Xcur: np.ndarray,
    ) -> np.ndarray:
        """
        returns:
            mu: (K, N, Tm, d)
        """
        assert self.c_ is not None
        assert self.A_ is not None
        assert self.B_ is not None

        N, Tm, d = Ylag.shape
        mu = np.zeros((self.K, N, Tm, d), dtype=float)

        for k in range(self.K):
            ar_term = np.einsum("ntd,ed->nte", Ylag, self.A_[k])
            cov_term = np.einsum("ntp,pe->nte", Xcur, self.B_[k])
            mu[k] = self.c_[k][None, None, :] + ar_term + cov_term

        return mu

    def _log_emission_probs_per_unit(
        self,
        Ycur: np.ndarray,
        mu: np.ndarray,
    ) -> np.ndarray:
        """
        Inputs:
            Ycur: (N, Tm, d)
            mu:   (K, N, Tm, d)

        Returns:
            loglik: (N, Tm, K)
        """
        assert self.sigma2_ is not None
        N, Tm, d = Ycur.shape
        loglik = np.zeros((N, Tm, self.K), dtype=float)

        for k in range(self.K):
            var = self.sigma2_[k][None, None, :]    # (1,1,d)
            resid = Ycur - mu[k]                    # (N,Tm,d)
            ll = -0.5 * (
                np.log(2.0 * np.pi * var) +
                (resid ** 2) / var
            )
            loglik[:, :, k] = np.sum(ll, axis=2)

        return loglik

    # ------------------------------------------------------------------
    # forward-backward per unit
    # ------------------------------------------------------------------
    def _forward_backward_single(
        self,
        loglik_tk: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        loglik_tk: (Tm, K)

        Returns:
            gamma: (Tm, K)
            xi:    (Tm-1, K, K)
            nll:   scalar negative log-likelihood
        """
        if self.Pi_ is None:
            raise RuntimeError("Pi_ is None inside _forward_backward_single.")
        if self.pi0_ is None:
            raise RuntimeError("pi0_ is None inside _forward_backward_single.")

        Tm, K = loglik_tk.shape
        logPi = np.log(np.clip(self.Pi_, 1e-12, None))
        logpi0 = np.log(np.clip(self.pi0_, 1e-12, None))

        alpha = np.zeros((Tm, K), dtype=float)
        beta = np.zeros((Tm, K), dtype=float)
        scales = np.zeros(Tm, dtype=float)

        alpha[0] = logpi0 + loglik_tk[0]
        scales[0] = float(_logsumexp(alpha[0], axis=0))
        alpha[0] -= scales[0]

        for t in range(1, Tm):
            tmp = alpha[t - 1][:, None] + logPi
            alpha[t] = loglik_tk[t] + _logsumexp(tmp, axis=0)
            scales[t] = float(_logsumexp(alpha[t], axis=0))
            alpha[t] -= scales[t]

        beta[-1] = 0.0
        for t in range(Tm - 2, -1, -1):
            tmp = logPi + loglik_tk[t + 1][None, :] + beta[t + 1][None, :]
            beta[t] = _logsumexp(tmp, axis=1) - scales[t + 1]

        log_gamma = alpha + beta
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        xi = np.zeros((Tm - 1, K, K), dtype=float)
        for t in range(Tm - 1):
            log_xi_t = (
                alpha[t][:, None]
                + logPi
                + loglik_tk[t + 1][None, :]
                + beta[t + 1][None, :]
            )
            log_xi_t -= float(_logsumexp(log_xi_t.reshape(-1), axis=0))
            xi[t] = np.exp(log_xi_t)

        nll = -np.sum(scales)
        return gamma, xi, float(nll)

    def _forward_backward_all_units(
        self,
        loglik_ntk: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        loglik_ntk: (N, Tm, K)

        Returns:
            gamma: (N, Tm, K)
            xi:    (N, Tm-1, K, K)
            total_nll: scalar
        """
        N, Tm, K = loglik_ntk.shape
        gamma = np.zeros((N, Tm, K), dtype=float)
        xi = np.zeros((N, Tm - 1, K, K), dtype=float)
        total_nll = 0.0

        for i in range(N):
            gi, xii, nlli = self._forward_backward_single(loglik_ntk[i])
            gamma[i] = gi
            xi[i] = xii
            total_nll += nlli

        return gamma, xi, float(total_nll)

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------
    def _update_markov(self, gamma: np.ndarray, xi: np.ndarray) -> None:
        """
        gamma: (N, Tm, K)
        xi:    (N, Tm-1, K, K)
        """
        self.pi0_ = np.mean(gamma[:, 0, :], axis=0)
        self.pi0_ = self.pi0_ / np.sum(self.pi0_)

        Pi_num = np.sum(xi, axis=(0, 1))
        self.Pi_ = _safe_normalize_rows(Pi_num)

    def _update_regression_params(
        self,
        Ylag: np.ndarray,
        Ycur: np.ndarray,
        Xcur: np.ndarray,
        gamma: np.ndarray,
    ) -> None:
        """
        gamma: (N, Tm, K)
        """
        assert self.c_ is not None
        assert self.A_ is not None
        assert self.B_ is not None
        assert self.sigma2_ is not None

        N, Tm, d = Ycur.shape
        p = Xcur.shape[2]

        Z = self._build_design(Ylag, Xcur)             # (N,Tm,1+d+p)
        Z2 = Z.reshape(-1, 1 + d + p)
        Y2 = Ycur.reshape(-1, d)

        for k in range(self.K):
            w = gamma[:, :, k].reshape(-1)             # (N*Tm,)
            w = np.clip(w, 1e-12, None)
            sw = np.sqrt(w)[:, None]

            Zw = Z2 * sw
            Yw = Y2 * sw

            Wk = _ridge_solve(Zw, Yw, self.ridge)

            self.c_[k] = Wk[0]
            self.A_[k] = Wk[1:1 + d].T
            self.B_[k] = Wk[1 + d:]

            mu_k = (
                self.c_[k][None, None, :]
                + np.einsum("ntd,ed->nte", Ylag, self.A_[k])
                + np.einsum("ntp,pe->nte", Xcur, self.B_[k])
            )

            resid_k = (Ycur - mu_k).reshape(-1, d)
            self.sigma2_[k] = _weighted_diag_var(resid_k, w)

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------
    def fit(self, Y_list: List[np.ndarray], X_list: List[np.ndarray]) -> "UnitSwitchingPVAR":
        _, _, Ylag, Ycur, Xcur = self._prepare_tensors(Y_list, X_list)
        self._initialize_params(Ylag, Ycur, Xcur)
        self.loss_history_ = []

        prev_nll = np.inf

        for _ in range(self.max_iter):
            mu = self._compute_mu_all_regimes(Ylag, Xcur)             # (K,N,Tm,d)
            loglik_ntk = self._log_emission_probs_per_unit(Ycur, mu)  # (N,Tm,K)
            gamma, xi, nll = self._forward_backward_all_units(loglik_ntk)

            self._update_markov(gamma, xi)
            self._update_regression_params(Ylag, Ycur, Xcur, gamma)

            self.loss_history_.append(float(nll))

            if abs(prev_nll - nll) < self.tol:
                break
            prev_nll = nll

        return self

    def forecast_and_score(
        self,
        Y_list: List[np.ndarray],
        X_list: List[np.ndarray],
        train_frac: float = 0.7,
    ) -> UnitSwitchingPVARResult:
        Y, X = self._stack_to_common_panel(Y_list, X_list)
        N, T, d = Y.shape
        train_end = max(3, int(train_frac * T))

        # fit only on training sample
        Y_train = [y[:train_end] for y in Y_list]
        X_train = [x[:train_end] for x in X_list]
        self.fit(Y_train, X_train)

        # full aligned tensors
        _, _, Ylag_full, Ycur_full, Xcur_full = self._prepare_tensors(Y_list, X_list)

        start_eval = train_end - 1
        Ylag_oof = Ylag_full[:, start_eval:, :]
        Ycur_oof = Ycur_full[:, start_eval:, :]
        Xcur_oof = Xcur_full[:, start_eval:, :]

        mu_oof = self._compute_mu_all_regimes(Ylag_oof, Xcur_oof)
        loglik_ntk = self._log_emission_probs_per_unit(Ycur_oof, mu_oof)
        gamma_oof, _, _ = self._forward_backward_all_units(loglik_ntk)

        yhat_oof = np.zeros_like(Ycur_oof)
        for k in range(self.K):
            yhat_oof += gamma_oof[:, :, k][:, :, None] * mu_oof[k]

        y_true = Ycur_oof.reshape(-1, d)
        y_pred = yhat_oof.reshape(-1, d)

        y_pred_clip = np.clip(y_pred, 1e-8, None)
        y_pred_clip /= np.sum(y_pred_clip, axis=1, keepdims=True)

        y_true_clip = np.clip(y_true, 1e-8, None)
        y_true_clip /= np.sum(y_true_clip, axis=1, keepdims=True)

        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))

        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

        true_state = np.argmax(y_true, axis=1)
        pred_state = np.argmax(y_pred, axis=1)
        acc = float(np.mean(true_state == pred_state))

        kl = float(np.mean(np.sum(y_true_clip * (np.log(y_true_clip) - np.log(y_pred_clip)), axis=1)))
        cross_entropy = float(-np.mean(np.sum(y_true_clip * np.log(y_pred_clip), axis=1)))

        W = np.zeros((self.K, 1 + d + self.p_, d), dtype=float)
        for k in range(self.K):
            W[k, 0, :] = self.c_[k]
            W[k, 1:1 + d, :] = self.A_[k].T
            W[k, 1 + d:, :] = self.B_[k]

        return UnitSwitchingPVARResult(
            metrics={
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "accuracy": acc,
                "kl": kl,
                "cross_entropy": cross_entropy,
            },
            y_true_oof=y_true,
            y_pred_oof=y_pred,
            regime_prob_pred=gamma_oof,
            regime_prob_filt=gamma_oof,
            W=W,
            A=self.A_.copy(),
            B=self.B_.copy(),
            c=self.c_.copy(),
            sigma2=self.sigma2_.copy(),
            Pi=self.Pi_.copy(),
            pi0=self.pi0_.copy(),
            loss_history=np.asarray(self.loss_history_, dtype=float),
            train_end=train_end,
        )