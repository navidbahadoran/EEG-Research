from __future__ import annotations

from dataclasses import dataclass
import numpy as np

EPS = 1e-12


def project_rows_to_simplex(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    M = np.clip(M, EPS, None)
    row_sums = np.sum(M, axis=1, keepdims=True)
    row_sums = np.where(row_sums <= EPS, 1.0, row_sums)
    return M / row_sums


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = float(np.sum((y_true - y_pred) ** 2))
    y_bar = np.mean(y_true, axis=0, keepdims=True)
    sst = float(np.sum((y_true - y_bar) ** 2))
    if sst <= EPS:
        return np.nan
    return 1.0 - sse / sst


def score_probability_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = project_rows_to_simplex(np.asarray(y_pred, dtype=float))

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    r2 = float(safe_r2(y_true, y_pred))

    true_state = np.argmax(y_true, axis=1)
    pred_state = np.argmax(y_pred, axis=1)
    accuracy = float(np.mean(true_state == pred_state))

    p = project_rows_to_simplex(np.clip(y_true, EPS, 1.0))
    q = project_rows_to_simplex(np.clip(y_pred, EPS, 1.0))
    kl = float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))
    cross_entropy = float(-np.mean(np.sum(p * np.log(q), axis=1)))

    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'accuracy': accuracy,
        'kl': kl,
        'cross_entropy': cross_entropy,
    }


@dataclass
class SequenceData:
    X: np.ndarray  # (T1, d) augmented regressors [1, ylag, xcur]
    Y: np.ndarray  # (T1, G)


class SwitchingPVARPrototype:
    """
    Simple pooled regime-switching VARX with Gaussian emissions.

    Observation equation for regime k:
        Y_t | Z_t=k ~ N(W_k x_t, diag(sigma2_k))

    where x_t = [1, Y_{t-1}, X_t].
    The latent regime follows a common Markov chain across units.

    This is a pragmatic EM prototype for the user's next modeling step.
    """

    def __init__(
        self,
        K: int,
        G: int,
        p: int,
        ridge: float = 1.0,
        max_iter: int = 25,
        tol: float = 1e-4,
        seed: int = 123,
        verbose: bool = True,
    ):
        self.K = int(K)
        self.G = int(G)
        self.p = int(p)
        self.d = 1 + self.G + self.p
        self.ridge = float(ridge)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rng = np.random.default_rng(seed)
        self.verbose = bool(verbose)
        self.loss_history_: list[float] = []

    def _build_sequences(self, Y_list: list[np.ndarray], X_list: list[np.ndarray]) -> list[SequenceData]:
        seqs: list[SequenceData] = []
        for Y, X in zip(Y_list, X_list):
            ylag = np.asarray(Y[:-1], dtype=float)
            ycur = np.asarray(Y[1:], dtype=float)
            xcur = np.asarray(X[:-1], dtype=float)
            Xreg = np.hstack([np.ones((ylag.shape[0], 1)), ylag, xcur])
            seqs.append(SequenceData(X=Xreg, Y=ycur))
        return seqs

    def _initialize_from_kmeans_like(self, seqs: list[SequenceData]):
        X_all = np.vstack([s.X for s in seqs])
        Y_all = np.vstack([s.Y for s in seqs])
        n = Y_all.shape[0]
        assign = self.rng.integers(0, self.K, size=n)
        # Use dominant-state buckets to stabilize initialization.
        dom = np.argmax(Y_all, axis=1)
        assign = dom % self.K

        W = np.zeros((self.K, self.d, self.G), dtype=float)
        sigma2 = np.zeros((self.K, self.G), dtype=float)
        for k in range(self.K):
            mask = assign == k
            if not np.any(mask):
                mask[self.rng.integers(0, n)] = True
            W[k], sigma2[k] = self._weighted_regression(X_all, Y_all, mask.astype(float))

        Pi = np.full((self.K, self.K), 1.0 / self.K, dtype=float)
        Pi += np.eye(self.K) * 2.0
        Pi = Pi / Pi.sum(axis=1, keepdims=True)
        pi0 = np.full(self.K, 1.0 / self.K, dtype=float)
        return W, sigma2, Pi, pi0

    def _weighted_regression(self, X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = np.asarray(w, dtype=float).reshape(-1)
        w = np.clip(w, 0.0, None)
        if float(np.sum(w)) <= EPS:
            w = np.ones_like(w)
        sw = np.sqrt(w).reshape(-1, 1)
        Xw = X * sw
        Yw = Y * sw
        XtX = Xw.T @ Xw
        penalty = self.ridge * np.eye(X.shape[1])
        penalty[0, 0] = 0.0
        W = np.linalg.solve(XtX + penalty, Xw.T @ Yw)
        resid = Y - X @ W
        sigma2 = (w[:, None] * resid**2).sum(axis=0) / max(np.sum(w), EPS)
        sigma2 = np.clip(sigma2, 1e-4, None)
        return W, sigma2

    def _log_emission_prob(self, seq: SequenceData) -> np.ndarray:
        T1 = seq.Y.shape[0]
        out = np.zeros((T1, self.K), dtype=float)
        for k in range(self.K):
            mu = seq.X @ self.W_[k]
            resid = seq.Y - mu
            # diagonal Gaussian likelihood up to constants
            out[:, k] = -0.5 * np.sum(np.log(2 * np.pi * self.sigma2_[k]) + resid**2 / self.sigma2_[k], axis=1)
        return out

    def _forward_backward(self, logB: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        T1, K = logB.shape
        log_alpha = np.zeros((T1, K), dtype=float)
        c = np.zeros(T1, dtype=float)

        a0 = np.log(np.clip(self.pi0_, EPS, None)) + logB[0]
        m = np.max(a0)
        aa = np.exp(a0 - m)
        c[0] = np.sum(aa)
        log_alpha[0] = np.log(aa / c[0] + EPS)

        logPi = np.log(np.clip(self.Pi_, EPS, None))
        for t in range(1, T1):
            tmp = np.exp(log_alpha[t - 1]) @ np.exp(logPi)
            tmp = np.log(np.clip(tmp, EPS, None)) + logB[t]
            m = np.max(tmp)
            aa = np.exp(tmp - m)
            c[t] = np.sum(aa)
            log_alpha[t] = np.log(aa / c[t] + EPS)

        beta = np.zeros((T1, K), dtype=float)
        for t in range(T1 - 2, -1, -1):
            nxt = np.exp(logB[t + 1] + beta[t + 1])
            tmp = self.Pi_ @ nxt
            beta[t] = np.log(np.clip(tmp, EPS, None))
            m = np.max(beta[t])
            beta[t] -= m

        gamma_log = log_alpha + beta
        gamma_log -= np.max(gamma_log, axis=1, keepdims=True)
        gamma = np.exp(gamma_log)
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        xi_sum = np.zeros((K, K), dtype=float)
        for t in range(T1 - 1):
            la = np.exp(log_alpha[t])[:, None]
            lb = np.exp(logB[t + 1] + beta[t + 1])[None, :]
            xi = la * self.Pi_ * lb
            denom = np.sum(xi)
            if denom > 0:
                xi_sum += xi / denom

        loglik = float(np.sum(np.log(np.clip(c, EPS, None))))
        alpha = np.exp(log_alpha)
        return gamma, xi_sum, loglik, alpha

    def fit(self, Y_list: list[np.ndarray], X_list: list[np.ndarray]):
        seqs = self._build_sequences(Y_list, X_list)
        self.W_, self.sigma2_, self.Pi_, self.pi0_ = self._initialize_from_kmeans_like(seqs)

        prev_ll = -np.inf
        for it in range(self.max_iter):
            gamma_list = []
            xi_total = np.zeros((self.K, self.K), dtype=float)
            pi0_acc = np.zeros(self.K, dtype=float)
            loglik = 0.0
            alphas = []
            for seq in seqs:
                logB = self._log_emission_prob(seq)
                gamma, xi_sum, ll, alpha = self._forward_backward(logB)
                gamma_list.append(gamma)
                xi_total += xi_sum
                pi0_acc += gamma[0]
                loglik += ll
                alphas.append(alpha)

            X_all = np.vstack([s.X for s in seqs])
            Y_all = np.vstack([s.Y for s in seqs])
            Gamma_all = np.vstack(gamma_list)
            for k in range(self.K):
                self.W_[k], self.sigma2_[k] = self._weighted_regression(X_all, Y_all, Gamma_all[:, k])

            self.pi0_ = np.clip(pi0_acc / np.sum(pi0_acc), EPS, None)
            self.pi0_ /= np.sum(self.pi0_)
            self.Pi_ = np.clip(xi_total, EPS, None)
            self.Pi_ /= np.sum(self.Pi_, axis=1, keepdims=True)

            self.loss_history_.append(-loglik)
            rel = np.nan if not np.isfinite(prev_ll) else (loglik - prev_ll) / max(abs(prev_ll), 1.0)
            if self.verbose:
                print(f"[SW-PVAR] iter={it+1}/{self.max_iter} loglik={loglik:.3f} rel_improve={rel:.3e}")
            if np.isfinite(prev_ll) and rel < self.tol:
                if self.verbose:
                    print(f"[SW-PVAR] converged at iter {it+1}")
                break
            prev_ll = loglik

        self.train_last_alpha_ = alphas
        return self

    def _predict_mean(self, x_reg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        alpha = np.clip(alpha, EPS, None)
        alpha /= np.sum(alpha)
        pred = np.zeros(self.G, dtype=float)
        for k in range(self.K):
            pred += alpha[k] * (x_reg @ self.W_[k])
        return pred

    def predict_test_one_step(self, Y_list: list[np.ndarray], X_list: list[np.ndarray], train_end: int):
        y_true_parts = []
        y_pred_parts = []
        regime_prob_parts = []
        filtered_prob_parts = []

        for i, (Y, X) in enumerate(zip(Y_list, X_list)):
            alpha_prev = self.train_last_alpha_[i][-1].copy()
            for t in range(train_end - 1, Y.shape[0] - 1):
                x_reg = np.concatenate([[1.0], Y[t], X[t]])
                alpha_pred = alpha_prev @ self.Pi_
                pred = self._predict_mean(x_reg, alpha_pred)
                pred = project_rows_to_simplex(pred.reshape(1, -1)).ravel()
                y_next = Y[t + 1]

                # update filter using observed next y and next regressors if available; here use same x_reg for simplicity
                logw = np.zeros(self.K, dtype=float)
                for k in range(self.K):
                    mu = x_reg @ self.W_[k]
                    resid = y_next - mu
                    logw[k] = np.log(np.clip(alpha_pred[k], EPS, None)) - 0.5 * np.sum(
                        np.log(2 * np.pi * self.sigma2_[k]) + resid**2 / self.sigma2_[k]
                    )
                m = np.max(logw)
                w = np.exp(logw - m)
                alpha_prev = w / np.sum(w)

                y_true_parts.append(y_next)
                y_pred_parts.append(pred)
                regime_prob_parts.append(alpha_pred)
                filtered_prob_parts.append(alpha_prev)

        y_true = np.asarray(y_true_parts, dtype=float)
        y_pred = np.asarray(y_pred_parts, dtype=float)
        metrics = score_probability_forecasts(y_true, y_pred)
        return {
            'Ytrue': y_true,
            'Ypred': y_pred,
            'regime_prob_pred': np.asarray(regime_prob_parts, dtype=float),
            'regime_prob_filt': np.asarray(filtered_prob_parts, dtype=float),
            **metrics,
        }

    def forecast_and_score(self, Y_list: list[np.ndarray], X_list: list[np.ndarray], train_frac: float = 0.7):
        T = Y_list[0].shape[0]
        train_end = max(4, min(T - 1, int(np.floor(train_frac * T))))
        Y_train = [Y[:train_end].copy() for Y in Y_list]
        X_train = [X[:train_end].copy() for X in X_list]
        self.fit(Y_train, X_train)
        out = self.predict_test_one_step(Y_list, X_list, train_end)
        out['train_end'] = train_end
        out['train_frac'] = float(train_frac)
        return out
