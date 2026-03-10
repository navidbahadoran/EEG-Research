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
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "accuracy": accuracy,
        "kl": kl,
        "cross_entropy": cross_entropy,
    }


def _gaussian_logpdf_diag(y: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.clip(np.asarray(sigma2, dtype=float), 1e-6, None)
    return -0.5 * np.sum(np.log(2.0 * np.pi * sigma2) + (y - mu) ** 2 / sigma2, axis=-1)


def forward_backward(logB: np.ndarray, Pi: np.ndarray, pi0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    T, K = logB.shape
    Pi = np.clip(np.asarray(Pi, dtype=float), EPS, None)
    Pi = Pi / Pi.sum(axis=1, keepdims=True)
    pi0 = np.clip(np.asarray(pi0, dtype=float), EPS, None)
    pi0 = pi0 / pi0.sum()

    alpha = np.zeros((T, K), dtype=float)
    scales = np.zeros(T, dtype=float)

    a0 = pi0 * np.exp(logB[0] - np.max(logB[0]))
    scales[0] = max(np.sum(a0), EPS)
    alpha[0] = a0 / scales[0]

    for t in range(1, T):
        pred = alpha[t - 1] @ Pi
        a = pred * np.exp(logB[t] - np.max(logB[t]))
        scales[t] = max(np.sum(a), EPS)
        alpha[t] = a / scales[t]

    beta = np.zeros((T, K), dtype=float)
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        tmp = Pi @ (np.exp(logB[t + 1] - np.max(logB[t + 1])) * beta[t + 1])
        beta[t] = tmp / max(np.sum(tmp), EPS)

    gamma = alpha * beta
    gamma = gamma / np.sum(gamma, axis=1, keepdims=True)

    xi_sum = np.zeros((K, K), dtype=float)
    for t in range(T - 1):
        numer = alpha[t][:, None] * Pi * (np.exp(logB[t + 1] - np.max(logB[t + 1])) * beta[t + 1])[None, :]
        denom = max(np.sum(numer), EPS)
        xi_sum += numer / denom

    loglik = float(np.sum(np.log(scales + EPS)) + np.sum(np.max(logB, axis=1)))
    return gamma, xi_sum, loglik, alpha


def weighted_ridge(X: np.ndarray, Y: np.ndarray, w: np.ndarray, ridge: float, penalize_intercept: bool = False) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    w = np.clip(w, 0.0, None)
    if float(np.sum(w)) <= EPS:
        w = np.ones_like(w)
    sw = np.sqrt(w)[:, None]
    Xw = X * sw
    Yw = Y * sw
    XtX = Xw.T @ Xw
    P = ridge * np.eye(X.shape[1])
    if not penalize_intercept:
        P[0, 0] = 0.0
    return np.linalg.solve(XtX + P, Xw.T @ Yw)


def weighted_sigma2(X: np.ndarray, Y: np.ndarray, W: np.ndarray, w: np.ndarray) -> np.ndarray:
    resid = np.asarray(Y, dtype=float) - np.asarray(X, dtype=float) @ np.asarray(W, dtype=float)
    w = np.clip(np.asarray(w, dtype=float).reshape(-1), 0.0, None)
    denom = max(float(np.sum(w)), EPS)
    sigma2 = (w[:, None] * resid ** 2).sum(axis=0) / denom
    return np.clip(sigma2, 1e-4, None)
