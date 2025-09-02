from __future__ import annotations
import numpy as np
from scipy.special import iv

def row_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / norms

def spherical_kmeans(Y: np.ndarray, K: int, iters: int = 30, seed: int = 0):
    """Spherical k-means (cosine). Y: (T,C) unit rows. Returns (mu[K,C], labels[T])."""
    rng = np.random.default_rng(seed)
    T, C = Y.shape
    mu = np.zeros((K, C))

    # k-means++-ish init
    mu[0] = Y[rng.integers(0, T)]
    for k in range(1, K):
        sims = np.max(Y @ mu[:k].T, axis=1)
        probs = np.maximum(1 - sims, 0)
        probs = probs / probs.sum() if probs.sum() > 0 else None
        mu[k] = Y[rng.choice(T, p=probs)] if probs is not None else Y[rng.integers(0, T)]

    labels = np.zeros(T, dtype=int)
    for _ in range(iters):
        sims = Y @ mu.T
        labels = sims.argmax(axis=1)
        for k in range(K):
            sel = Y[labels == k]
            if len(sel) == 0:
                mu[k] = Y[rng.integers(0, T)]
            else:
                v = sel.mean(axis=0)
                mu[k] = v / (np.linalg.norm(v) + 1e-12)
    return mu, labels


# vMF helpers
def A_p(kappa: float, p: int) -> float:
    """Stable A_p(kappa) using piecewise approximations to avoid overflow/underflow."""
    if kappa <= 1e-6:
        # first-order series around 0
        return float(kappa / (p / 2.0))
    if kappa >= 200.0:
        # large-kappa asymptotics
        a = (p - 1.0) / (2.0 * kappa)
        b = (p - 1.0) * (p - 3.0) / (8.0 * kappa**2)
        return float(1.0 - a + b)
    # moderate range: use scaled Bessel to avoid overflow
    nu = p / 2.0 - 1.0
    num = ive(nu + 1.0, kappa)    # scaled I_{nu+1}
    den = ive(nu, kappa) + 1e-24  # scaled I_{nu}
    return float(num / den)


def invert_A_p(rbar: float, p: int) -> float:
    """Monotone bracketed solve for kappa; clip rbar and use good initial guesses."""
    rbar = float(np.clip(rbar, 1e-8, 1 - 1e-8))
    # good initial guess (Banerjee et al., 2005)
    k0 = rbar * (p - rbar**2) / (1.0 - rbar**2 + 1e-12)
    lo, hi = 1e-8, max(200.0, 2.0 * k0 + 10.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = A_p(mid, p)
        if val < rbar:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def estimate_vmf_params(Y_dir: np.ndarray, mu: np.ndarray, labels: np.ndarray):
    """Return (kappas[K], posteriors[T,K]) under equal mixture weights."""
    T, C = Y_dir.shape
    K = mu.shape[0]
    # kappas
    kappas = np.zeros(K)
    for k in range(K):
        sel = Y_dir[labels == k]
        kappas[k] = 0.0 if len(sel) == 0 else invert_A_p(float(np.linalg.norm(sel.mean(axis=0))), C)
    # log-lik
    nu = C / 2.0 - 1.0
    logC = np.where(
        kappas < 1e-8,
        - (C / 2.0) * np.log(2 * np.pi),
        nu * np.log(kappas + 1e-12) - (C / 2.0) * np.log(2 * np.pi) - np.log(iv(nu, kappas) + 1e-12)
    )
    dot = Y_dir @ mu.T
    loglik = logC[None, :] + dot * kappas[None, :]
    lse = loglik.max(axis=1, keepdims=True)
    post = np.exp(loglik - lse)
    post /= post.sum(axis=1, keepdims=True)
    return kappas, post
