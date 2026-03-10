from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .unit_switching_pvar import UnitSwitchingPVAR


def project_rows_to_simplex(Y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    Y = np.clip(Y, eps, None)
    Y /= np.sum(Y, axis=1, keepdims=True)
    return Y


def score_probability_forecasts(Ytrue: np.ndarray, Ypred: np.ndarray) -> Dict[str, float]:
    Ytrue = np.asarray(Ytrue, dtype=float)
    Ypred = np.asarray(Ypred, dtype=float)

    mse = float(np.mean((Ytrue - Ypred) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((Ytrue - Ypred) ** 2))
    ss_tot = float(np.sum((Ytrue - np.mean(Ytrue, axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

    true_state = np.argmax(Ytrue, axis=1)
    pred_state = np.argmax(Ypred, axis=1)
    accuracy = float(np.mean(true_state == pred_state))

    Ytrue_clip = np.clip(Ytrue, 1e-8, None)
    Ytrue_clip /= np.sum(Ytrue_clip, axis=1, keepdims=True)

    Ypred_clip = np.clip(Ypred, 1e-8, None)
    Ypred_clip /= np.sum(Ypred_clip, axis=1, keepdims=True)

    kl = float(np.mean(np.sum(Ytrue_clip * (np.log(Ytrue_clip) - np.log(Ypred_clip)), axis=1)))
    cross_entropy = float(-np.mean(np.sum(Ytrue_clip * np.log(Ypred_clip), axis=1)))

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "accuracy": accuracy,
        "kl": kl,
        "cross_entropy": cross_entropy,
    }


@dataclass
class SLDSPanelResult:
    metrics: Dict[str, float]
    y_true_oof: np.ndarray
    y_pred_oof: np.ndarray
    x_true_oof: np.ndarray
    x_pred_oof: np.ndarray
    regime_prob_pred: np.ndarray
    regime_prob_filt: np.ndarray
    C: np.ndarray
    d: np.ndarray
    A_latent: np.ndarray
    B_latent: np.ndarray
    c_latent: np.ndarray
    sigma2: np.ndarray
    Pi: np.ndarray
    pi0: np.ndarray
    loss_history: np.ndarray
    train_end: int


class PanelSLDSPrototype:
    """
    Approximate SLDS-style panel model.

    Observation layer:
        y_{i,t} ≈ d + C x_{i,t}

    Latent dynamics:
        x_{i,t} = c_{z_{i,t}} + A_{z_{i,t}} x_{i,t-1} + B_{z_{i,t}} X_{i,t} + w_{i,t}

    where z_{i,t} is unit-specific and estimated using UnitSwitchingPVAR.
    """

    def __init__(
        self,
        latent_dim: int = 3,
        K: int = 3,
        ridge: float = 1e-4,
        max_iter: int = 25,
        tol: float = 1e-4,
        random_state: int = 0,
        verbose: bool = False,
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.K = int(K)
        self.ridge = float(ridge)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        self.C_: Optional[np.ndarray] = None
        self.d_: Optional[np.ndarray] = None
        self.msvar_: Optional[UnitSwitchingPVAR] = None
        self.xlatent_full_: Optional[List[np.ndarray]] = None

    def _fit_observation_map(self, Y_list: List[np.ndarray], train_end: int) -> None:
        Y_train = np.vstack([np.asarray(Y[:train_end], dtype=float) for Y in Y_list])
        self.d_ = Y_train.mean(axis=0)

        Yc = Y_train - self.d_
        _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
        self.C_ = Vt[: self.latent_dim].T  # (G, r)

    def _encode(self, Y: np.ndarray) -> np.ndarray:
        if self.C_ is None or self.d_ is None:
            raise RuntimeError("Observation map not fitted.")
        return (np.asarray(Y, dtype=float) - self.d_) @ self.C_

    def _decode(self, X: np.ndarray) -> np.ndarray:
        if self.C_ is None or self.d_ is None:
            raise RuntimeError("Observation map not fitted.")
        Yhat = np.asarray(X, dtype=float) @ self.C_.T + self.d_
        return project_rows_to_simplex(Yhat)

    def fit(self, Y_list: List[np.ndarray], X_list: List[np.ndarray], train_end: int) -> "PanelSLDSPrototype":
        self._fit_observation_map(Y_list, train_end)

        Xlatent_list = [self._encode(Y) for Y in Y_list]

        self.msvar_ = UnitSwitchingPVAR(
            K=self.K,
            ridge=self.ridge,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        self.msvar_.fit(
            [X[:train_end] for X in Xlatent_list],
            [Z[:train_end] for Z in X_list],
        )

        self.xlatent_full_ = Xlatent_list
        return self

    def forecast_and_score(
        self,
        Y_list: List[np.ndarray],
        X_list: List[np.ndarray],
        train_frac: float = 0.7,
    ) -> SLDSPanelResult:
        T = min(Y.shape[0] for Y in Y_list)
        train_end = max(4, min(T - 1, int(np.floor(train_frac * T))))

        self.fit(Y_list, X_list, train_end=train_end)

        if self.msvar_ is None or self.xlatent_full_ is None:
            raise RuntimeError("Model not fitted.")

        latent_result = self.msvar_.forecast_and_score(
            self.xlatent_full_,
            X_list,
            train_frac=train_frac,
        )

        x_true_oof = latent_result.y_true_oof
        x_pred_oof = latent_result.y_pred_oof

        y_true_oof = np.vstack([np.asarray(Y[train_end:], dtype=float) for Y in Y_list])
        y_pred_oof = self._decode(x_pred_oof)

        metrics = score_probability_forecasts(y_true_oof, y_pred_oof)

        return SLDSPanelResult(
            metrics=metrics,
            y_true_oof=y_true_oof,
            y_pred_oof=y_pred_oof,
            x_true_oof=x_true_oof,
            x_pred_oof=x_pred_oof,
            regime_prob_pred=latent_result.regime_prob_pred,
            regime_prob_filt=latent_result.regime_prob_filt,
            C=self.C_.copy(),
            d=self.d_.copy(),
            A_latent=latent_result.A.copy(),
            B_latent=latent_result.B.copy(),
            c_latent=latent_result.c.copy(),
            sigma2=latent_result.sigma2.copy(),
            Pi=latent_result.Pi.copy(),
            pi0=latent_result.pi0.copy(),
            loss_history=latent_result.loss_history.copy(),
            train_end=train_end,
        )