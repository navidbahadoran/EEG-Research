# pvar_full_model.py
from __future__ import annotations
import numpy as np

def whiten_factors(F):
    T = F.shape[0]
    S = (F.T @ F) / max(T, 1)
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.clip(eigvals, 1e-12, None)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return F @ W, W

class FullPVARFactorALS:
    """
    Implements:
      y_it = A_it y_i,t-1 + B_it z_it + Lambda_i f_t + u_it
      vec(A_it) = D_i h_t
      vec(B_it) = C_i g_t
    with ALS + ridge.
    """
    def __init__(self, G, p, rf, rg, rh,
                 lam_D, lam_C, lam_L, lam_f, lam_g, lam_h,
                 max_iter=15, tol=1e-4, seed=123):
        self.G, self.p = G, p
        self.rf, self.rg, self.rh = rf, rg, rh
        self.lam_D, self.lam_C, self.lam_L = lam_D, lam_C, lam_L
        self.lam_f, self.lam_g, self.lam_h = lam_f, lam_g, lam_h
        self.max_iter, self.tol = max_iter, tol
        self.rng = np.random.default_rng(seed)

        self.D = None
        self.C = None
        self.L = None
        self.f = None
        self.g = None
        self.h = None

    def fit(self, Y_list, Z_list):
        N = len(Y_list)
        T, G = Y_list[0].shape
        p = Z_list[0].shape[1]
        assert G == self.G and p == self.p
        for i in range(N):
            assert Y_list[i].shape == (T, G)
            assert Z_list[i].shape == (T, p)

        # use transitions t=1..T-1
        Ylag_list = [Y[:-1] for Y in Y_list]   # (T-1,G)
        Ycur_list = [Y[1:]  for Y in Y_list]   # (T-1,G)
        Zcur_list = [Z[1:]  for Z in Z_list]   # (T-1,p)
        T1 = T - 1

        # init factors
        f = self.rng.standard_normal((T1, self.rf))
        g = self.rng.standard_normal((T1, self.rg))
        h = self.rng.standard_normal((T1, self.rh))
        f, _ = whiten_factors(f)
        g, _ = whiten_factors(g)
        h, _ = whiten_factors(h)

        # init loadings
        D = [self.rng.standard_normal((G*G, self.rh)) * 0.01 for _ in range(N)]
        C = [self.rng.standard_normal((G*p, self.rg)) * 0.01 for _ in range(N)]
        L = [self.rng.standard_normal((G, self.rf)) * 0.01 for _ in range(N)]

        prev_sse = np.inf

        for it in range(self.max_iter):
            # ---- Update subject-specific loadings (D_i, C_i, L_i)
            for i in range(N):
                Ycur = Ycur_list[i]
                Ylag = Ylag_list[i]
                Zcur = Zcur_list[i]

                qD = G*G*self.rh
                qC = G*p*self.rg
                qL = G*self.rf
                q = qD + qC + qL

                XTX = np.zeros((q, q))
                XTy = np.zeros((q,))

                for t in range(T1):
                    y = Ycur[t]      # (G,)
                    ylag = Ylag[t]   # (G,)
                    z = Zcur[t]      # (p,)

                    # M = I_G ⊗ ylag'
                    M = np.kron(np.eye(G), ylag.reshape(1, -1))   # (G, G^2)
                    # N = I_G ⊗ z'
                    Nmat = np.kron(np.eye(G), z.reshape(1, -1))   # (G, Gp)
                    # F = f_t' ⊗ I_G
                    Fblk = np.kron(f[t].reshape(1, -1), np.eye(G)) # (G, G*rf)

                    HD = np.kron(h[t].reshape(1, -1), M)          # (G, G^2*rh)
                    GC = np.kron(g[t].reshape(1, -1), Nmat)       # (G, Gp*rg)

                    Phi = np.concatenate([HD, GC, Fblk], axis=1)  # (G, q)

                    XTX += Phi.T @ Phi
                    XTy += Phi.T @ y

                reg = np.zeros(q)
                reg[:qD] = self.lam_D
                reg[qD:qD+qC] = self.lam_C
                reg[qD+qC:] = self.lam_L
                theta = np.linalg.solve(XTX + np.diag(reg), XTy)

                D[i] = theta[:qD].reshape((G*G, self.rh), order="F")
                C[i] = theta[qD:qD+qC].reshape((G*p, self.rg), order="F")
                L[i] = theta[qD+qC:].reshape((G, self.rf), order="F")

            # ---- Update time factors (h_t, g_t, f_t)
            for t in range(T1):
                rtot = self.rh + self.rg + self.rf
                XTX = np.zeros((rtot, rtot))
                XTy = np.zeros((rtot,))

                for i in range(N):
                    y = Ycur_list[i][t]
                    ylag = Ylag_list[i][t]
                    z = Zcur_list[i][t]

                    M = np.kron(np.eye(G), ylag.reshape(1, -1))    # (G, G^2)
                    Nmat = np.kron(np.eye(G), z.reshape(1, -1))    # (G, Gp)

                    Ah = (M @ D[i])                                 # (G, rh)
                    Bg = (Nmat @ C[i])                              # (G, rg)
                    Lf = L[i]                                       # (G, rf)

                    Phi = np.concatenate([Ah, Bg, Lf], axis=1)      # (G, rtot)

                    XTX += Phi.T @ Phi
                    XTy += Phi.T @ y

                reg = np.concatenate([
                    self.lam_h * np.ones(self.rh),
                    self.lam_g * np.ones(self.rg),
                    self.lam_f * np.ones(self.rf),
                ])
                beta = np.linalg.solve(XTX + np.diag(reg), XTy)

                h[t] = beta[:self.rh]
                g[t] = beta[self.rh:self.rh+self.rg]
                f[t] = beta[self.rh+self.rg:]

            # ---- Normalize factors and absorb into loadings
            f, Wf = whiten_factors(f)
            g, Wg = whiten_factors(g)
            h, Wh = whiten_factors(h)

            invWf = np.linalg.inv(Wf)
            invWg = np.linalg.inv(Wg)
            invWh = np.linalg.inv(Wh)
            for i in range(N):
                L[i] = L[i] @ invWf.T
                C[i] = C[i] @ invWg.T
                D[i] = D[i] @ invWh.T

            # ---- Compute SSE
            sse = 0.0
            for i in range(N):
                for t in range(T1):
                    ylag = Ylag_list[i][t]
                    z = Zcur_list[i][t]

                    M = np.kron(np.eye(G), ylag.reshape(1, -1))
                    Nmat = np.kron(np.eye(G), z.reshape(1, -1))

                    yhat = (M @ (D[i] @ h[t])) + (Nmat @ (C[i] @ g[t])) + (L[i] @ f[t])
                    err = Ycur_list[i][t] - yhat
                    sse += float(err @ err)

            rel = abs(prev_sse - sse) / max(prev_sse, 1.0)
            print(f"Iter {it+1:02d} | SSE={sse:.3e} | rel_improve={rel:.3e}")
            if rel < self.tol:
                break
            prev_sse = sse

        self.D, self.C, self.L = D, C, L
        self.f, self.g, self.h = f, g, h
        return self

    @staticmethod
    def fit_var1(F):
        X = F[:-1]
        Y = F[1:]
        Phi = np.linalg.lstsq(X, Y, rcond=None)[0]
        return Phi

    def forecast_and_score(self, Y_list, Z_list, train_frac=0.7):
        """
        Honest prediction:
          - fit model on train portion of time
          - fit VAR(1) on factors using train portion
          - forecast factors on test horizon
          - predict y_{t} using forecasted factors
        """
        N = len(Y_list)
        T, G = Y_list[0].shape
        T1 = T - 1
        split = int(np.floor(train_frac * T1))

        # fit on train slice only (need +1 for lag)
        Ytr = [Y[:split+1] for Y in Y_list]
        Ztr = [Z[:split+1] for Z in Z_list]
        self.fit(Ytr, Ztr)

        # fit factor dynamics on train (up to split-1 transitions)
        Phi_f = self.fit_var1(self.f[:split])
        Phi_g = self.fit_var1(self.g[:split])
        Phi_h = self.fit_var1(self.h[:split])

        fhat = self.f.copy()
        ghat = self.g.copy()
        hhat = self.h.copy()

        # forecast forward starting at split
        for t in range(split, T1-1):
            fhat[t+1] = fhat[t] @ Phi_f
            ghat[t+1] = ghat[t] @ Phi_g
            hhat[t+1] = hhat[t] @ Phi_h

        ytrue_all = []
        ypred_all = []

        for i in range(N):
            Y = Y_list[i]
            Z = Z_list[i]
            for t in range(split, T1):
                ylag = Y[t]
                ytrue = Y[t+1]
                zcur = Z[t+1]

                M = np.kron(np.eye(G), ylag.reshape(1, -1))
                Nmat = np.kron(np.eye(G), zcur.reshape(1, -1))

                ypred = (M @ (self.D[i] @ hhat[t])) + (Nmat @ (self.C[i] @ ghat[t])) + (self.L[i] @ fhat[t])

                ytrue_all.append(ytrue)
                ypred_all.append(ypred)

        Ytrue = np.vstack(ytrue_all)
        Ypred = np.vstack(ypred_all)

        mse = float(np.mean((Ytrue - Ypred) ** 2))
        rmse = float(np.sqrt(mse))

        ss_res = float(np.sum((Ytrue - Ypred) ** 2))
        ss_tot = float(np.sum((Ytrue - Ytrue.mean(axis=0, keepdims=True)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        return {
            "rmse": rmse,
            "mse": mse,
            "r2": r2,
            "n_test_points": int(Ytrue.shape[0]),
            "Ytrue": Ytrue,
            "Ypred": Ypred,
        }
