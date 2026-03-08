# vmf_panel_builder.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd

EPS = 1e-12

def _entropy_row(z: np.ndarray) -> float:
    z = np.clip(z, EPS, 1.0)
    return float(-np.sum(z * np.log(z)))

def build_time_varying_covariates(Z: np.ndarray) -> np.ndarray:
    """
    Z: (T, K) vMF probabilities
    Returns X: (T, p) time-varying covariates computed causally from Z up to time t.
    """
    T, K = Z.shape

    # Instant features
    ent = np.array([_entropy_row(Z[t]) for t in range(T)], dtype=float)              # (T,)
    conf = np.max(Z, axis=1).astype(float)                                           # (T,)
    state = np.argmax(Z, axis=1).astype(int)                                         # (T,)

    # Volatility and switching (defined from t>=1; set t=0 to 0)
    dZ = np.zeros(T, dtype=float)
    sw = np.zeros(T, dtype=float)
    for t in range(1, T):
        dZ[t] = float(np.linalg.norm(Z[t] - Z[t-1], ord=1))
        sw[t] = 1.0 if state[t] != state[t-1] else 0.0

    # Optional: transition one-hot for (s_{t-1}, s_t)
    # There are K*K possible transitions.
    trans = np.zeros((T, K*K), dtype=float)
    for t in range(1, T):
        trans[t, state[t-1]*K + state[t]] = 1.0

    # Stack covariates
    X = np.column_stack([
        ent, conf, dZ, sw, trans
    ])  # shape (T, 4 + K*K)

    return X

def load_vmf_npz(npz_path: str) -> np.ndarray:
    d = np.load(npz_path, allow_pickle=True)
    P = d["P"]
    # P is (T, K)
    if P.ndim != 2:
        raise ValueError(f"P must be 2D, got {P.shape} in {npz_path}")
    return P.astype(float)

def make_vmf_panel_from_csv(
    csv_path: str,
    vmf_dir: str,
    task: str,
    id_col: str = "subject",
    task_col: str = "task",
    npz_col: str = "probabilities_file",
    age_col: str = "age",
    sex_col: str = "sex",
    targets: tuple[str, ...] = ("attention", "p_factor"),
    stride: int = 10,
):
    """
    Returns:
      subjects: list[str]
      Y_list: list[np.ndarray] each (T_common, K) where Y=Z
      X_list: list[np.ndarray] each (T_common, p) time-varying + repeated demographics
      y_targets: pd.DataFrame indexed by subject with columns targets
    """
    df = pd.read_csv(csv_path)

    # filter this task
    if task_col in df.columns:
        df = df[df[task_col].astype(str) == str(task)].copy()

    # find each subject's npz filename
    def _basename(p):
        return os.path.basename(str(p))

    df["_npz_base"] = df[npz_col].apply(_basename)

    # Load each subject Z
    rows = []
    for _, r in df.iterrows():
        sub = str(r[id_col]) if id_col in df.columns else None
        npz_base = r["_npz_base"]
        npz_path = os.path.join(vmf_dir, npz_base)
        if not os.path.exists(npz_path):
            continue
        rows.append((sub, npz_path, r))
    if not rows:
        raise RuntimeError(f"No vMF npz files found for task={task} in {vmf_dir}")

    # Load Z (with stride to reduce T)
    Z_list = []
    meta = []
    for sub, npz_path, r in rows:
        Z = load_vmf_npz(npz_path)
        Z = Z[::stride]  # downsample windows to speed up
        Z_list.append(Z)
        meta.append((sub, r))

    # Align to common T
    common_T = min(z.shape[0] for z in Z_list)
    Z_list = [z[:common_T] for z in Z_list]

    # Build time-varying covariates from Z + add demographics repeated over time
    Y_list, X_list, subjects = [], [], []
    target_rows = []
    for (sub, r), Z in zip(meta, Z_list):
        X_tv = build_time_varying_covariates(Z)  # (T, p_tv)
        # demographics (repeat)
        age = float(r[age_col]) if age_col in r.index and pd.notna(r[age_col]) else np.nan
        sex = r[sex_col] if sex_col in r.index else np.nan

        # Encode sex safely: map to {0,1}, unknown -> nan
        sex_val = np.nan
        if isinstance(sex, str):
            s = sex.strip().lower()
            if s in ("m", "male", "1"):
                sex_val = 1.0
            elif s in ("f", "female", "0"):
                sex_val = 0.0
        elif pd.notna(sex):
            try:
                sex_val = float(sex)
            except Exception:
                sex_val = np.nan

        demo = np.array([age, sex_val], dtype=float)
        demo_rep = np.repeat(demo[None, :], common_T, axis=0)

        X = np.hstack([X_tv, demo_rep])  # (T, p_tv + 2)

        subjects.append(sub)
        Y_list.append(Z)     # outcome is Z
        X_list.append(X)

        # targets
        tr = {"subject": sub}
        for tname in targets:
            tr[tname] = float(r[tname]) if tname in r.index and pd.notna(r[tname]) else np.nan
        target_rows.append(tr)

    y_targets = pd.DataFrame(target_rows).set_index("subject")
    return subjects, Y_list, X_list, y_targets, common_T