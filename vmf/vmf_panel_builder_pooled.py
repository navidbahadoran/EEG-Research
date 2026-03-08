from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


def _entropy_row(z: np.ndarray) -> float:
    z = np.clip(z, EPS, 1.0)
    z = z / np.sum(z)
    return float(-np.sum(z * np.log(z)))


def _encode_sex(sex) -> float:
    if isinstance(sex, str):
        s = sex.strip().lower()
        if s in ("m", "male", "1"):
            return 1.0
        if s in ("f", "female", "0"):
            return 0.0
        return np.nan
    if pd.notna(sex):
        try:
            return float(sex)
        except Exception:
            return np.nan
    return np.nan


_VMF_FILE_RE = re.compile(
    r"^sub-(?P<subject>[^_]+)_(?P<task>[^_]+)_probabilities$",
    re.IGNORECASE,
)


def _infer_subject_task_from_filename(path_like: str) -> tuple[str | None, str | None]:
    stem = Path(str(path_like)).stem
    m = _VMF_FILE_RE.match(stem)
    if not m:
        return None, None
    return m.group("subject"), m.group("task")


def load_vmf_npz(npz_path: str) -> np.ndarray:
    d = np.load(npz_path, allow_pickle=True)
    key = None
    for cand in ("P", "probabilities", "Z"):
        if cand in d:
            key = cand
            break
    if key is None:
        raise KeyError(f"No probability array found in {npz_path}. Expected one of: P, probabilities, Z")

    P = np.asarray(d[key], dtype=float)
    if P.ndim != 2:
        raise ValueError(f"Probability array must be 2D, got {P.shape} in {npz_path}")

    row_sums = np.sum(P, axis=1, keepdims=True)
    row_sums = np.where(row_sums <= EPS, 1.0, row_sums)
    P = np.clip(P, EPS, None)
    P = P / row_sums
    return P


def build_time_varying_covariates(
    Z: np.ndarray,
    summary_window: int = 25,
) -> tuple[np.ndarray, list[str]]:
    """
    Build causal covariates from vMF probabilities Z_t.

    Features at time t use only information up to and including t.

    Returns
    -------
    X_tv : (T, p_tv)
    feature_names : list[str]

    Included features:
      - entropy_t
      - occupancy_roll_k over last window
      - switching_rate_roll over last window
      - volatility_roll over last window
      - transition_roll_{a}_{b} over last window
    """
    T, K = Z.shape
    state = np.argmax(Z, axis=1).astype(int)

    entropy = np.array([_entropy_row(Z[t]) for t in range(T)], dtype=float)

    step_vol = np.zeros(T, dtype=float)
    switch_ind = np.zeros(T, dtype=float)
    step_trans = np.zeros((T, K * K), dtype=float)

    for t in range(1, T):
        step_vol[t] = float(np.linalg.norm(Z[t] - Z[t - 1], ord=1))
        switch_ind[t] = 1.0 if state[t] != state[t - 1] else 0.0
        step_trans[t, state[t - 1] * K + state[t]] = 1.0

    occupancy_roll = np.zeros((T, K), dtype=float)
    switching_rate_roll = np.zeros(T, dtype=float)
    volatility_roll = np.zeros(T, dtype=float)
    transition_roll = np.zeros((T, K * K), dtype=float)

    for t in range(T):
        start = max(0, t - summary_window + 1)

        states_win = state[start : t + 1]
        sw_win = switch_ind[start : t + 1]
        vol_win = step_vol[start : t + 1]
        trans_win = step_trans[start : t + 1]

        for k in range(K):
            occupancy_roll[t, k] = np.mean(states_win == k)

        switching_rate_roll[t] = float(np.mean(sw_win))
        volatility_roll[t] = float(np.mean(vol_win))

        denom = np.sum(trans_win)
        if denom > 0:
            transition_roll[t] = np.sum(trans_win, axis=0) / denom

    X_tv = np.column_stack(
        [
            entropy.reshape(-1, 1),
            occupancy_roll,
            switching_rate_roll.reshape(-1, 1),
            volatility_roll.reshape(-1, 1),
            transition_roll,
        ]
    )

    feature_names = (
        ["entropy"]
        + [f"occupancy_roll_state_{k}" for k in range(K)]
        + ["switching_rate_roll"]
        + ["volatility_roll"]
        + [f"transition_roll_{a}_{b}" for a in range(K) for b in range(K)]
    )

    return X_tv, feature_names


def make_vmf_panel_pooled(
    csv_path: str,
    vmf_dir: str,
    task_col: str = "task",
    id_col: str = "subject_ID",
    npz_col: str = "probabilities_file",
    age_col: str = "age",
    sex_col: str = "sex",
    targets: tuple[str, ...] = ("attention", "p_factor"),
    stride: int = 10,
    drop_baseline_task: str | None = None,
    summary_window: int = 25,
):
    """
    Build pooled subject-task panel for vMF modeling.

    Model interpretation:
      - response series: Y_i,t = Z_i,t  (vMF probability vector)
      - covariates X_i,t are causal summaries built from Z_i,1:t
        plus age/sex/task.

    Returns
    -------
    units : list[str]
        unit id = "{subject}|{task}"
    tasks : list[str]
    Y_list : list[np.ndarray]
        each is (T_common, K)
    X_list : list[np.ndarray]
        each is (T_common, p)
    y_targets : pd.DataFrame
        indexed by unit_id with target/outcome columns
    common_T : int
    task_levels : list[str]
    baseline_task : str
    feature_names : list[str]
    unit_meta : pd.DataFrame
    """
    df = pd.read_csv(csv_path)

    if npz_col not in df.columns:
        raise ValueError(f"CSV missing npz column '{npz_col}'")

    # Infer subject/task if needed
    if id_col not in df.columns:
        inferred = df[npz_col].apply(_infer_subject_task_from_filename)
        df[id_col] = [x[0] for x in inferred]

    if task_col not in df.columns:
        inferred = df[npz_col].apply(_infer_subject_task_from_filename)
        df[task_col] = [x[1] for x in inferred]

    if task_col not in df.columns:
        raise ValueError(f"CSV missing task column '{task_col}'")
    if id_col not in df.columns or df[id_col].isna().all():
        raise ValueError(f"CSV missing subject column '{id_col}' and could not infer it from '{npz_col}'")

    df["_npz_base"] = df[npz_col].apply(lambda p: os.path.basename(str(p)))

    rows = []
    for _, r in df.iterrows():
        sub = str(r[id_col])
        task = str(r[task_col])
        npz_path = os.path.join(vmf_dir, r["_npz_base"])
        if os.path.exists(npz_path):
            rows.append((sub, task, npz_path, r))

    if not rows:
        raise RuntimeError(f"No vMF npz files found under {vmf_dir}")

    Z_raw = []
    meta = []
    for sub, task, npz_path, r in rows:
        Z = load_vmf_npz(npz_path)[::stride]
        if Z.shape[0] < 5:
            continue
        unit_id = f"{sub}|{task}"
        Z_raw.append(Z)
        meta.append((unit_id, sub, task, r, npz_path))

    if not Z_raw:
        raise RuntimeError("No usable vMF series remained after loading/downsampling")

    common_T = min(z.shape[0] for z in Z_raw)
    Z_list = [z[:common_T] for z in Z_raw]

    task_levels = sorted(list({m[2] for m in meta}))
    if not task_levels:
        raise RuntimeError("No task levels found")

    if drop_baseline_task is None or drop_baseline_task not in task_levels:
        drop_baseline_task = task_levels[0]

    task_levels_used = [t for t in task_levels if t != drop_baseline_task]
    task_to_dummy = {t: j for j, t in enumerate(task_levels_used)}

    units, tasks = [], []
    Y_list_out, X_list_out = [], []
    target_rows = []
    unit_meta_rows = []

    feature_names = None

    for (unit_id, sub, task, r, npz_path), Z in zip(meta, Z_list):
        X_tv, tv_feature_names = build_time_varying_covariates(Z, summary_window=summary_window)
        if feature_names is None:
            feature_names = tv_feature_names.copy()

        age_val = float(r[age_col]) if age_col in r.index and pd.notna(r[age_col]) else np.nan
        sex_val = _encode_sex(r[sex_col]) if sex_col in r.index else np.nan

        demo = np.repeat(np.array([[age_val, sex_val]], dtype=float), common_T, axis=0)
        demo_names = ["age", "sex"]

        task_rep = np.zeros((common_T, len(task_levels_used)), dtype=float)
        task_names = [f"task_{t}" for t in task_levels_used]
        if task in task_to_dummy:
            task_rep[:, task_to_dummy[task]] = 1.0

        X = np.hstack([X_tv, demo, task_rep])

        units.append(unit_id)
        tasks.append(task)
        Y_list_out.append(Z)
        X_list_out.append(X)

        tr = {
            "unit_id": unit_id,
            "subject": sub,
            "task": task,
            "age": age_val,
            "sex": sex_val,
        }
        for tname in targets:
            tr[tname] = float(r[tname]) if tname in r.index and pd.notna(r[tname]) else np.nan
        target_rows.append(tr)

        unit_meta_rows.append(
            {
                "unit_id": unit_id,
                "subject": sub,
                "task": task,
                "npz_path": npz_path,
                "T_common": common_T,
            }
        )

    y_targets = pd.DataFrame(target_rows).set_index("unit_id")
    unit_meta = pd.DataFrame(unit_meta_rows).set_index("unit_id")

    feature_names = feature_names + demo_names + task_names

    return (
        units,
        tasks,
        Y_list_out,
        X_list_out,
        y_targets,
        common_T,
        task_levels,
        drop_baseline_task,
        feature_names,
        unit_meta,
    )