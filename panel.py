from __future__ import annotations
import numpy as np
from typing import List

def build_response_logpower(X: np.ndarray) -> np.ndarray:
    """Return y_it = log(X^2 + eps) transposed to (N=C, T)."""
    return np.log(np.square(X) + 1e-6).T

def build_A_block(T: int, sex_male1: float, age_years: float, task_rest1: float) -> np.ndarray:
    age_centered = age_years - 30.0
    return np.column_stack([
        np.ones(T),
        np.full(T, sex_male1),
        np.full(T, age_centered),
        np.full(T, task_rest1),
    ])

def session_to_tod_hours(session_name: str) -> float:
    n = session_name.lower()
    if "morning" in n: return 10.0
    if "afternoon" in n: return 15.0
    return 12.0

def build_B_block_time_of_day(sessions: List[str]) -> tuple[np.ndarray, np.ndarray]:
    """B_t: [sin, cos] of ToD; M_t: same shape mask (1=observed)."""
    omega = 2 * np.pi / 24.0
    tau = np.array([session_to_tod_hours(s) for s in sessions])
    B = np.column_stack([np.sin(omega * tau), np.cos(omega * tau)])
    M = np.ones_like(B)
    return B, M

def build_Z_block(posteriors: np.ndarray) -> np.ndarray:
    return posteriors
