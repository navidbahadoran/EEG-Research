# eeg/windowing.py
from __future__ import annotations
import numpy as np
from typing import Iterator, Tuple


def sliding_windows(x: np.ndarray, win: int, step: int) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Yield (start_index, window) where window has shape (win, G).
    x must be shape (T, G).
    """
    T = x.shape[0]
    for start in range(0, T - win + 1, step):
        yield start, x[start:start + win]
