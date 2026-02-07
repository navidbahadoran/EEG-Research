# config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class Paths:
    """Project paths. Put your data under ./data."""
    project_root: Path
    data_dir: Path
    vmf_npz_dir: Path
    eeg_raw_dir: Path

    @staticmethod
    def default(project_root: Optional[Path] = None) -> "Paths":
        root = project_root or Path(__file__).resolve().parent
        data = root / "data"
        return Paths(
            project_root=root,
            data_dir=data,
            vmf_npz_dir=data / "vmf_npz",
            eeg_raw_dir=data / "eeg_raw",
        )

@dataclass
class ModelConfig:
    """Core model configuration."""
    # VAR lag order for EEG model
    L: int = 1

    # Dimensions for latent factor blocks (used later)
    r_f: int = 2  # interactive effects factors f_t
    r_g: int = 1  # slope factors g_t
    r_h: int = 1  # propagation factors h_t

    # Regularization for ridge baselines
    ridge: float = 1e-2

