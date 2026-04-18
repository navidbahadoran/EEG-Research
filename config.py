import os
from pathlib import Path


def _default_vmf_data_dir() -> Path:
    """
    Resolve a sensible default data directory:
    1) explicit VMF_DATA_DIR env var
    2) historical local Windows path (if it exists)
    3) repo-local dev_data directory for portable smoke tests
    """
    env_value = os.getenv("VMF_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser()

    windows_default = Path(
        r"D:/Navid/FSU/OneDrive - Florida State University/FSU/Courses/2025-2026/Spring/Sahar Research/EEG Data/data_for_navid"
    )
    if windows_default.exists():
        return windows_default

    return Path(__file__).resolve().parent / "dev_data"


VMF_DATA_DIR = _default_vmf_data_dir()
RAW_EEG_DIR = Path(os.getenv("RAW_EEG_DIR", str(VMF_DATA_DIR / "raw"))).expanduser()
VMF_DIR = Path(os.getenv("VMF_DIR", str(VMF_DATA_DIR / "vMF"))).expanduser()
VMF_CSV_PATH = Path(
    os.getenv("VMF_CSV_PATH", str(VMF_DATA_DIR / "vmf_fixedmus_summary_K7.csv"))
).expanduser()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs")).expanduser()

# Data dimensions (from your example)
G_CHANNELS = 32
K_VMF = 7

# Modeling / estimation config
RANDOM_SEED = 123

# Full model ranks (start small for stability)
RF = 3   # rank of interactive effects f_t
RG = 3   # rank of covariate-slope factors g_t
RH = 3   # rank of dynamics factors h_t

# Ridge penalties (tune later)
LAM_D = 50.0      # penalty for D_i (A-loadings)
LAM_C = 50.0      # penalty for C_i (B-loadings)
LAM_L = 50.0      # penalty for Lambda_i
LAM_F = 1.0       # penalty for factors f_t
LAM_G = 1.0       # penalty for factors g_t
LAM_H = 1.0       # penalty for factors h_t

MAX_ITER = 15
TOL = 1e-4

# Forecast / evaluation
TRAIN_FRAC = 0.7

# Practical: downsample time to make estimation fast (set 1 to disable)
TIME_STRIDE = 10   # e.g., 20 means use every 10th time point

