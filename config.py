# =========================
# config.py
# =========================
# config.py
# config.py
from pathlib import Path

# Public-safe alias for repo
VMF_DATA_DIR = Path("<VMF_DATA_DIR>")  # set locally

RAW_EEG_DIR  = VMF_DATA_DIR / "raw"
VMF_DIR      = VMF_DATA_DIR / "vMF"
VMF_CSV_PATH = VMF_DATA_DIR / "vmf_fixedmus_summary_K7.csv"

OUTPUT_DIR   = Path("outputs")

# Data dimensions (from your example)
G_CHANNELS = 32
K_VMF = 7

# Modeling / estimation config
RANDOM_SEED = 123

# Full model ranks (start small for stability)
RF = 2   # rank of interactive effects f_t
RG = 2   # rank of covariate-slope factors g_t
RH = 2   # rank of dynamics factors h_t

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
TIME_STRIDE = 10   # e.g., 10 means use every 10th time point

