# =========================
# config.py
# =========================
# config.py
from pathlib import Path

# Public-safe alias paths (set these locally)
VMF_DATA_DIR = Path(
    r"D:/Navid/FSU/OneDrive - Florida State University/FSU/Courses/2025-2026/Spring/"
    r"Sahar Research/EEG Data/data_for_navid"
)          # contains vMF/ and raw/ and the CSV
RAW_EEG_DIR  = VMF_DATA_DIR / "raw"            # *.npy
VMF_DIR      = VMF_DATA_DIR / "vMF"            # *.npz
VMF_CSV_PATH = VMF_DATA_DIR / "vmf_fixedmus_summary_K7.csv"

OUTPUT_DIR   = Path("outputs")

# Stage A settings
K_VMF = 7

# Stage B settings (baseline)
SFREQ_FALLBACK = 128.0         # only used if raw file has no metadata available
WIN_SEC = 2.0                  # window length (seconds)
STEP_SEC = 1.0                 # stride (seconds)

# EEG bands (Hz) for bandpower features
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

RANDOM_SEED = 123

# from pathlib import Path

# # Folder containing BOTH:
# #   - all vMF .npz files
# #   - vmf_fixedmus_summary_K7.csv
# DATA_DIR = Path(
#     r"D:/Navid/FSU/OneDrive - Florida State University/FSU/Courses/2025-2026/Spring/"
#     r"Sahar Research/EEG Data/data_for_navid"
# )

# CSV_NAME = "vmf_fixedmus_summary_K7.csv"
# CSV_PATH = DATA_DIR / CSV_NAME

# # vMF mixture size
# VMF_K = 7

# # Output folder (created automatically)
# OUTPUT_DIR = Path("outputs")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RANDOM_SEED = 123
# N_FOLDS = 5

# # If true: use P[:, :K-1] when directly using P as regressors (simplex collinearity fix).
# DROP_LAST_PROB_COL = True
