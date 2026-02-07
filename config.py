# =========================
# config.py
# =========================
from pathlib import Path

# Folder containing BOTH:
#   - all vMF .npz files
#   - vmf_fixedmus_summary_K7.csv
DATA_DIR = Path(
    r"D:/Navid/FSU/OneDrive - Florida State University/FSU/Courses/2025-2026/Spring/"
    r"Sahar Research/EEG Data/data_for_navid"
)

CSV_NAME = "vmf_fixedmus_summary_K7.csv"
CSV_PATH = DATA_DIR / CSV_NAME

# vMF mixture size
VMF_K = 7

# Output folder (created automatically)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 123
N_FOLDS = 5

# If true: use P[:, :K-1] when directly using P as regressors (simplex collinearity fix).
DROP_LAST_PROB_COL = True
