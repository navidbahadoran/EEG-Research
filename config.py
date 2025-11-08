# config.py
from __future__ import annotations

# vMF / IFE configs
K_VMF = 4
RANK_GRID = [1, 2, 3]

# Axis-1 order for sessions in clean_EC.npy
SESSION_NAMES = ["day1_morning", "day1_afternoon", "day2_morning", "day2_afternoon"]

# Subject metadata for clean_EC.npy (index-aligned)
# tuple: (label, male=1/female=0, age_years)
SUBJECT_META = [
    ("AM", 0, 24),
    ("CL", 0, 23),
    ("CQ", 1, 26),
    ("DB", 1, 22),
    ("DC", 1, 20),
    ("DL", 1, 23),
    ("ErL", 1, 22),
    ("EvL", 0, 20),
    ("HZ", 1, 23),
    ("KM", 0, 24),
    ("LS", 0, 21),
    ("TL", 1, 23),
]

# Files for the extra CSV subject (same order as SESSION_NAMES)
CSV_SUBJECT_FILES = [
    "subject_0_day1_morning.csv",
    "subject_0_day1_afternoon.csv",
    "subject_0_day2_morning.csv",
    "subject_0_day2_afternoon.csv",
]

# Metadata for the extra (CSV) subject — edit to the true values if needed
EXTRA_SUBJECT_META = ("S_extra", 1, 23)  # (label, male=1/0, age)
