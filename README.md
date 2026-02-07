# EEG-Research-main

This repo is a **teachable** implementation scaffold for your EEG panel VAR project.

## What you can do now (before raw EEG arrives)
You already have **vMF time series** stored in `.npz`. Start with:
1. Put `.npz` files in `data/vmf_npz/`
2. Run:
   ```bash
   python run_vmf_pipeline.py --csv vmf_fixedmus_summary_K7.csv --K 7
   ```
   This will:
   - load each vMF time series Z_{it}
   - extract dynamic features (occupancy, entropy, switching, volatility)
   - run quick baselines to predict traits (p_factor, attention, etc.)
   - write `vmf_subject_features.csv`

## When raw EEG arrives
Preprocess raw EEG into **windowed features**:
  - for each subject/session/task, create an `.npz` containing `Y` with shape (T, G)
Then use `panel_varx.py` (pooled ridge VARX baseline) or extend to the full model:
  y_{it} = A_{it} y_{i,t-1} + B_{it} X_{it} + Λ_i f_t + u_{it}

## Folder layout
See `data/README.md`.
