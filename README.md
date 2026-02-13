# Dynamic EEG Modeling with Panel VAR and Latent Factors

This repository implements a clean, end-to-end research pipeline for modeling EEG dynamics using:

- time-series analysis  
- panel vector autoregressions (PVAR)  
- latent factor models  
- directional EEG representations (von Mises–Fisher, vMF)

The project is designed as a teaching-quality research codebase, progressing from interpretable vMF-only analysis (Stage A) to a full dynamic PVAR model with heterogeneous and time-varying effects (Stage B).

---

## 1. Scientific Objective

We model multichannel EEG data using the dynamic system

y_it = A_it y_i,t-1 + B_it X_it + Λ_i f_t + u_it,

where:

- y_it ∈ R^G is window-level multichannel EEG features  
- A_it ∈ R^{G×G} governs time-varying propagation  
- X_it are observed covariates (currently vMF features Z_it)  
- B_it are time-varying covariate slopes  
- f_t are latent common brain-state factors  
- Λ_i are subject-specific factor loadings  
- u_it is idiosyncratic noise  

This structure allows subject heterogeneity, regime changes, low-rank interpretability, and honest out-of-sample prediction.

---

## 2. Data Layout (Public-Safe)

All data live outside the repository and are referenced through aliases in config.py.

<VMF_DATA_DIR>/
├── raw/
├── vMF/
└── vmf_fixedmus_summary_K7.csv

---

## 3. Repository Structure

```text
.
├── README.md
├── config.py
├── data_index.py
├── run_vmf_pipeline.py
├── stageA_predict.py
├── run_full_pvar_task.py
│
├── vmf_npz.py
├── vmf_features.py
├── vmf_dataset.py
│
├── eeg/
│   ├── __init__.py
│   ├── raw_eeg_npy.py
│   ├── windowing.py
│   ├── eeg_features.py
│   ├── panel_dataset.py
│   └── pvar_full_model.py
│
├── notebooks/
│   ├── 00_vmf_results.ipynb
│   └── 10_stageB_results.ipynb
│
├── baselines/
│   ├── stageB_build_panel.py
│   └── stageB_predict_varx.py
│
└── outputs/
```

---

## 4. Stage A — vMF Time-Series Analysis

Run:
python run_vmf_pipeline.py  
python stageA_predict.py

Outputs:
- outputs/vmf_subject_features.csv  
- outputs/stageA_oof_predictions.csv  

Notebook:
- notebooks/00_vmf_results.ipynb

---

## 5. Stage B — Full Dynamic PVAR Model

Steps:
1. python data_index.py  
2. python run_full_pvar_task.py  

Outputs:
- outputs/full_pvar_<TASK>_metrics.csv  
- outputs/full_pvar_<TASK>_predictions.npz  

Notebook:
- notebooks/10_stageB_results.ipynb

---

## 6. Design Principles

- separation of data, modeling, visualization  
- reproducible experiments  
- student-readable code  

---

## Final Note

This repository is a complete MVP implementation of a dynamic EEG PVAR model.
