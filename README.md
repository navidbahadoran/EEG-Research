# Dynamic EEG Modeling with Panel VAR and Latent Factors

This repository contains a fully implemented, documented, and reproducible
research pipeline for modeling EEG data using dynamic panel vector
autoregressions (PVARs) with latent factor structure.

The project is designed to be both scientifically meaningful and educational,
guiding students step by step from exploratory analysis of latent EEG structure
to full dynamic modeling of raw multichannel EEG signals.

---

## 1. Project Overview

EEG data are high-dimensional, dynamic, and heterogeneous across subjects and
experimental contexts. This project develops a principled statistical framework
that:

- models temporal dependence and cross-channel interactions,
- allows subject-specific and time-varying dynamics,
- incorporates latent neural structure through factors,
- connects directional EEG features to raw EEG dynamics.

---

## 2. Staged Design

### Stage A — vMF Time-Series Analysis

Goal: understand latent directional EEG structure before modeling raw voltages.

Key steps:
- Fit vMF mixture models to windowed EEG segments.
- Treat posterior probabilities as multivariate time series.
- Extract interpretable dynamic summaries.
- Predict subject-level cognitive traits using out-of-sample validation.

---

### Stage B — Dynamic Panel VAR with Latent Factors

Goal: model raw multichannel EEG dynamics while accounting for latent structure
and subject heterogeneity.

Core model:
y_it = A_it y_i,t-1 + B_it X_it + Lambda_i f_t + u_it

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
Prediction accuracy is used only as a diagnostic.---

---

## 3. Data Layout (Public-Safe)

All data live outside the repository and are referenced through aliases in config.py.

```text
<VMF_DATA_DIR>/
├── raw/
├── vMF/
└── vmf_fixedmus_summary_K7.csv
```
---

## 4. Repository Structure

```text
.
├── README.md
├── config.py
├── data_index.py
├── run_full_pvar_task_parallel.py
│── run_full_pvar_all_task_parallel.py
│
├── run_vmf_pipeline.py
├── stageA_predict.py
├── vmf_npz.py
├── vmf_features.py
├── vmf_dataset.py

├── eeg/
│   ├── __init__.py
│   ├── raw_eeg_npy.py
│   ├── windowing.py
│   ├── eeg_features.py
│   ├── panel_dataset.py
│   └── pvar_full_model_parallel_adaptive.py
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

## 5. How to Run

Stage A:
python run_vmf_pipeline.py
python stageA_predict.py

Stage B:
python data_index.py
python run_full_pvar_all_tasks_parallel.py

---

## 6. Interpretation

Small R² values for raw EEG prediction are expected and reflect noise-dominated
signals. The model is intended for structural analysis, not black-box
forecasting.

---

## 7. Educational Use

This repository is suitable for advanced undergraduate and graduate-level
training in time series analysis, econometrics, and applied neuroscience.
