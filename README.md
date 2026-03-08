
# Dynamic EEG Modeling with Panel VAR and Latent Factors

This repository contains a reproducible research pipeline for modeling EEG data
using dynamic panel vector autoregressions (PVARs) with latent factor
structure and directional EEG features derived from vMF mixture models.

The project is designed to be both:
- a scientific modeling framework for EEG dynamics
- an educational reference implementation for dynamic panel models in neuroscience

The pipeline moves step-by-step from exploratory analysis of latent brain
states to full dynamic modeling of EEG time series.

---

# 1. Scientific Motivation

EEG signals are:
- high-dimensional
- strongly time dependent
- heterogeneous across subjects
- driven by latent neural regimes

Traditional models struggle to capture this structure.

This project develops a framework that:
- models temporal dependence and cross-channel interactions
- allows subject-specific dynamics
- captures latent neural regimes via common factors
- integrates directional EEG features from vMF state models

The goal is structural understanding of brain dynamics, not black-box prediction.

---

# 2. Staged Modeling Strategy

The project proceeds in two stages.

## Stage A — vMF State Dynamics

Goal: understand latent directional EEG states.

EEG windows are projected onto the unit sphere and modeled using a
von Mises–Fisher (vMF) mixture model.

Each window yields a probability vector:

Z_t ∈ Δ^{K−1}

where K = 7 latent brain states.

From the time series of probability vectors we compute interpretable features:
- entropy
- state occupancy
- switching rates
- volatility
- transition statistics

These features are used to predict subject-level variables such as:
- attention
- p-factor
- internalizing
- externalizing

This stage establishes that state dynamics contain meaningful information.

---

## Stage B — Dynamic Panel VAR with Latent Factors

Stage B models the dynamics of the vMF probability vectors themselves.

The pooled model is:

Y_{i,t+1} = A_{i,t} Y_{i,t} + B_{i,t} X_{i,t} + Λ_i f_t + u_{i,t+1}

where:

| term | meaning |
|-----|------|
| Y_it | vMF state probability vector (7 states) |
| X_it | dynamic summary features + demographics |
| A_it | time-varying autoregressive matrix |
| B_it | time-varying covariate effect matrix |
| f_t | latent neural regime factor |
| Λ_i | subject loading matrix |

Time variation is modeled using low-rank factorization:

A_{i,t} = Σ h_t^(m) D_i^(m)  
B_{i,t} = Σ g_t^(m) C_i^(m)

Thus the model contains three latent processes:

| factor | interpretation |
|------|-------------|
| f_t | common neural activity |
| g_t | modulation of covariate effects |
| h_t | modulation of autoregressive dynamics |

---

# 3. Data Layout

All EEG and vMF data live outside the repository.

Paths are configured in config.py.

VMF_DATA_DIR/
├── raw/ (raw EEG files)
├── vMF/ (vMF posterior probability .npz files)
└── vmf_fixedmus_summary_K7.csv (metadata)

Each .npz file contains a matrix:

P ∈ R^{T × 7}

representing posterior probabilities of the seven latent states.

---

# 4. Repository Structure

.
├── README.md
├── config.py
├── data_index.py
│
├── vmf/
│   ├── vmf_panel_builder_pooled.py
│   ├── vmf_npz.py
│   └── vmf_features.py
│
├── eeg/
│   └── pvar_full_model_parallel_adaptive.py
│
├── scripts/
│   └── run_vmf_factor_pvar_pooled_runner.py
│
├── notebooks/
│   └── 20_vmf_pvar_pooled_results.ipynb
│
├── outputs/
│
└── paper/
    Overleaf LaTeX manuscript

---

# 5. Core Modules

## vmf_panel_builder_pooled.py
Builds the panel dataset used by the model.

Steps:
1. Load vMF probabilities
2. Downsample time series
3. Construct dynamic features
4. Add demographic variables
5. Produce Y_list and X_list

## pvar_full_model_parallel_adaptive.py
Core estimation engine implementing the class:

PVARFactorALSParallelAdaptive

The model is estimated using Alternating Least Squares (ALS).

Each iteration:
1. update subject parameters
2. update latent factors
3. compute loss

Subject updates are parallelized across CPU cores.

## run_vmf_factor_pvar_pooled_runner.py
Main pipeline driver.

Workflow:
build panel → standardize covariates → fit model → generate forecasts → save outputs

---

# 6. Running the Pipeline

Run the pooled model:

python scripts/run_vmf_factor_pvar_pooled_runner.py

The script will:
1. construct the panel dataset
2. fit the factor-PVAR model
3. generate out-of-sample forecasts
4. save results to outputs/

---

# 7. Output Files

## Metrics
vmf_pvar_pooled_metrics.csv

## Model Artifacts
vmf_pvar_pooled_artifacts.npz

Contains:
- y_true_oof
- y_pred_oof
- f_t
- g_t
- h_t
- Lambda
- loss_history

## Metadata
- vmf_pvar_pooled_targets.csv
- vmf_pvar_pooled_unit_meta.csv
- vmf_pvar_pooled_feature_info.csv

## Latent summaries
vmf_pvar_pooled_latent_subject_summary.csv

---

# 8. Visualization

Model results are explored in:

notebooks/20_vmf_pvar_pooled_results.ipynb

The notebook analyzes:
- prediction metrics
- KL divergence
- state prediction accuracy
- latent factors
- subject-level summaries

---

# 9. Interpretation of Results

Prediction accuracy is not the primary objective.

EEG voltages and state probabilities are noisy signals, so small R² values are expected.

The model is designed to uncover:
- latent neural regimes
- subject heterogeneity
- stimulus-driven synchronization

rather than maximize short-term forecast accuracy.

---

# 10. Educational Use

The repository can be used as a teaching resource for:
- dynamic panel models
- factor models
- time-series analysis
- neuroscience data science

The codebase mirrors the mathematical framework documented in the
accompanying paper.

---

# 11. Associated Paper

The full methodology and documentation are provided in the LaTeX manuscript:

paper/

which explains:
- the statistical model
- estimation algorithm
- code implementation
- computational performance
