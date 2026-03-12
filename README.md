# EEG Dynamic Regime Modeling with vMF State Probabilities

## Overview

This repository studies **dynamic EEG brain-state probabilities** derived from directional features using a sequence of econometric and machine learning models.

The response variable throughout the project is a **7-dimensional probability vector of vMF brain states**.  
The overall goal is to understand:

- how EEG-derived brain states evolve over time,
- whether those dynamics exhibit **regime switching**,
- whether regimes are better modeled as **global** or **unit-specific**,
- and which covariates explain transitions between brain states.

The repository implements a complete workflow:

1. build directional EEG features and vMF state probabilities,
2. construct pooled panel data,
3. fit and compare dynamic models,
4. interpret regimes, state interactions, and covariate effects,
5. prepare figures and tables for papers and presentations.

---

## High-level modeling progression

The project compares several levels of dynamic complexity.

### 1. Pooled factor-PVAR
A pooled dynamic panel model with latent factors.

- Smooth dynamics
- Shared across units
- No regime switching

### 2. Baseline models
Simple reference models used for comparison.

- Persistence baseline
- Ridge VARX baseline
- Markov transition baseline
- Multinomial dominant-state classifier

### 3. Global switching PVAR prototype
A switching model with a **single regime path shared across all units**.

### 4. Option A: Switching factor-PVAR
A global switching model augmented with latent factors.

### 5. Option B: Unit-specific switching PVAR
A switching model where **each unit has its own regime path**, while regime-specific coefficients are shared across units.

This is currently the **best-performing model**.

### 6. Option C: SLDS-style panel prototype
A latent continuous-state model with switching dynamics, used to test whether low-dimensional latent compression improves over direct state switching.

---

## Main empirical conclusion so far

Across the tested models:

- pooled smooth models capture average dynamics but miss heterogeneity,
- global switching models improve flexibility,
- **unit-specific switching models perform best**,
- low-dimensional SLDS-style compression does not outperform direct unit-specific switching.

In short:

> EEG state dynamics are best described by **unit-specific switching among a small number of shared dynamic regimes**, rather than by a single global regime path or a strongly compressed latent dynamical system.

---

## Repository structure

```text
repo_root/
│
├── README.md
├── config.py
├── data_index.py
│
├── eeg/
│   Core model implementations
│
│   ├── __init__.py
│   ├── pvar_full_model_parallel_adaptive.py
│   ├── vmf_baselines.py
│   ├── switching_pvar.py
│   ├── switching_factor_pvar.py
│   ├── unit_switching_pvar.py
│   └── slds_panel.py
│
├── vmf/
│   Feature construction and panel building
│
│   ├── vmf_features.py
│   ├── vmf_panel_builder.py
│   └── vmf_panel_builder_pooled.py
│
├── scripts/
│   Model runners and shared utilities
│
│   ├── model_common.py
│   ├── run_vmf_factor_pvar_pooled_runner.py
│   ├── run_vmf_baselines.py
│   ├── run_switching_pvar_prototype.py
│   ├── run_switching_factor_pvar.py
│   ├── run_unit_switching_pvar.py
│   └── run_slds_panel_prototype.py
│
├── notebooks/
│   Analysis, diagnostics, and presentation notebook(s)
│
│   └── final_model_analysis.ipynb
│
├── outputs/
│   Saved artifacts and metrics
│
└── archive/
    Old notebooks, prototypes, and superseded scripts
```

---

## What each active file does

## `eeg/`

### `pvar_full_model_parallel_adaptive.py`
Main pooled factor-PVAR model.

### `vmf_baselines.py`
Baseline models used for comparison.

### `switching_pvar.py`
Prototype global switching VARX model.

### `switching_factor_pvar.py`
Option A: switching factor-PVAR.

### `unit_switching_pvar.py`
Option B: unit-specific switching PVAR.

### `slds_panel.py`
Option C: SLDS-style latent switching panel model.

---

## `vmf/`

### `vmf_features.py`
Builds and processes directional EEG features.

### `vmf_panel_builder.py`
General panel construction utilities.

### `vmf_panel_builder_pooled.py`
Main pooled panel builder used in the current workflow.

---

## `scripts/`

### `model_common.py`
Shared utilities for:
- loading the panel,
- train-only standardization,
- saving metrics,
- common scoring helpers.

### `run_vmf_factor_pvar_pooled_runner.py`
Runs the pooled factor-PVAR model.

### `run_vmf_baselines.py`
Runs all baseline models and writes a baseline comparison table.

### `run_switching_pvar_prototype.py`
Runs the global switching PVAR prototype.

### `run_switching_factor_pvar.py`
Runs Option A.

### `run_unit_switching_pvar.py`
Runs Option B.

### `run_slds_panel_prototype.py`
Runs Option C.

---

## Data representation

For each unit \(i\) and time \(t\), the response is a 7-dimensional state probability vector

\[
y_{i,t} = (p_{i,t,1}, \dots, p_{i,t,7})
\]

with entries summing to one.

The panel has the form:

- \(N\) units
- \(T\) time points
- \(G = 7\) response dimensions
- \(p \approx 65\) covariates

Typical full-sample dimensions in the current project are approximately:

- \(N \approx 302\)
- \(T \approx 5870\)
- \(G = 7\)
- \(p = 65\)

---

## Covariates

The covariates typically include:

- entropy,
- switching-rate summaries,
- volatility summaries,
- occupancy summaries,
- transition summaries,
- task indicators,
- subject-level covariates such as age and sex.

These enter the models through covariate coefficient blocks \(B_k\).

---

## Why there are 3 regimes but 7 states

A common source of confusion is the distinction between **brain states** and **dynamic regimes**.

### The 7 states
These are the instantaneous vMF probability states.

### The 3 regimes
These are **dynamic contexts** that govern how the 7 states evolve over time.

So the model is **not** replacing 7 states with 3 states.  
Instead, it is saying:

- the observed brain state probabilities still live in 7 dimensions,
- but the **rules of motion** for those 7 probabilities change across 3 regimes.

A useful intuition is:

- **states** = instantaneous configurations,
- **regimes** = dynamic laws governing transitions between configurations.

---

## Preprocessing design

The cleaned repository should use:

- `load_panel_data()` for **raw panel loading only**
- `standardize_from_train()` for **train-only covariate standardization**

That means runners should:

1. load raw `Y_list`, `X_list`,
2. compute `train_end`,
3. standardize `X_list` using training periods only,
4. fit the model.

This avoids leakage and keeps preprocessing consistent across models.

---

## How to run the code

Recommended run order:

### 1. Pooled factor-PVAR
```bash
python scripts/run_vmf_factor_pvar_pooled_runner.py
```

### 2. Baselines
```bash
python scripts/run_vmf_baselines.py
```

### 3. Global switching PVAR prototype
```bash
python scripts/run_switching_pvar_prototype.py
```

### 4. Option A: Switching factor-PVAR
```bash
python scripts/run_switching_factor_pvar.py
```

### 5. Option B: Unit-specific switching PVAR
```bash
python scripts/run_unit_switching_pvar.py
```

### 6. Option C: SLDS-style panel prototype
```bash
python scripts/run_slds_panel_prototype.py
```

---

## Outputs

All outputs are saved under:

```text
outputs/
```

Typical files include:

- `*_metrics.csv`
- `*_artifacts.npz`

Suggested artifact naming prefixes:

- `vmf_pvar_pooled_*`
- `baseline_*`
- `switching_pvar_k3_*`
- `switching_factor_pvar_k3_*`
- `unit_switching_pvar_k3_*`
- `slds_panel_k3_r3_*`

---

## Key metrics reported

Each model typically reports:

- MSE
- RMSE
- \(R^2\)
- dominant-state accuracy
- KL divergence
- cross-entropy

These metrics allow direct comparison across models.

---

## Presentation Notebook

The repository is using one main notebook:

```text
notebooks/final_model_analysis.ipynb
```

This notebook will:

1. load all saved outputs,
2. create the final model comparison table,
3. produce all main figures,
4. summarize the scientific interpretation.

Notebook sections:

1. Data overview
2. Baseline comparison
3. Pooled factor-PVAR results
4. Switching prototype results
5. Option A results
6. Option B results
7. Option C results
8. Regime diagnostics
9. Covariate importance
10. Brain-state interaction networks
11. Switching heterogeneity across units
12. Final takeaway slide/table

---

## Recommended figures for paper and presentation

The final notebook generates:

### Model comparison
- final comparison table across all models

### Regime diagnostics
- regime usage shares
- regime probabilities over time
- transition matrices

### Dynamics
- regime-specific \(A_k\) heatmaps
- regime-specific covariate importance

### Prediction quality
- confusion matrix of true vs predicted dominant state
- state recall

### Scientific interpretation
- regime-specific brain-state interaction networks
- switching frequency across units
- regime entropy across units
- unit × time dominant-regime heatmap

---

## Current model ranking

Based on the current results:

1. **Option B — Unit-specific switching PVAR**  
2. Option A — Switching factor-PVAR  
3. Option C — SLDS-style panel prototype  
4. Global switching prototype  
5. Pooled factor-PVAR / baselines

---

## Author

Navid Bahadoran  
Florida State University
