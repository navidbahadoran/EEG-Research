
# EEG Dynamic Regime Modeling with vMF State Probabilities

## Overview

This repository studies **dynamic EEG brain-state probabilities** derived from directional features using econometric and machine learning models.

The response variable is a **7‑dimensional probability vector of directional EEG states** obtained from a **von Mises–Fisher (vMF) mixture model**.

The goal is to understand:

- how EEG-derived brain states evolve over time,
- whether those dynamics exhibit **regime switching**,
- whether regimes are **global or unit‑specific**, and
- which models best capture the temporal structure of EEG state probabilities.

The repository implements a full research workflow:

1. build directional EEG features and vMF state probabilities
2. construct panel datasets
3. estimate dynamic models
4. compare model performance
5. interpret regimes and state interactions
6. produce figures for papers and presentations

---

# Main Result

Across all tested models, the best-performing specification is:

### **Unit‑Specific Switching PVAR (Model B)**

Each unit follows its own regime path while **sharing regime‑specific dynamics across units**.

> EEG state dynamics are best described by **unit‑specific switching among a small number of shared dynamic regimes**.

---

# Repository Structure

repo_root/

README.md  
config.py  
data_index.py  

eeg/  
    Core model implementations

    pvar_full_model_parallel_adaptive.py  
    vmf_baselines.py  
    switching_pvar.py  
    switching_factor_pvar.py  
    unit_switching_pvar.py  
    slds_panel.py  

vmf/  
    Feature construction and panel building

    vmf_features.py  
    vmf_panel_builder.py  
    vmf_panel_builder_pooled.py  

scripts/  
    Model runners

    model_common.py  
    run_vmf_factor_pvar_pooled_runner.py  
    run_vmf_baselines.py  
    run_switching_pvar_prototype.py  
    run_switching_factor_pvar.py  
    run_unit_switching_pvar.py  
    run_slds_panel_prototype.py  

notebooks/  
    final_model_analysis.ipynb

outputs/  
    saved artifacts and metrics

archive/  
    earlier experiments and prototypes

---

# Modeling Progression

1. **Pooled Factor‑PVAR**  
   Smooth dynamic panel model with latent factors.

2. **Baseline Models**
   - Persistence
   - Ridge VARX
   - Markov transition
   - Multinomial classifier

3. **Global Switching PVAR**
   Single regime path shared across all units.

4. **Switching Factor‑PVAR (Option A)**
   Global switching with latent factors.

5. **Unit‑Specific Switching PVAR (Option B)**  
   Each unit has its own regime path.  
   **Best performing model.**

6. **SLDS Panel Prototype (Option C)**
   Switching linear dynamical system with latent compression.

---

# Data Representation

For unit i and time t:

y_it = (p_it1, ..., p_it7)

This represents the probability distribution over **7 EEG directional states**.

Typical dataset dimensions:

Units ≈ 302  
Time windows ≈ 1761  
States = 7  
Covariates ≈ 65

---

# States vs Regimes

Important distinction:

**EEG states (7)**  
Instantaneous vMF probability states.

**Dynamic regimes (3)**  
Latent contexts governing how those probabilities evolve.

Thus:

states = instantaneous configuration  
regimes = dynamic laws governing transitions

---

# Running the Models

Recommended order:

Pooled factor‑PVAR

python scripts/run_vmf_factor_pvar_pooled_runner.py

Baselines

python scripts/run_vmf_baselines.py

Global switching prototype

python scripts/run_switching_pvar_prototype.py

Switching factor‑PVAR

python scripts/run_switching_factor_pvar.py

Unit‑specific switching PVAR

python scripts/run_unit_switching_pvar.py

SLDS prototype

python scripts/run_slds_panel_prototype.py

---

# Outputs

All model outputs are stored in

outputs/

Typical files:

*_metrics.csv  
*_artifacts.npz

Example prefixes:

vmf_pvar_pooled_*  
baseline_*  
switching_pvar_k3_*  
switching_factor_pvar_k3_*  
unit_switching_pvar_k3_*  
slds_panel_k3_*

---

# Evaluation Metrics

Models are compared using:

MSE  
RMSE  
R²  
dominant‑state accuracy  
KL divergence  
cross entropy

---

# Analysis Notebook

Main notebook:

notebooks/final_model_analysis.ipynb

The notebook:

1. loads saved outputs
2. builds the model comparison table
3. generates regime diagnostics
4. produces figures used in the paper and poster

---

# Key Figures

Model comparison table  
Regime usage shares  
Transition matrix  
Regime‑state interaction heatmap  
Switching heterogeneity across units  
Prediction confusion matrix

---

# Author

Navid Bahadoran  
Department of Mathematics  
Florida State University
