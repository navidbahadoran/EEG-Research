
# EEG Dynamic Regime Modeling with vMF State Probabilities

## Overview

This repository studies **dynamic EEG brain‑state probabilities** derived from directional features using econometric and machine‑learning models.

The response variable throughout the project is a **7‑dimensional probability vector of directional EEG states** obtained from a **von Mises–Fisher (vMF) mixture model**.

The project investigates:

- how EEG‑derived brain states evolve over time
- whether those dynamics exhibit **regime switching**
- whether regimes are **global or unit‑specific**
- which models best capture the temporal structure of EEG state probabilities

The repository implements a complete workflow:

1. build directional EEG features and vMF state probabilities  
2. construct panel datasets  
3. estimate dynamic models  
4. compare model performance  
5. interpret regimes and state interactions  
6. produce figures for papers and presentations  

---

# Main Result

Across all tested models, the best‑performing specification is:

## **Unit‑Specific Switching PVAR (Model B)**

Each unit follows its **own regime path**, while the regime‑specific dynamics are **shared across units**.

> EEG state dynamics are best described by **unit‑specific switching among a small number of shared dynamic regimes**, rather than by a single global regime path.

---

# Repository Structure

```
EEG-Research/
│
├── README.md
├── config.py                 # Project configuration parameters
├── data_index.py             # Data indexing and consistency checks
│
├── eeg/                      # Core model implementations
│   ├── __init__.py
│   ├── pvar_full_model_parallel_adaptive.py
│   ├── vmf_baselines.py
│   ├── switching_pvar.py
│   ├── switching_factor_pvar.py
│   ├── unit_switching_pvar.py
│   └── slds_panel.py
│
├── vmf/                      # Feature construction and panel building
│   ├── vmf_features.py
│   ├── vmf_panel_builder.py
│   └── vmf_panel_builder_pooled.py
│
├── scripts/                  # Model runners
│   ├── model_common.py
│   ├── run_vmf_factor_pvar_pooled_runner.py
│   ├── run_vmf_baselines.py
│   ├── run_switching_pvar_prototype.py
│   ├── run_switching_factor_pvar.py
│   ├── run_unit_switching_pvar.py
│   └── run_slds_panel_prototype.py
│
├── notebooks/
│   └── final_model_analysis.ipynb   # Main analysis notebook
│
├── outputs/                  # Saved artifacts and metrics
│
└── archive/                  # Old experiments and prototypes
```

---

# Modeling Progression

The project compares models with increasing dynamic flexibility.

### 1. Pooled Factor‑PVAR
Smooth dynamic panel model with latent factors.

### 2. Baseline Models
Used as reference benchmarks.

- Persistence model  
- Ridge VARX regression  
- Markov transition baseline  
- Multinomial dominant‑state classifier  

### 3. Global Switching PVAR
Single regime path shared across all units.

### 4. Switching Factor‑PVAR (Option A)
Global switching model augmented with latent factors.

### 5. **Unit‑Specific Switching PVAR (Option B)**

Each unit has its own regime path while sharing regime‑specific dynamics.

**This is the best‑performing model.**

### 6. SLDS‑Style Panel Prototype (Option C)
Latent state‑space switching model used to test dimensionality reduction approaches.

---

# Data Representation

For each unit \(i\) and time \(t\), the response variable is a **7‑dimensional probability vector**

y_it = (p_it1, ..., p_it7)

representing posterior probabilities of directional EEG states.

Typical dataset dimensions:

| Quantity | Approximate Value |
|----------|------------------|
| Units | ~302 |
| Time windows | ~1761 |
| EEG states | 7 |
| Covariates | ~65 |

---

# States vs Regimes

A key conceptual distinction:

### EEG States (7)
Instantaneous directional states estimated from the vMF mixture.

### Dynamic Regimes (3)
Latent contexts governing how the **probability vector evolves over time**.

Thus:

- **states** = instantaneous configurations
- **regimes** = dynamic laws governing transitions

The regimes control the **dynamics of the probability vector**, not the states themselves.

---

# Running the Models

Recommended execution order:

### 1. Pooled Factor‑PVAR

```
python scripts/run_vmf_factor_pvar_pooled_runner.py
```

### 2. Baselines

```
python scripts/run_vmf_baselines.py
```

### 3. Global Switching Prototype

```
python scripts/run_switching_pvar_prototype.py
```

### 4. Switching Factor‑PVAR

```
python scripts/run_switching_factor_pvar.py
```

### 5. Unit‑Specific Switching PVAR

```
python scripts/run_unit_switching_pvar.py
```

### 6. SLDS Prototype

```
python scripts/run_slds_panel_prototype.py
```

---

# Outputs

Model outputs are saved in

```
outputs/
```

Typical files:

```
*_metrics.csv
*_artifacts.npz
```

Example artifact prefixes:

```
vmf_pvar_pooled_*
baseline_*
switching_pvar_k3_*
switching_factor_pvar_k3_*
unit_switching_pvar_k3_*
slds_panel_k3_*
```

---

# Evaluation Metrics

Models are compared using:

- MSE
- RMSE
- R²
- dominant‑state accuracy
- KL divergence
- cross‑entropy

---

# Analysis Notebook

Main analysis notebook:

```
notebooks/final_model_analysis.ipynb
```

The notebook:

1. loads saved outputs
2. builds model comparison tables
3. visualizes regime structure
4. produces figures for papers and presentations

---

# Key Figures

The analysis notebook generates:

- model comparison table
- regime usage shares
- transition matrices
- regime–state interaction heatmap
- switching heterogeneity across units
- prediction confusion matrix

---

# Author

Navid Bahadoran  
Department of Mathematics  
Florida State University

GitHub: https://github.com/navidbahadoran/EEG-Research
