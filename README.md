# EEG Dynamic Regime Modeling with vMF State Probabilities

## Overview

This repository studies **dynamic EEG brain-state probabilities** derived from directional features using econometric and machine learning models.

The response variable is a **7‑dimensional probability vector of directional EEG states** obtained from a **von Mises–Fisher (vMF) mixture model**.

The goal is to understand:

* how EEG-derived brain states evolve over time,
* whether those dynamics exhibit **regime switching**,
* whether regimes are **global or unit‑specific**, and
* which models best capture the temporal structure of EEG state probabilities.

The repository implements a full research workflow:

1. build directional EEG features and vMF state probabilities
2. construct panel datasets
3. estimate dynamic models
4. compare model performance
5. interpret regimes and state interactions
6. produce figures for papers and presentations

\---

# Main Result

Across all tested models, the best-performing specification is:

### **Unit‑Specific Switching PVAR (Model B)**

Each unit follows its own regime path while **sharing regime‑specific dynamics across units**.

> EEG state dynamics are best described by \\\\\\\*\\\\\\\*unit‑specific switching among a small number of shared dynamic regimes\\\\\\\*\\\\\\\*.

\---

# Repository Structure

```text

repo\\\\\\\_root/

│

├── README.md

├── config.py

├── data\\\\\\\_index.py

│

├── eeg/

│   Core model implementations

│

│   ├── \\\\\\\_\\\\\\\_init\\\\\\\_\\\\\\\_.py

│   ├── pvar\\\\\\\_full\\\\\\\_model\\\\\\\_parallel\\\\\\\_adaptive.py

│   ├── vmf\\\\\\\_baselines.py

│   ├── switching\\\\\\\_pvar.py

│   ├── switching\\\\\\\_factor\\\\\\\_pvar.py

│   ├── unit\\\\\\\_switching\\\\\\\_pvar.py

│   └── slds\\\\\\\_panel.py

│

├── vmf/

│   Feature construction and panel building

│

│   ├── vmf\\\\\\\_features.py

│   ├── vmf\\\\\\\_panel\\\\\\\_builder.py

│   └── vmf\\\\\\\_panel\\\\\\\_builder\\\\\\\_pooled.py

│

├── scripts/

│   Model runners and shared utilities

│

│   ├── model\\\\\\\_common.py

│   ├── run\\\\\\\_vmf\\\\\\\_factor\\\\\\\_pvar\\\\\\\_pooled\\\\\\\_runner.py

│   ├── run\\\\\\\_vmf\\\\\\\_baselines.py

│   ├── run\\\\\\\_switching\\\\\\\_pvar\\\\\\\_prototype.py

│   ├── run\\\\\\\_switching\\\\\\\_factor\\\\\\\_pvar.py

│   ├── run\\\\\\\_unit\\\\\\\_switching\\\\\\\_pvar.py

│   └── run\\\\\\\_slds\\\\\\\_panel\\\\\\\_prototype.py

│

├── notebooks/

│   Analysis, diagnostics, and presentation notebook(s)

│

│   └── final\\\\\\\_model\\\\\\\_analysis.ipynb

│

├── outputs/

│   Saved artifacts and metrics

│

└── archive/

\\\&#x20;   Old notebooks, prototypes, and superseded scripts

```

\---

# Modeling Progression

The project compares increasing levels of dynamic complexity.

### 1. Pooled Factor-PVAR
Dynamic panel model with latent factors.

- smooth dynamics
- no switching

---

### 2. Baseline Models

Used for benchmarking.

- Persistence baseline
- Ridge VARX
- Markov transition baseline
- Multinomial dominant-state classifier

---

### 3. Global Switching PVAR

Single regime path shared by all units.

---

### 4. Option A: Switching Factor-PVAR

Adds latent factors to the global switching model.

---

### 5. **Option B: Unit-Specific Switching PVAR**

Each unit has its own regime path while sharing dynamic laws.

**This is the best-performing model.**

---

### 6. Option C: SLDS-style Panel Model

Latent continuous state model with switching dynamics.

Used to test whether dimensionality reduction improves performance.

---

# Data Representation

For each unit \(i\) and time \(t\), the response variable is a **7-dimensional probability vector**

\[
y_{i,t} = (p_{i,t,1}, \dots, p_{i,t,7})
\]

representing the posterior probability of each directional EEG state.

Typical dataset dimensions:

| Quantity | Value |
|--------|------|
| Units | ~302 |
| Time windows | ~5870 |
| States | 7 |
| Covariates | ~65 |

---

# States vs Regimes

A key conceptual distinction:

### EEG States (7)

Instantaneous probability states from the vMF mixture.

### Dynamic Regimes (3)

Latent contexts governing how state probabilities evolve over time.

So the model does **not replace states with regimes**.

Instead:

- **states = instantaneous configurations**
- **regimes = dynamic laws governing transitions**

\---

# Data Representation

For unit i and time t:

y\_it = (p\_it1, ..., p\_it7)

This represents the probability distribution over **7 EEG directional states**.

Typical dataset dimensions:

Units ≈ 302  
Time windows ≈ 1761  
States = 7  
Covariates ≈ 65

\---

# States vs Regimes

Important distinction:

**EEG states (7)**  
Instantaneous vMF probability states.

**Dynamic regimes (3)**  
Latent contexts governing how those probabilities evolve.

Thus:

states = instantaneous configuration  
regimes = dynamic laws governing transitions

\---

# Running the Models

Recommended order:

Pooled factor‑PVAR

python scripts/run\_vmf\_factor\_pvar\_pooled\_runner.py

Baselines

python scripts/run\_vmf\_baselines.py

Global switching prototype

python scripts/run\_switching\_pvar\_prototype.py

Switching factor‑PVAR

python scripts/run\_switching\_factor\_pvar.py

Unit‑specific switching PVAR

python scripts/run\_unit\_switching\_pvar.py

SLDS prototype

python scripts/run\_slds\_panel\_prototype.py

\---

# Outputs

All model outputs are stored in

outputs/

Typical files:

\*\_metrics.csv  
\*\_artifacts.npz

Example prefixes:

vmf\_pvar\_pooled\_\*  
baseline\_\*  
switching\_pvar\_k3\_\*  
switching\_factor\_pvar\_k3\_\*  
unit\_switching\_pvar\_k3\_\*  
slds\_panel\_k3\_\*

\---

# Evaluation Metrics

Models are compared using:

MSE  
RMSE  
R²  
dominant‑state accuracy  
KL divergence  
cross entropy

\---

# Analysis Notebook

Main notebook:

notebooks/final\_model\_analysis.ipynb

The notebook:

1. loads saved outputs
2. builds the model comparison table
3. generates regime diagnostics
4. produces figures used in the paper and poster

\---

# Key Figures

Model comparison table  
Regime usage shares  
Transition matrix  
Regime‑state interaction heatmap  
Switching heterogeneity across units  
Prediction confusion matrix

\---

# Author

Navid Bahadoran  
Department of Mathematics  
Florida State University

