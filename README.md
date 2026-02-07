# Dynamic EEG Modeling  
## Stage A: vMF Time-Series Analysis (Before Raw EEG)

---

## 1. What is this project?

This repository is part of a larger research project on **dynamic EEG modeling** using ideas from:

- time-series analysis  
- panel data econometrics  
- latent factor models  
- neuroscience  

The **full project** aims to model multichannel EEG signals using **panel vector autoregressions (PVARs)** with:
- subject heterogeneity  
- time-varying dynamics  
- latent brain states  
- covariates derived from EEG structure  

However, **this repository currently implements only Stage A**.

---

## 2. What is Stage A?

### Stage A = vMF-only analysis (no raw EEG amplitudes)

In Stage A, we **do not use raw EEG signals** (voltages, bandpower, etc.).

Instead, we start from **directional EEG representations** produced by a **von Mises–Fisher (vMF) mixture model**.

Each subject has time-varying vMF outputs that describe **latent EEG states over time**.

### What we study in Stage A

We ask:

> Can the *temporal dynamics* of vMF latent states predict subject-level cognitive and behavioral traits such as  
> - attention  
> - general psychopathology (`p_factor`)?

This stage is:
- scientifically meaningful on its own  
- a complete undergraduate-level research project  
- a necessary foundation for the full EEG model  

Nothing in Stage A is throwaway.

---

## 3. Conceptual model behind Stage A

For each subject \( i \) and time window \( t \):

- We observe a **vMF posterior probability vector**
  \[
  Z_{it} = (P_{it,1}, \dots, P_{it,K}) \in \Delta^{K-1}
  \]
- Here, \(K = 7\) latent vMF states  
- \(Z_{it}\) evolves over time and reflects changing brain configurations  

### What we do with \(Z_{it}\)

1. Treat \(Z_{it}\) as a **time series**  
2. Extract **dynamic features** from it:
   - entropy  
   - switching rate  
   - volatility  
   - state occupancy  
   - transition structure  
3. Aggregate these features to the **subject level**  
4. Predict subject-level traits using **out-of-sample, subject-wise validation**

This corresponds to building and validating the **covariate component** of the full EEG model.

---

## 4. Data assumptions (Stage A)

All required data live in a single directory, referred to throughout this README as:

```
<VMF_DATA_DIR>/
```

This directory must contain:

1. **vMF `.npz` files** (one per subject × task/session)  
2. A CSV file named:
   ```
   vmf_fixedmus_summary_K7.csv
   ```

Actual local paths are defined in `config.py` and are **not exposed** in this public repository.

---

## 5. Repository structure (Stage A)

```
.
├── run_vmf_pipeline.py      # MAIN entry point (run this)
├── config.py                # configuration and constants
├── vmf_npz.py               # loader for vMF .npz files
├── vmf_features.py          # feature extraction from P(t,k)
├── vmf_dataset.py           # dataset construction
├── outputs/                 # generated results
├── legacy/                  # archived / unused code
└── README.md
```
