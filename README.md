# Dynamic EEG Modeling  
## Stage A: vMF Time-Series Analysis (Before Raw EEG)

---

## 1. What is this project?

This repository is part of a larger research program on **dynamic EEG modeling** that combines ideas from:

- time-series analysis  
- panel data econometrics  
- latent factor models  
- neuroscience  

The **full project** aims to model multichannel EEG signals using **panel vector autoregressions (PVARs)** with:
- subject heterogeneity  
- time-varying dynamics  
- latent brain states  
- covariates derived from EEG structure  

This repository currently implements **Stage A**, which is a **self-contained mini-project** and a foundation for later stages.

---

## 2. What is Stage A?

### Stage A = vMF-only analysis (no raw EEG amplitudes)

In Stage A, we **do not use raw EEG signals** (voltages, bandpower, etc.).

Instead, we start from **directional EEG representations** produced by a  
**von Mises–Fisher (vMF) mixture model** applied to windowed EEG data.

Each subject has a **vMF posterior probability time series** that describes how
latent EEG states evolve over time.

### Stage A as a mini-project

Stage A is designed as a **warm-up research project** whose goals are:

1. Learn how to work with **multivariate time series** derived from EEG  
2. Build intuition about **latent EEG state dynamics**  
3. Engineer **interpretable dynamic features**  
4. Perform **proper, leakage-free prediction**  
5. Evaluate predictive performance rigorously  

Stage A stands on its own and is suitable for:
- undergraduate research
- posters or short reports
- validation of EEG representations

Nothing in Stage A is throwaway.

---

## 3. Conceptual model behind Stage A

For each subject \( i \) and time window \( t \):

- We observe a **vMF posterior probability vector**
  \[
  Z_{it} = (P_{it,1}, \dots, P_{it,K}) \in \Delta^{K-1}
  \]
- Here, \(K = 7\) latent vMF states  
- \(Z_{it}\) evolves over time and reflects changing latent brain configurations  

### What we do with \(Z_{it}\)

1. Treat \(Z_{it}\) as a **time series**  
2. Extract **dynamic features**:
   - entropy (state uncertainty)  
   - switching rate  
   - volatility  
   - state occupancy  
   - transition structure  
3. Aggregate these features to the **subject level**  
4. Use them to **predict subject-level outcomes** using cross-validation  

This corresponds to building and validating the **covariate component** of the
full EEG model.

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

### 4.1 vMF `.npz` file contents

Each `.npz` file contains:

| Key | Description |
|----|-------------|
| `P` | Array `(T, K)` of vMF posterior probabilities |
| `kappa` | Concentration parameters |
| `logalpha` | Mixture weights |
| `mus_fixed` | Component templates over EEG channels |
| `subject` | Subject identifier |
| `task` | Task or session identifier |
| `sfreq` | Sampling frequency |
| `ch_names` | EEG channel names |

Important:
- `P` is the **main object** used in Stage A  
- Each row of `P` sums to 1  
- `T` can be large (tens of thousands of time points)

---

### 4.2 CSV file contents

`vmf_fixedmus_summary_K7.csv` contains metadata and labels:

| Column | Meaning |
|------|--------|
| `probabilities_file` | Path to the `.npz` file |
| `attention` | Subject-level attention score |
| `p_factor` | General psychopathology factor |
| `internalizing` | (optional) |
| `externalizing` | (optional) |

Paths may be machine-specific; **only the filename is used**.

---

## 5. Repository structure (Stage A)

```
.
├── run_vmf_pipeline.py          # Stage A feature engineering
├── stageA_predict.py            # Stage A prediction + evaluation
├── config.py                    # configuration and constants
├── vmf_npz.py                   # loader for vMF .npz files
├── vmf_features.py              # feature extraction from P(t,k)
├── vmf_dataset.py               # dataset construction
├── notebooks/
│   └── 00_vmf_results.ipynb     # visualization & reporting only
├── outputs/                     # generated results
├── legacy/                      # archived / unused code
└── README.md
```

---

## 6. How to run Stage A (recommended workflow)

Stage A is intentionally split into **two steps**:
1. feature engineering  
2. prediction & evaluation  

### Step 1: Build subject-level features

```bash
python run_vmf_pipeline.py
```

This produces:

```
outputs/vmf_subject_features.csv
```

This file is **not a prediction** — it is a **feature-engineered dataset**.

---

### Step 2: Run Stage A prediction

```bash
python stageA_predict.py
```

This produces:

```
outputs/stageA_metrics.csv
outputs/stageA_oof_predictions.csv
outputs/stageA_model_info.json
```

---

### Step 3: Visualize results

```bash
jupyter notebook
```

Open:

```
notebooks/00_vmf_results.ipynb
```

The notebook loads outputs and generates plots. It does not rerun the pipeline.

---

## 7. Outputs (Stage A)

All outputs live in:

```
outputs/
```

Key files:

| File | Meaning |
|----|--------|
| `vmf_subject_features.csv` | Subject-level features |
| `stageA_metrics.csv` | Cross-validated performance metrics |
| `stageA_oof_predictions.csv` | Out-of-fold predictions |
| `stageA_model_info.json` | Model configuration snapshot |

---

## 8. Interpretation of Stage A results

Stage A answers:

> Does the temporal organization of latent EEG states carry cognitive information?

Indicators of success:
- positive out-of-sample R²  
- meaningful true vs predicted plots  
- interpretable associations  

---

## 9. What Stage A is not

Stage A does **not**:
- predict EEG time series  
- estimate VAR coefficients  
- infer connectivity  

These belong to **Stage B**.

---

## 10. Next stage (Stage B)

Stage B introduces:
- raw EEG time series  
- VAR / VARX models  
- latent factors  

Stage A remains unchanged and reusable.
