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

```text
<VMF_DATA_DIR>/
```

This directory must contain:

1. **vMF `.npz` files** (one per subject × task/session)  
2. A CSV file named:
   ```text
   vmf_fixedmus_summary_K7.csv
   ```

Actual local paths are defined in `config.py` and are **not exposed** in this public repository.

---

### 4.1 vMF `.npz` file contents

Each `.npz` file contains:

| Key | Description |
|----|-------------|
| `P` | Array of shape `(T, K)` with vMF posterior probabilities |
| `kappa` | Concentration parameters |
| `logalpha` | Mixture weights |
| `mus_fixed` | Component templates over EEG channels |
| `subject` | Subject identifier |
| `task` | Task or session identifier |
| `sfreq` | Sampling frequency |
| `ch_names` | EEG channel names |

Important points:
- `P` is the **main object** used in Stage A  
- Each row of `P` sums to 1 (probability simplex)  
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

Paths in `probabilities_file` may be machine-specific;  
**only the filename is used** to locate `.npz` files.

---

## 5. Repository structure (Stage A)

```text
.
├── run_vmf_pipeline.py      # MAIN entry point (run this)
├── config.py                # configuration and constants
├── vmf_npz.py               # loader for vMF .npz files
├── vmf_features.py          # feature extraction from P(t,k)
├── vmf_dataset.py           # dataset construction
├── notebooks/               # visualization + reporting (no pipeline logic)
│   └── 00_vmf_results.ipynb
├── outputs/                 # generated results (created automatically)
├── legacy/                  # archived / unused code
└── README.md
```

**Rule:**  
👉 Only `run_vmf_pipeline.py` should be executed directly.

---

## 6. How to run Stage A (recommended workflow)

Stage A is run as a **script-first pipeline**. The notebook is used only for visualization.

### Step 1: Run the pipeline (from repo root)

```bash
python run_vmf_pipeline.py
```

This will:
- load the CSV + vMF `.npz` files,
- extract dynamic features from `P(t,k)`,
- aggregate features to subject level,
- write results to:

```text
outputs/vmf_subject_features.csv
```

### Step 2: Visualize results in the notebook

Open Jupyter from the repo root:

```bash
jupyter notebook
```

Then open:

```text
notebooks/00_vmf_results.ipynb
```

The notebook loads the output CSV and produces graphs (histograms, scatter plots, correlations).

> Important: If you change feature extraction, edit `vmf_features.py` and re-run the pipeline.
> Do not duplicate pipeline logic inside the notebook.

---

## 7. File-by-file explanation (important for students)

### 7.1 `run_vmf_pipeline.py` (MAIN FILE)

This is the **only file you run**.

What it does:
1. Loads the CSV and vMF `.npz` files  
2. Extracts dynamic features from vMF time series  
3. Aggregates features to the subject level  
4. Saves results to `outputs/`  
5. Runs predictive models using subject-wise cross-validation  
6. Prints performance metrics (R², MSE)

---

### 7.2 `config.py`

Central configuration file.

Defines:
- alias `<VMF_DATA_DIR>`  
- CSV filename  
- number of vMF components (`K = 7`)  
- output directory  
- random seed and CV settings  

---

### 7.3 `vmf_npz.py`

Low-level data loader.

Responsibilities:
- Load a single vMF `.npz` file  
- Validate the probability matrix `P`  
- Ensure correct shape `(T, K)`  
- Extract metadata (`subject`, `task`, etc.)  

---

### 7.4 `vmf_features.py`

Scientific core of Stage A.

Given a vMF time series `P(t,k)`, it computes:
- occupancy (mean probability per state)  
- entropy (mean/std)  
- switching rate  
- volatility  
- transition structure  
- confidence dynamics  

---

### 7.5 `vmf_dataset.py`

Dataset construction layer.

What it does:
1. Reads the CSV file  
2. Maps CSV paths to local `.npz` filenames  
3. Loads vMF time series  
4. Extracts features from each time series  
5. Builds a clean table  
6. Aggregates results to the subject level  

---

### 7.6 `notebooks/00_vmf_results.ipynb`

Visualization and reporting only.

It:
- loads `outputs/vmf_subject_features.csv`
- generates plots:
  - feature distributions
  - feature vs target scatters
  - correlations
  - quick sanity baseline fits

It does **not**:
- load `.npz` directly
- recompute features
- replace the pipeline

---

## 8. Outputs

All outputs are written to:

```text
outputs/
```

Primary output file:

```text
vmf_subject_features.csv
```

---

## 9. Interpretation of Stage A results

If Stage A works well, you should observe:
- positive out-of-sample R² for `attention` and/or `p_factor`,
- meaningful associations with entropy, switching, and occupancy.

Key message:

> Even without raw EEG amplitudes, the temporal organization of latent vMF states carries substantial cognitive information.

---

## 10. Next stage (Stage B)

When raw EEG data arrive:
- window EEG into multichannel time series,
- reintroduce VAR models,
- add latent factors and time-varying dynamics.

Stage A remains unchanged and fully reusable.
