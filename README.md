EEG Panel Model with EM + Interactive Fixed Effects

This repository implements a factor-augmented panel model for multichannel EEG with missing covariates.
It combines:

Regression on subject-level covariates (sex, age, task)

Regression on time-varying covariates (time-of-day harmonics)

Directional / vMF features from spectral structure

Interactive Fixed Effects (IFE) (low-rank latent factors)

EM-style imputations for missing sex/age/ToD

Optional heavy-tail robustness (Student-t / IRLS extension)

The model predicts the multichannel EEG signal (e.g., log-power per channel) for held-out sessions and evaluates in-sample and out-of-sample performance per subject.

🧩 Model Overview

For subject d and time t:

𝑦
𝑡
(
𝑑
)
=
𝜇
(
𝑑
)
+
𝐶
𝑎
𝑎
(
𝑑
)
+
𝐶
𝑏
𝑏
𝑡
(
𝑑
)
+
𝐶
𝑧
𝑔
(
𝑧
𝑡
(
𝑑
)
;
𝜃
)
+
𝛬
(
𝑑
)
𝑓
𝑡
(
𝑑
)
+
𝜀
𝑡
(
𝑑
)
y
t
(d)
	​

=μ
(d)
+C
a
	​

a
(d)
+C
b
	​

b
t
(d)
	​

+C
z
	​

g(z
t
(d)
	​

;θ)+Λ
(d)
f
t
(d)
	​

+ε
t
(d)
	​

Symbol	Meaning
$\mathbf{y}^{(d)}_t \in \mathbb{R}^p$	Multichannel EEG (e.g., log-power)
$\mathbf{a}^{(d)}$	Subject-level covariates (sex, age, task)
$\mathbf{b}^{(d)}_t$	Time-varying covariates (time-of-day harmonics)
$\mathbf{z}^{(d)}_t$	Directional features (vMF posteriors)
$\mathbf{f}^{(d)}_t$	Latent EEG factors
$\boldsymbol{\Lambda}^{(d)}$	Channel loadings
$\boldsymbol{\varepsilon}^{(d)}_t$	Noise or residual

Missing covariates are handled by EM:

E-step: impute sex/age/ToD from posteriors

M-step: refit regression + factor structure

📁 Repository Structure
project_root/
│
├── config.py          # Constants: session names, subject metadata, hyperparams
├── dataset.py         # Load .npy tensor, concatenate sessions
├── directional.py     # vMF features: spherical k-means, posteriors
├── ife.py             # Interactive Fixed Effects + Bai–Ng rank selection
├── impute.py          # Posterior P(sex|·), ToD estimation grid
├── panel.py           # Builds y (EEG), B (ToD), Z (vMF), and masks
│
├── design.py          # [NEW] unified builders for A/B/Z blocks and masks
├── model.py           # [NEW] EEGPanelIFEMI class (fit/EM/predict)
├── run_refactored.py  # [NEW] entry point for subject-wise training/eval
│
├── clean_EC.npy       # (not tracked) large EEG tensor
└── results/           # optional output folder


Keep: core + new files
Remove: main.py, run_all.py, main.ipynb, and io_utils.py (if unused)

⚙️ Installation
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy scipy pandas scikit-learn tqdm


Optional:

pip install matplotlib joblib

🧠 Data Format

EEG tensor: clean_EC.npy
Shape: (subjects, sessions, channels, time), e.g. (12, 4, 32, 90000)

Each subject’s metadata (label, sex, age) is stored in config.SUBJECT_META:

[
  ("AM", 0, 24),
  ("CL", 0, 23),
  ("CQ", 1, 26),
  ...
]


Use memory-mapping for large files:

np.load("clean_EC.npy", mmap_mode="r")

▶️ Run the Pipeline

From the repo root:

python run_refactored.py

What it does

Loads the EEG tensor and metadata

Builds design matrices:

A: subject covariates (sex, age, task)

B: time-of-day harmonics

Z: vMF directional posteriors

Selects latent rank (Bai–Ng IC)

Fits IFE model with EM imputations

Evaluates train/test MSE and $R^2$

Saves summary_refactored.csv

📊 Example Output
[Done] AM: r=3, train={'mse': 2.53e-09, 'r2': 0.77}, test={'mse': 3.24e-09, 'r2': 0.49}
...
=== Summary (refactored) ===
 subject  rank  train_mse  train_r2  test_mse  test_r2
 AM         3   ...        0.77      ...       0.49
 CL         3   ...        0.77      ...       0.75


rank → chosen latent factor dimension

train_r2 / test_r2 → variance explained (fit & generalization)

High test $R^2$ = reproducible EEG structure; low = session variability

🧩 Core Modules
File	Purpose
model.py	EEGPanelIFEMI: full EM + IFE model, metrics
design.py	Builds A/B/Z with masks, handles missing
directional.py	vMF clustering and posterior features
ife.py	Factor decomposition + rank selection
impute.py	Posterior inference for sex/ToD
panel.py	Builds y, B, and Z blocks
dataset.py	Loads and concatenates .npy EEG data
config.py	Subject metadata and constants
🎯 What the Model Predicts

The model predicts multichannel EEG signals for unseen sessions:

𝑦
^
𝑡
(
𝑑
)
=
𝜇
(
𝑑
)
+
𝐶
𝑎
𝑎
(
𝑑
)
+
𝐶
𝑏
𝑏
𝑡
(
𝑑
)
+
𝐶
𝑧
𝑔
(
𝑧
𝑡
(
𝑑
)
)
+
𝛬
(
𝑑
)
𝑓
^
𝑡
(
𝑑
)
y
^
	​

t
(d)
	​

=μ
(d)
+C
a
	​

a
(d)
+C
b
	​

b
t
(d)
	​

+C
z
	​

g(z
t
(d)
	​

)+Λ
(d)
f
^
t
(d)
	​


where

𝑓
^
𝑡
(
𝑑
)
=
(
𝛬
(
𝑑
)
⊤
𝛬
(
𝑑
)
)
−
1
𝛬
(
𝑑
)
⊤
(
𝑦
𝑡
(
𝑑
)
−
covariate effects
)
f
^
t
(d)
	​

=(Λ
(d)⊤
Λ
(d)
)
−1
Λ
(d)⊤
(y
t
(d)
	​

−covariate effects)

Thus, predictions reflect expected EEG channel activity given subject traits, time-of-day, and learned latent factors.

💡 Interpretation

High test $R^2$ → EEG structure stable across sessions

Low test $R^2$ → strong non-stationarity or heavy-tailed noise

Rank $r$ ≈ number of dominant latent EEG factors per subject

🧪 Tips & Extensions

Use em_iters=2 (default); increase if many covariates are missing

Adjust r_grid=[1,2,3,4] if spectral gaps suggest more spikes

Use IRLS (t-robust) mode in ife.py for outlier control

Parallel fitting with joblib for multi-core systems

Add plots: predicted vs. actual EEG or factor loadings



