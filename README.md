# \# EEG Panel Model with EM + Interactive Fixed Effects

# 

# This repository implements a \*\*factor-augmented panel model\*\* for multichannel EEG with \*\*missing covariates\*\*. It combines:

# \- Regression on \*\*subject-level\*\* covariates (sex, age, task),

# \- Regression on \*\*time-varying\*\* covariates (time-of-day harmonics),

# \- \*\*Directional / vMF features\*\* from spectral structure,

# \- \*\*Interactive Fixed Effects (IFE)\*\* (low-rank latent factors),

# \- \*\*EM-style\*\* imputations for missing sex/age/ToD,

# \- Optional \*\*heavy-tail robustness\*\* (Student-t / IRLS extension).

# 

# The model predicts the \*\*multichannel EEG signal\*\* (e.g., log-power per channel) for held-out sessions and evaluates in-sample and out-of-sample performance per subject.

# 

# ---

# 

# \## 🧩 Model Overview

# 

# For subject \\(d\\) and time \\(t\\):

# 

# \\\[

# \\by^{(d)}\_t

# = \\bmu^{(d)} + \\bC\_a \\ba^{(d)} + \\bC\_b \\bb^{(d)}\_t + \\bC\_z g(\\bz^{(d)}\_t;\\theta)

# \+ \\bLambda^{(d)} \\bff^{(d)}\_t + \\beps^{(d)}\_t

# \\]

# 

# | Symbol | Meaning |

# |:--|:--|

# | \\( \\by^{(d)}\_t \\in \\mathbb{R}^p \\) | Multichannel EEG (e.g., log-power) |

# | \\( \\ba^{(d)} \\) | Subject-level covariates (sex, age, task) |

# | \\( \\bb^{(d)}\_t \\) | Time-varying covariates (time-of-day harmonics) |

# | \\( \\bz^{(d)}\_t \\) | Directional features (vMF posteriors) |

# | \\( \\bff^{(d)}\_t \\) | Latent EEG factors |

# | \\( \\bLambda^{(d)} \\) | Channel loadings |

# | \\( \\beps^{(d)}\_t \\) | Noise or residual |

# 

# Missing covariates are handled by \*\*EM\*\*:

# \- E-step imputes sex/age/ToD from posteriors,

# \- M-step refits regression + factor structure.

# 

# ---

# 

# \## 📁 Repository Structure



