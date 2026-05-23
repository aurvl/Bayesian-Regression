# Bayesian Linear Regression in Econometrics

<p align="center">
  <img src="https://img.shields.io/badge/Bayesian-111827?style=for-the-badge" alt="Bayesian">
  <img src="https://img.shields.io/badge/Linear_Regression-2563EB?style=for-the-badge" alt="Linear Regression">
  <img src="https://img.shields.io/badge/Ridge-7C3AED?style=for-the-badge" alt="Ridge">
  <img src="https://img.shields.io/badge/Empirical_Bayes-0F766E?style=for-the-badge" alt="Empirical Bayes">
  <img src="https://img.shields.io/badge/Econometrics-0891B2?style=for-the-badge" alt="Econometrics">
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white&style=for-the-badge" alt="NumPy">
</p>

<p align="center">
  <img src="img/rd_coefficient_comparison.png" alt="R&D coefficient: OLS/Ridge point estimates vs Bayesian posteriors" width="650">
</p>

A from-scratch derivation and implementation of **Bayesian linear regression** and
its relatives (OLS, Ridge, Empirical Bayes), placed on a single common axis. The
project shows that a flat prior recovers OLS, a zero-mean Gaussian prior is exactly
Ridge, and Empirical Bayes learns the prior strength from the data by maximizing the
marginal likelihood. A real-world economic dataset (innovation, R&D, firm
performance) serves only as an illustration to make the shrinkage and
uncertainty-quantification mechanics visible.

## Repository structure

```
.
├── data/
│   ├── raw/                 # original WBES indicator files
│   └── processed/           # cleaned cross-section + data dictionary
├── img/                     # figures used in the notebook & README
├── modules.py               # OLS / Ridge / Bayesian / Empirical Bayes classes
├── notebook.ipynb           # main analysis (derivations + experiments)
├── cholesky_decomposition.md
└── README.md
```

## Objective

Classical OLS becomes unstable with small samples, correlated regressors, or high
estimation variance. The goal is **methodological**: derive and implement a Bayesian
framework that treats coefficients as distributions, quantifies their uncertainty
(credible intervals), and regularizes weak, poorly identified coefficients via
shrinkage — then show how OLS, Ridge, and Empirical Bayes are all special cases of
the same idea.

## Data & Stack

- **Data:** World Bank Enterprise Surveys (WBES) 2023, cross-section of economies.
  Target = real annual sales growth; predictors = R&D spending and product
  innovation (firm shares, %).
- **Stack:** Python, NumPy, SciPy, pandas, matplotlib / seaborn, Plotly. All
  estimators implemented from scratch (no statsmodels / scikit-learn fitting).

## Results

Using the R&D coefficient as a running example:

| Method | R&D $\hat\beta$ | Std. Error |
|---|---|---|
| OLS | −0.121 | 0.129 |
| Ridge ($\lambda=1$) | −0.121 | 0.129 |
| Bayesian | −0.031 | 0.117 |
| Empirical Bayes | −0.061 | 0.125 |

Shrinkage pulls the weak, noisy R&D coefficient toward the prior (−0.12 → −0.03)
while the strong product-innovation effect stays stable, and all standard errors
shrink — trading a little in-sample fit for more robust uncertainty.

## How to run

```bash
pip install numpy pandas scipy matplotlib seaborn plotly
jupyter notebook notebook.ipynb
```

Run the cells top to bottom. Models can also be imported directly:

```python
from modules import OLSRegression, BayesianRegression, RidgeRegression, EmpiricalBayesianRegression
```

## Notes

- The economic dataset is illustrative only — the focus is the statistical method,
  not an economic conclusion.
- Empirical Bayes estimates a point value of $\tau^2$ from the data; it is not a
  fully Bayesian treatment (no hyperprior), and can overfit if used carelessly.
- The log marginal likelihood uses a Cholesky factorization for numerical stability
  (see `cholesky_decomposition.md`).
