"""Regression models and helpers used in the Bayesian regression notebook.

Contains OLS, Ridge, Bayesian and Empirical Bayes linear regression
implementations (from scratch), plus a 3D regression-plot helper.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import minimize_scalar


class OLSRegression:
    def __init__(self, X, y):
        self.beta = None
        self.X = X
        self.y = y
        self.sigma_squared = None

    def fit(self):
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y

    def summary(self, feature_names=None):
        if self.beta is None:
            raise ValueError("Model not fitted yet.")

        n, k = self.X.shape

        y_pred = self.X @ self.beta
        residuals = self.y - y_pred

        # sums of squares
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.y - np.mean(self.y))**2)
        ss_reg = ss_tot - ss_res

        # goodness of fit
        r_squared = 1 - ss_res / ss_tot
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)

        # residual variance
        sigma_squared = ss_res / (n - k)
        self.sigma_squared = sigma_squared

        # variance covariance matrix
        var_beta = sigma_squared * np.linalg.inv(self.X.T @ self.X)

        # std errors
        se_beta = np.sqrt(np.diag(var_beta))

        # t-stats
        t_stats = self.beta / se_beta

        # p-values
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

        # F-stat
        f_stat = (ss_reg / (k - 1)) / (ss_res / (n - k))
        f_pvalue = 1 - stats.f.cdf(f_stat, k - 1, n - k)

        # feature names
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(k)]

        # coefficients table
        coef_table = pd.DataFrame({
            "Estimate": self.beta,
            "Std. Error": se_beta,
            "t value": t_stats,
            "Pr(>|t|)": p_values
        }, index=feature_names)

        print()
        print("=" * 70)
        print("OLS Regression Results")
        print("=" * 70)
        print(coef_table.round(4))
        print("-" * 70)
        print(f"Residual standard error: {np.sqrt(sigma_squared):.4f} on {n-k} DF")
        print(f"Multiple R-squared: {r_squared:.4f}")
        print(f"Adjusted R-squared: {adj_r_squared:.4f}")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"Prob (F-statistic): {f_pvalue:.4g}")

    def get_coefficients(self):
        if self.beta is None:
            raise ValueError("Model not fitted yet.")
        return self.beta, self.sigma_squared


class BayesianRegression:
    def __init__(self, X, y, beta_prior_mean, beta_prior_cov):
        self.X = X
        self.y = y
        self.beta_prior_mean = beta_prior_mean
        self.beta_prior_cov = beta_prior_cov
        self.beta_posterior_mean = None
        self.beta_posterior_cov = None

    def fit(self):
        sigma_squared = np.var(self.y - self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y)
        XTX = self.X.T @ self.X
        XTy = self.X.T @ self.y
        self.beta_posterior_cov = np.linalg.inv(XTX / sigma_squared + np.linalg.inv(self.beta_prior_cov))
        self.beta_posterior_mean = self.beta_posterior_cov @ (XTy / sigma_squared + np.linalg.inv(self.beta_prior_cov) @ self.beta_prior_mean)

    def summary(self, feature_names=None):
        if self.beta_posterior_mean is None or self.beta_posterior_cov is None:
            raise ValueError("Model not fitted yet.")

        # building summary table
        se_posterior = np.sqrt(np.diag(self.beta_posterior_cov))
        t_stats = self.beta_posterior_mean / se_posterior
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

        # metrics
        n = self.y.shape[0]
        y_pred = self.X @ self.beta_posterior_mean
        residuals = self.y - y_pred

        # sums of squares
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.y - np.mean(self.y))**2)

        # goodness of fit
        r_squared = 1 - ss_res / ss_tot
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - self.X.shape[1])

        # F-statistic
        f_stat = (ss_tot - ss_res) / (self.X.shape[1] - 1) / (ss_res / (n - self.X.shape[1]))
        f_pvalue = 1 - stats.f.cdf(f_stat, self.X.shape[1] - 1, n - self.X.shape[1])

        # feature names
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.X.shape[1])]
        summary_table = pd.DataFrame({
            "Posterior Mean": self.beta_posterior_mean,
            "Std. Error": se_posterior,
            "t value": t_stats,
            "Pr(>|t|)": p_values
        }, index=feature_names)
        print()
        print("=" * 70)
        print("Bayesian Regression Results")
        print("=" * 70)
        print(summary_table.round(4))
        print("-" * 70)
        print(f"Residual standard error: {np.sqrt(ss_res / (n - self.X.shape[1])):.4f} on {n - self.X.shape[1]} DF")
        print(f"Multiple R-squared: {r_squared:.4f}")
        print(f"Adjusted R-squared: {adj_r_squared:.4f}")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"Prob (F-statistic): {f_pvalue:.4g}")
        print(f"Posterior covariance matrix:\n{self.beta_posterior_cov}")

    def get_posterior(self):
        if self.beta_posterior_mean is None or self.beta_posterior_cov is None:
            raise ValueError("Model not fitted yet.")
        return self.beta_posterior_mean, self.beta_posterior_cov


class RidgeRegression:
    def __init__(self, X, y, lambda_):
        self.X = X
        self.y = y
        self.lambda_ = lambda_
        self.beta = None
        self.sigma_squared = None
        self.standard_errors = None

    def fit(self):
        n, k = self.X.shape
        I = np.eye(k)
        I[0, 0] = 0  # To not penalize the intercept
        self.beta = np.linalg.inv(self.X.T @ self.X + self.lambda_ * I) @ self.X.T @ self.y
        self.sigma_squared = np.sum((self.y - self.X @ self.beta)**2) / (n - k)
        self.standard_errors = np.sqrt(np.diag(np.linalg.inv(self.X.T @ self.X + self.lambda_ * I) * self.sigma_squared))

    def summary(self, feature_names=None):
        if self.beta is None:
            raise ValueError("Model not fitted yet.")

        # feature names
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.X.shape[1])]

        coef_table = pd.DataFrame({
            "Coefficient": self.beta,
            "Standard Error": self.standard_errors
        }, index=feature_names)

        print()
        print("=" * 70)
        print(f"Ridge Regression Results (lambda={self.lambda_})")
        print("=" * 70)
        print(coef_table.round(4))

    def get_coefficients(self):
        if self.beta is None:
            raise ValueError("Model not fitted yet.")
        return self.beta, self.sigma_squared


class EmpiricalBayesianRegression:
    def __init__(self, X, y, sigma_squared=None):
        self.X = X
        self.y = y
        self.beta = None
        self.cov = None
        self.optimal_tau2 = None
        self.lml = self.log_marginal_likelihood
        self.sigma_squared = sigma_squared
        if self.sigma_squared is None:
            beta_ols = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
            residuals = self.y - self.X @ beta_ols
            n, k = self.X.shape
            self.sigma_squared = (residuals @ residuals) / (n - k)

    def log_marginal_likelihood(self, tau2):
        n = self.X.shape[0]
        sigma2 = self.sigma_squared
        sigma_y = sigma2 * np.eye(n) + tau2 * (self.X @ self.X.T)
        # Cholesky for log-det and quadratic form (cf. cholesky_decomposition.md)
        jitter = 1e-8
        L = np.linalg.cholesky(sigma_y + jitter * np.eye(n))
        alpha = np.linalg.solve(L, self.y)
        quad = alpha @ alpha
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (n * np.log(2.0 * np.pi) + log_det + quad)

    def optimize(self, tau2_range=[1e-6, 100], summary=False):
        res = minimize_scalar(lambda t: -self.log_marginal_likelihood(t), bounds=tau2_range, method='bounded')

        if summary:
            print(f"Optimal tau^2: {res.x:.6f}")
            print(f"Log marginal likelihood at optimal tau^2: {-res.fun:.4f}")

        self.optimal_tau2 = res.x
        return self.optimal_tau2

    def fit(self):
        tau2 = self.optimal_tau2
        if tau2 is None:
            raise ValueError("Call optimize() before fit().")
        XTX = self.X.T @ self.X
        XTy = self.X.T @ self.y
        k = self.X.shape[1]
        sigma2 = self.sigma_squared
        prior_prec = np.eye(k) / tau2
        self.cov = np.linalg.inv(XTX / sigma2 + prior_prec)
        self.beta = self.cov @ (XTy / sigma2)

    def predict(self, X_new):
        if self.beta is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return X_new @ self.beta

    def summary(self, feature_names=None):
        if self.beta is None or self.cov is None:
            raise ValueError("Model not fitted yet.")

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.X.shape[1])]

        se_beta = np.sqrt(np.diag(self.cov))
        coef_table = pd.DataFrame({
            "Coefficient": self.beta,
            "Std. Error": se_beta
        }, index=feature_names)

        print()
        print("=" * 70)
        print("Empirical Bayesian Regression Results")
        print("=" * 70)
        print(coef_table.round(4))
        print(f"Posterior covariance matrix:\n{self.cov}")


def regression_plot(df, x_col, y_col, z_col, beta, grid_size=20, title=None, labels=None):
    if labels is None:
        labels = {
            x_col: x_col,
            y_col: y_col,
            z_col: z_col,
        }
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        title=title,
        labels=labels,
    )
    x_range = np.linspace(df[x_col].min(), df[x_col].max(), grid_size)
    y_range = np.linspace(df[y_col].min(), df[y_col].max(), grid_size)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = beta[0] + beta[1] * x_grid + beta[2] * y_grid
    surface = go.Surface(x=x_range, y=y_range, z=z_grid, opacity=0.5, showscale=False)
    fig.add_traces([surface])
    return fig
