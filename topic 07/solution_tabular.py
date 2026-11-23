"""Step 07 REFERENCE SOLUTION — try tabular.py first, then compare.

Idea of the day: a tabular foundation model (TabPFN) predicts by *conditioning* on the training
table (in-context learning) instead of *fitting* parameters. What it approximates is the Bayesian
posterior predictive
        p(y* | x*, D) = ∫ p(y* | x*, θ) p(θ | D) dθ.
For a linear-Gaussian prior that integral has a closed form, so we can write a "PFN by hand"
and compare it with the old way (OLS / gradient-boosted trees fitted per dataset).
"""
from __future__ import annotations

import numpy as np

QUANTILES = (0.1, 0.5, 0.9)


# ---------------------------------------------------------------- TODO 1: leakage rule
def context_mask(t_rows: np.ndarray, origin: int, h: int) -> np.ndarray:
    """Rows usable as training context when standing at month `origin`.

    Row s has target pi_{s+h} - pi_s, which is only *observed* at month s+h.
    So s is usable iff s + h <= origin.
    """
    return t_rows + h <= origin


# ---------------------------------------------------------------- TODO 2: evidence
def log_marginal_likelihood(X: np.ndarray, y: np.ndarray, tau2: float, sigma2: float) -> float:
    """log N(y; 0, sigma2*I + tau2*X X^T)  for  y = X beta + e, beta~N(0,tau2 I), e~N(0,sigma2 I).

    (X standardized, y centered.) Uses a Cholesky factor K = L L^T:
        log p = -1/2 y^T K^{-1} y - sum(log diag L) - n/2 log(2 pi)
    """
    n = len(y)
    K = sigma2 * np.eye(n) + tau2 * X @ X.T
    L = np.linalg.cholesky(K)
    a = np.linalg.solve(L, y)          # a = L^{-1} y  ->  y^T K^{-1} y = a^T a
    return float(-0.5 * a @ a - np.log(np.diag(L)).sum() - 0.5 * n * np.log(2 * np.pi))


# ---------------------------------------------------------------- TODO 3: posterior predictive
def bayes_predictive(X: np.ndarray, y: np.ndarray, Xq: np.ndarray, tau2: float, sigma2: float):
    """Posterior predictive mean and variance at query rows Xq.

        A      = X^T X / sigma2 + I / tau2
        Sigma  = A^{-1}                       (posterior covariance of beta)
        mu     = Sigma X^T y / sigma2         (posterior mean of beta)
        mean*  = Xq mu
        var*   = sigma2 + diag(Xq Sigma Xq^T) (noise + parameter uncertainty)
    """
    p = X.shape[1]
    A = X.T @ X / sigma2 + np.eye(p) / tau2
    Sigma = np.linalg.inv(A)
    mu = Sigma @ X.T @ y / sigma2
    mean = Xq @ mu
    var = sigma2 + np.einsum("ij,jk,ik->i", Xq, Sigma, Xq)
    return mean, var


# ---------------------------------------------------------------- TODO 4: scoring rule
def pinball_loss(y: np.ndarray, qhat: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss:  rho_q(u) = max(q*u, (q-1)*u),  u = y - qhat."""
    u = y - qhat
    return float(np.mean(np.maximum(q * u, (q - 1) * u)))


# ================================================================ scaffolding (same in starter)
def standardize(Xc: np.ndarray, Xq: np.ndarray):
    """Scale with CONTEXT statistics only (query rows never touch mu/sd)."""
    mu, sd = Xc.mean(0), Xc.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xc - mu) / sd, (Xq - mu) / sd


def empirical_bayes(X: np.ndarray, y: np.ndarray):
    """Pick (tau2, sigma2) on a small grid by maximizing the evidence (type-II ML)."""
    vy = float(np.var(y)) + 1e-9
    best = None
    for tau2 in (0.003, 0.01, 0.03, 0.1, 0.3):
        for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
            sigma2 = frac * vy
            lml = log_marginal_likelihood(X, y, tau2 * vy, sigma2)
            if best is None or lml > best[0]:
                best = (lml, tau2 * vy, sigma2)
    return best[1], best[2]


def gaussian_quantiles(mean: np.ndarray, var: np.ndarray, qs=QUANTILES) -> np.ndarray:
    from scipy.stats import norm
    sd = np.sqrt(var)
    return np.stack([mean + norm.ppf(q) * sd for q in qs], axis=1)
