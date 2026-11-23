"""Step 07 STARTER — fill the 4 `# TODO (you):` blocks.

Idea of the day: a tabular foundation model (TabPFN) predicts by *conditioning* on the training
table (in-context learning) instead of *fitting* parameters. What it approximates is the Bayesian
posterior predictive
        p(y* | x*, D) = ∫ p(y* | x*, θ) p(θ | D) dθ.
For a linear-Gaussian prior that integral has a closed form, so we can write a "PFN by hand"
and compare it with the old way (OLS / gradient-boosted trees fitted per dataset).

Running before you fill the TODOs raises NotImplementedError — that's expected.
Check yourself:   python check_step07.py
Reference run:    python run_step07.py solution
"""
from __future__ import annotations

import numpy as np

QUANTILES = (0.1, 0.5, 0.9)


def context_mask(t_rows: np.ndarray, origin: int, h: int) -> np.ndarray:
    """Rows usable as training context when standing at month `origin` (boolean array).

    Row s has target pi_{s+h} - pi_s, which is only *observed* at month s+h.
    """
    # TODO (you): return a boolean mask over t_rows.
    # Hint: a row is usable only if its target is already known at `origin`.
    #       Using s <= origin instead is the classic look-ahead leak (h-1 rows peek into the future).
    raise NotImplementedError("TODO 1: context_mask")


def log_marginal_likelihood(X: np.ndarray, y: np.ndarray, tau2: float, sigma2: float) -> float:
    """log N(y; 0, K),  K = sigma2*I + tau2*X X^T   (X standardized, y centered).

    Model: y = X beta + e, beta ~ N(0, tau2 I), e ~ N(0, sigma2 I). Integrating beta out gives K.
    """
    # TODO (you): compute the Gaussian log density of y with covariance K.
    # Hint: L = np.linalg.cholesky(K);  a = np.linalg.solve(L, y)  ->  y^T K^{-1} y = a @ a
    #       log|K| = 2 * sum(log(diag(L)))
    #       log p  = -0.5*a@a - 0.5*log|K| - 0.5*n*log(2*pi)
    raise NotImplementedError("TODO 2: log_marginal_likelihood")


def bayes_predictive(X: np.ndarray, y: np.ndarray, Xq: np.ndarray, tau2: float, sigma2: float):
    """Posterior predictive (mean, variance) at query rows Xq. Returns two 1-D arrays."""
    # TODO (you): the closed-form "in-context" prediction.
    # Hint:  A     = X^T X / sigma2 + I / tau2
    #        Sigma = inv(A)                 (posterior cov of beta)
    #        mu    = Sigma X^T y / sigma2   (posterior mean of beta)
    #        mean  = Xq mu
    #        var   = sigma2 + diag(Xq Sigma Xq^T)   <- np.einsum("ij,jk,ik->i", Xq, Sigma, Xq)
    raise NotImplementedError("TODO 3: bayes_predictive")


def pinball_loss(y: np.ndarray, qhat: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss of predicted q-quantile qhat."""
    # TODO (you): u = y - qhat;  rho_q(u) = max(q*u, (q-1)*u);  return the mean.
    # Hint: for q=0.5 this is 0.5*MAE. Under-predicting a high quantile is cheap, over... check signs!
    raise NotImplementedError("TODO 4: pinball_loss")


# ================================================================ scaffolding (done for you)
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
