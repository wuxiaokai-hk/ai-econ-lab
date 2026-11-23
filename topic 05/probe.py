"""Step 05 — STARTER. Fill the three `# TODO (you):` blocks, then run
`python check_step05.py`. Reference: solution_probe.py (peek only after trying).

Linear probes on sentence embeddings, plus the sanity baselines that
interpretability claims must beat (shuffled labels, random directions).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ---------------------------------------------------------------------------
# 1. Logistic-regression probe, fitted by gradient descent
#    p = sigmoid(X w + b)
#    L(w,b) = -mean[y log p + (1-y) log(1-p)] + (lam/2)||w||^2
#    dL/dw = X'(p - y)/n + lam w ,   dL/db = mean(p - y)
# ---------------------------------------------------------------------------
def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float = 1e-2,
                 lr: float = 0.5, n_iter: int = 2000) -> tuple[np.ndarray, float]:
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(n_iter):
        # TODO (you): one gradient-descent step.
        #   1) p = predicted probabilities (use sigmoid)
        #   2) grad_w, grad_b from the formulas in the box above
        #   3) w <- w - lr*grad_w ; b <- b - lr*grad_b
        # Hint: X.T @ (p - y) / n is the same "X'e/n" you know from OLS.
        raise NotImplementedError("TODO 1: gradient step")
    return w, b


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)


# ---------------------------------------------------------------------------
# 2. Difference-of-means direction (the "no training" baseline)
#    d = mu_1 - mu_0 ; classify 1 if x.d > t, t = d.(mu_1 + mu_0)/2
# ---------------------------------------------------------------------------
def diff_of_means(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    # TODO (you): class means mu1 (y==1, hawkish) and mu0 (y==0, dovish),
    # direction d = mu1 - mu0, threshold t = d . (mu1 + mu0) / 2.
    # Hint: X[y == 1].mean(axis=0). Return (d, float(t)).
    # (This is LDA with an identity covariance — worth noting in an interview.)
    raise NotImplementedError("TODO 2: difference of means")


# ---------------------------------------------------------------------------
# 3. Permutation test: is the probe's accuracy better than chance *for this
#    data and this probe*?   p = (1 + #{null_b >= real}) / (1 + B)
# ---------------------------------------------------------------------------
def permutation_pvalue(score_fn: Callable[[np.ndarray, np.ndarray], float],
                       X: np.ndarray, y: np.ndarray, B: int = 200,
                       seed: int = 0) -> tuple[float, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    real = score_fn(X, y)
    # TODO (you): build the null distribution and the p-value.
    #   null[b] = score_fn(X, rng.permutation(y))   for b = 1..B
    #   p = (1 + #{null >= real}) / (1 + B)      <- the "+1" keeps p > 0
    # Return (real, null_as_np_array, float(p)).
    raise NotImplementedError("TODO 3: permutation test")


# ---------------------------------------------------------------------------
# Scaffolding (given in the starter too)
# ---------------------------------------------------------------------------
def kfold_accuracy(fit_predict: Callable, X: np.ndarray, y: np.ndarray,
                   k: int = 5, seed: int = 0) -> float:
    """Stratified k-fold accuracy. fit_predict(Xtr, ytr, Xte) -> 0/1 preds."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        folds[idx] = np.arange(len(idx)) % k
    correct = 0
    for f in range(k):
        te, tr = folds == f, folds != f
        correct += np.sum(fit_predict(X[tr], y[tr], X[te]) == y[te])
    return correct / len(y)


def logistic_fp(Xtr, ytr, Xte, lam=1e-2):
    w, b = fit_logistic(Xtr, ytr, lam=lam)
    return (predict_proba(Xte, w, b) > 0.5).astype(int)


def dom_fp(Xtr, ytr, Xte):
    d, t = diff_of_means(Xtr, ytr)
    return (Xte @ d > t).astype(int)


def random_direction_accuracy(X, y, k: int, n_draws: int = 20, seed: int = 0):
    """Probe on k RANDOM directions of the embedding space (frozen, untrained).
    If this is close to the full probe, the 'concept' is not special."""
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_draws):
        R = rng.standard_normal((X.shape[1], k))
        R /= np.linalg.norm(R, axis=0)
        Z = X @ R
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        accs.append(kfold_accuracy(logistic_fp, Z, y))
    return float(np.mean(accs)), float(np.std(accs))
