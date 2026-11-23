"""
ai-econ-lab — Step 03 REFERENCE SOLUTION: split conformal vs adaptive conformal (ACI)

Try step03/conformal.py first. Then compare with this file.

Ideas:
  score        s_t = |y_t - yhat_t|
  split CP     q = k-th smallest of the last n scores, k = ceil((n+1)(1-alpha))
               interval = yhat_t ± q
  ACI          alpha_{t+1} = alpha_t + gamma * (alpha - err_t),  err_t = 1{y_t outside}
               (Gibbs & Candès 2021, NeurIPS)  -> long-run miss rate -> alpha
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# TODO 1 (solution): finite-sample conformal quantile
# ---------------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the k-th smallest score, k = ceil((n+1)(1-alpha)).

    - alpha <= 0  -> np.inf   (we demand 100% coverage: infinite interval)
    - alpha >= 1  -> 0.0      (we accept always missing: point interval)
    - k > n       -> np.inf   (too few calibration points for this alpha)
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if alpha <= 0:
        return np.inf
    if alpha >= 1:
        return 0.0
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return np.inf
    if k < 1:
        return 0.0
    return float(np.sort(scores)[k - 1])


# ---------------------------------------------------------------------------
# TODO 2 (solution): rolling split conformal with a FIXED alpha
# ---------------------------------------------------------------------------
def split_conformal_intervals(y, yhat, alpha: float = 0.1, window: int = 60):
    """Online split conformal: at each t, calibrate on the previous `window`
    absolute residuals (only information available before t).

    Returns (lo, hi, miss) arrays of len(y); NaN before the first full window.
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    T = y.size
    scores = np.abs(y - yhat)
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    miss = np.full(T, np.nan)
    for t in range(window, T):
        q = conformal_quantile(scores[t - window:t], alpha)
        lo[t], hi[t] = yhat[t] - q, yhat[t] + q
        miss[t] = float(not (lo[t] <= y[t] <= hi[t]))
    return lo, hi, miss


# ---------------------------------------------------------------------------
# TODO 3 (solution): adaptive conformal inference (ACI)
# ---------------------------------------------------------------------------
def aci_intervals(y, yhat, alpha: float = 0.1, gamma: float = 0.02, window: int = 60):
    """Same as split conformal, but the working level alpha_t is updated
    after each outcome:  alpha_{t+1} = alpha_t + gamma * (alpha - err_t).

    A miss (err=1) lowers alpha_t  -> wider next interval.
    A hit  (err=0) raises alpha_t a little -> narrower next interval.

    Returns (lo, hi, miss, alpha_path).
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    T = y.size
    scores = np.abs(y - yhat)
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    miss = np.full(T, np.nan)
    alpha_path = np.full(T, np.nan)
    alpha_t = alpha
    for t in range(window, T):
        alpha_path[t] = alpha_t
        q = conformal_quantile(scores[t - window:t], alpha_t)
        lo[t], hi[t] = yhat[t] - q, yhat[t] + q
        err = float(not (lo[t] <= y[t] <= hi[t]))
        miss[t] = err
        alpha_t = alpha_t + gamma * (alpha - err)
    return lo, hi, miss, alpha_path


# ---------------------------------------------------------------------------
# Scaffolding (same in starter): report by regime
# ---------------------------------------------------------------------------
def coverage_table(dates, y, lo, hi, miss, regimes: dict[str, tuple[str, str]]):
    """Coverage and median width per named date regime."""
    import pandas as pd

    df = pd.DataFrame({"date": pd.to_datetime(dates), "y": y, "lo": lo, "hi": hi, "miss": miss})
    df = df.dropna(subset=["miss"])
    rows = []
    for name, (start, end) in regimes.items():
        d = df[(df.date >= start) & (df.date <= end)]
        if d.empty:
            continue
        width = (d.hi - d.lo).replace([np.inf], np.nan)
        rows.append({
            "regime": name,
            "n": len(d),
            "coverage": 1 - d.miss.mean(),
            "median_width_pp": width.median(),
            "share_infinite": float(np.isinf(d.hi - d.lo).mean()),
        })
    return pd.DataFrame(rows)
