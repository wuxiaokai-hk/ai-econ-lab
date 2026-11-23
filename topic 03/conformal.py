"""
ai-econ-lab — Step 03: put guaranteed-coverage intervals around ANY forecaster
                       (split conformal vs adaptive conformal inference, ACI)

You fill in 3 TODO blocks (~20–30 min). Then run:
    python step03/check_step03.py          # unit checks (no internet needed)
    python step03/run_step03.py            # real Canadian CPI from BoC Valet

Ideas you are implementing:
  score        s_t = |y_t - yhat_t|                 (how wrong the forecast was)
  split CP     q = k-th smallest of the last n scores, k = ceil((n+1)(1-alpha))
               interval_t = yhat_t ± q
  ACI          alpha_{t+1} = alpha_t + gamma * (alpha - err_t),  err_t = 1{y_t outside}
               (Gibbs & Candès 2021) -> long-run miss rate is pushed back to alpha
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the k-th smallest score, k = ceil((n+1)(1-alpha)).

    Edge cases (needed for ACI, where alpha_t can leave (0,1)):
    - alpha <= 0  -> np.inf
    - alpha >= 1  -> 0.0
    - k > n       -> np.inf   (too few calibration points)

    Worked example: scores = [0.1, 0.2, ..., 0.9] (n=9), alpha=0.1
        k = ceil(10 * 0.9) = 9  -> return 0.9
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    # TODO (you): implement the finite-sample conformal quantile.
    # Hint: handle the alpha edge cases first, then
    #       k = math.ceil((n + 1) * (1 - alpha)); sort; remember Python is 0-indexed.
    raise NotImplementedError("TODO 1: conformal_quantile")


# ---------------------------------------------------------------------------
def split_conformal_intervals(y, yhat, alpha: float = 0.1, window: int = 60):
    """Online split conformal with a FIXED alpha.

    At each t >= window, calibrate on the previous `window` absolute residuals
    (only info available before t), then form yhat[t] ± q.

    Returns (lo, hi, miss) arrays of len(y); NaN before the first full window.
    miss[t] = 1.0 if y[t] is outside [lo[t], hi[t]], else 0.0.
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    T = y.size
    scores = np.abs(y - yhat)
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    miss = np.full(T, np.nan)
    for t in range(window, T):
        # TODO (you): compute q from scores[t-window:t], fill lo[t], hi[t], miss[t].
        # Hint: 3–4 lines. Do NOT use scores[t] itself — that would be look-ahead.
        raise NotImplementedError("TODO 2: split_conformal_intervals")
    return lo, hi, miss


# ---------------------------------------------------------------------------
def aci_intervals(y, yhat, alpha: float = 0.1, gamma: float = 0.02, window: int = 60):
    """Adaptive conformal inference.

    Same as split conformal, but use a working level alpha_t that is updated
    AFTER seeing y[t]:
        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

    Worked example (alpha=0.1, gamma=0.05, alpha_t=0.10):
        miss -> alpha_{t+1} = 0.10 + 0.05*(0.1 - 1) = 0.055  (wider next time)
        hit  -> alpha_{t+1} = 0.10 + 0.05*(0.1 - 0) = 0.105  (slightly narrower)

    Returns (lo, hi, miss, alpha_path) where alpha_path[t] is the alpha_t USED at t.
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
        # TODO (you): (a) q from scores[t-window:t] at level alpha_t,
        #             (b) fill lo/hi/miss as in TODO 2,
        #             (c) update alpha_t with the ACI rule.
        # Hint: the order matters — record, predict, observe, then update.
        raise NotImplementedError("TODO 3: aci_intervals")
    return lo, hi, miss, alpha_path


# ---------------------------------------------------------------------------
# Scaffolding (done for you): report coverage and width by regime
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
