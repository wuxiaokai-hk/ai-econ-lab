"""
ai-econ-lab · Step 04 — attention as an analog forecaster for Canadian CPI inflation.

At each forecast origin T:
  query  q  = shape of the last L months of YoY inflation
  keys   K  = shapes of every past L-month window that is fully observed at T
  values V  = what inflation did h months after each past window
  forecast  = y[T] + attention(q, K, V)

Compares dense, top-k sparse, sliding-window and compressed(+top-k) attention vs a
naive (no-change) forecast, and prints which past episodes "now" attends to.

    python step04/run_step04.py                    # uses your attention.py
    python step04/run_step04.py --solution         # uses the reference solution
    python step04/run_step04.py --h 6 --tau 0.1    # other horizon / temperature
    python step04/run_step04.py --synthetic        # offline demo series
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--solution", action="store_true")
p.add_argument("--synthetic", action="store_true")
p.add_argument("--h", type=int, default=3, help="forecast horizon in months")
p.add_argument("--L", type=int, default=12, help="window length (months)")
p.add_argument("--tau", type=float, default=0.03, help="softmax temperature")
p.add_argument("--k", type=int, default=10, help="top-k keys kept")
p.add_argument("--block", type=int, default=4, help="compression block size")
p.add_argument("--window", type=int, default=60, help="sliding-window size")
p.add_argument("--start", default="2000-01", help="first forecast origin")
args = p.parse_args()

if args.solution:
    import solution_attention as A
else:
    import attention as A

SERIES = "V41690973"   # CPI, all-items, Canada (2002=100), Statistics Canada via BoC Valet
CACHE = Path(__file__).resolve().parents[1] / "data" / f"cpi_{SERIES}.csv"


def load_cpi_yoy():
    """Monthly CPI YoY inflation (%) from the Bank of Canada Valet API, cached to data/."""
    if CACHE.exists():
        s = pd.read_csv(CACHE, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import requests
        url = f"https://www.bankofcanada.ca/valet/observations/{SERIES}/json"
        obs = requests.get(url, timeout=30).json()["observations"]
        s = pd.Series({o["d"]: float(o[SERIES]["v"]) for o in obs if o.get(SERIES)},
                      name="cpi")
        s.index = pd.to_datetime(s.index)
        CACHE.parent.mkdir(exist_ok=True)
        s.to_csv(CACHE)
    return (100 * s.pct_change(12)).dropna().rename("yoy")


def synthetic_yoy(n=480, seed=1):
    """Offline stand-in: AR(1) inflation around 2% with a 2021-style surge and unwind."""
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = 2.0
    for t in range(1, n):
        y[t] = 2.0 + 0.95 * (y[t - 1] - 2.0) + rng.normal(0, 0.25)
    surge = np.r_[np.linspace(0, 5, 18), np.linspace(5, 0, 24)]
    y[n - 70:n - 70 + len(surge)] += surge
    idx = pd.date_range("1986-01-01", periods=n, freq="MS")
    return pd.Series(y, index=idx, name="yoy")


y_s = synthetic_yoy() if args.synthetic else load_cpi_yoy()
y = y_s.to_numpy()
dates = y_s.index
h, L, tau = args.h, args.L, args.tau

origins = [T for T in range(len(y)) if dates[T] >= pd.Timestamp(args.start) and T + h < len(y)]

methods = {
    "naive (no change)": lambda q, K, V: (0.0, None),
    "dense attention": lambda q, K, V: A.attention(q, K, V, tau),
    f"top-{args.k} sparse": lambda q, K, V: A.topk_attention(q, K, V, args.k, tau),
    f"sliding window {args.window}": lambda q, K, V: A.sliding_window_attention(q, K, V, args.window, tau),
    f"compressed x{args.block} + top-{max(1, args.k // args.block)}": lambda q, K, V: A.topk_attention(
        q, *A.compress_kv(K, V, args.block), max(1, args.k // args.block), tau),
}
keys_touched = {
    "naive (no change)": lambda n: 0,
    "dense attention": lambda n: n,
    f"top-{args.k} sparse": lambda n: min(args.k, n),
    f"sliding window {args.window}": lambda n: min(args.window, n),
    f"compressed x{args.block} + top-{max(1, args.k // args.block)}": lambda n: max(1, args.k // args.block),
}

rows = []
for T in origins:
    K, V, _ = A.build_memory(y, T, L, h)
    q = A.window_features(y, T, L)
    for name, f in methods.items():
        delta, _ = f(q, K, V)
        rows.append(dict(origin=dates[T], method=name, fc=y[T] + float(delta),
                         actual=y[T + h], keys=keys_touched[name](len(K))))
res = pd.DataFrame(rows)
res["abs_err"] = (res.fc - res.actual).abs()

surge = (res.origin >= "2021-01-01") & (res.origin <= "2023-12-01")
tab = res.groupby("method", sort=False).agg(MAE=("abs_err", "mean"), keys_per_query=("keys", "mean"))
tab["MAE 2021-23"] = res[surge].groupby("method", sort=False).abs_err.mean()
naive = tab.loc["naive (no change)", "MAE"]
tab["rel. MAE vs naive"] = tab.MAE / naive

pd.set_option("display.width", 120)
print(f"\nCanadian CPI YoY inflation, h={h} months, L={L}, tau={tau}, "
      f"{len(origins)} origins from {dates[origins[0]]:%Y-%m} to {dates[origins[-1]]:%Y-%m}"
      + ("  [SYNTHETIC DATA]" if args.synthetic else ""))
print(tab.round(3).to_string())

# --- Interpretability: which past episodes does the latest window resemble? -----------
T = len(y) - 1
K, V, ends = A.build_memory(y, T, L, h)
q = A.window_features(y, T, L)
_, w = A.attention(q, K, V, tau)
top = np.argsort(w)[::-1][:5]
print(f"\nLatest origin {dates[T]:%Y-%m} (inflation {y[T]:.2f}%): top-5 analog windows")
for i in top:
    s = ends[i]
    print(f"  window ending {dates[s]:%Y-%m}  weight {w[i]:.3f}   "
          f"then moved {V[i]:+.2f} pp over {h} months")
print(f"  -> attention forecast for {h} months ahead: {y[T] + w @ V:.2f}%"
      f"   (weights on top 5 = {w[top].sum():.0%})")
