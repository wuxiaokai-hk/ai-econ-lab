"""Offline checks for Step 01 (no internet, no model download, runs in seconds).

    python check_step01.py              # checks YOUR file: step01_chronos_cpi.py
    python check_step01.py --solution   # checks solution_step01.py
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd

def _hint(exc_type, exc, tb):
    import traceback

    traceback.print_exception(exc_type, exc, tb)
    if exc_type in (TypeError, NotImplementedError) or "ellipsis" in str(exc):
        print("\n→ Looks like a `# TODO (you):` block is still empty (`...` or NotImplementedError).")


sys.excepthook = _hint
mod_name = "solution_step01" if "--solution" in sys.argv else "step01_chronos_cpi"
S = importlib.import_module(mod_name)
passed = 0


def ok(msg: str) -> None:
    global passed
    passed += 1
    print(f"  ✓ {msg}")


print(f"Checking {mod_name}.py")

# 1) seasonal naive ------------------------------------------------------------
hist = np.arange(1, 25, dtype=float)                      # 1..24, m=12 -> last season = 13..24
fc = S.seasonal_naive(hist, 12)["median"]
assert np.allclose(fc, np.arange(13, 25)), f"seasonal_naive h=12 gave {fc}"
fc = S.seasonal_naive(hist, 15)["median"]
assert np.allclose(fc, list(range(13, 25)) + [13, 14, 15]), f"h>m should wrap around, got {fc}"
fc = S.seasonal_naive(np.array([1, 2, 3, 4, 5.0]), 4, m=2)["median"]
assert np.allclose(fc, [4, 5, 4, 5]), f"m=2 case gave {fc}"
ok("seasonal_naive repeats the last season (also for h > m)")

# 2) MASE ----------------------------------------------------------------------
# training series with seasonal diffs all = 2 -> scale = 2
y_train = np.concatenate([np.zeros(12), np.full(12, 2.0)])
v = S.mase(np.array([1.0, 1.0]), np.array([0.0, 3.0]), y_train, m=12)  # MAE = (1+2)/2 = 1.5
assert np.isclose(v, 0.75), f"MASE should be 1.5/2 = 0.75, got {v}"
v = S.mase(np.array([5.0]), np.array([5.0]), y_train, m=12)
assert np.isclose(v, 0.0), "perfect forecast should give MASE 0"
ok("mase = MAE / in-sample mean |y_t - y_{t-m}|")

# 3) coverage ------------------------------------------------------------------
c = S.interval_coverage(np.array([1, 2, 3, 10.0]), np.zeros(4), np.full(4, 3.0))
assert np.isclose(c, 0.75), f"3 of 4 inside -> 0.75, got {c}"
c = S.interval_coverage(np.array([0.0, 3.0]), np.zeros(2), np.full(2, 3.0))
assert np.isclose(c, 1.0), "endpoints count as inside"
ok("interval_coverage counts actuals inside [lo, hi] (inclusive)")

# 4) backtest: no leakage, correct alignment ------------------------------------
idx = pd.date_range("2000-01-01", periods=60, freq="MS")
y = pd.Series(np.arange(60, dtype=float) + 100, index=idx)
seen = []


def spy(history: np.ndarray, h: int) -> dict:
    seen.append(history.copy())
    return {"median": np.repeat(history[-1], h), "lo": None, "hi": None}


def oracle(history: np.ndarray, h: int) -> dict:
    # knows the truth for a linear series: next values are last+1..last+h
    med = history[-1] + np.arange(1, h + 1)
    return {"median": med, "lo": med - 0.5, "hi": med + 0.5}


origins = [idx[24], idx[36], idx[55]]  # last one has only 4 future points -> must be skipped
res = S.rolling_origin_backtest(y, {"spy": spy, "oracle": oracle}, origins, h=6)
assert set(res.columns) >= {"model", "origin", "mase", "coverage"}, f"columns: {list(res.columns)}"
assert len(res) == 4, f"expected 2 usable origins x 2 models = 4 rows, got {len(res)}"
assert seen[0][-1] == y[idx[24]] and len(seen[0]) == 25, "history must END AT the origin (inclusive)"
assert all(len(s) in (25, 37) for s in seen), "history leaked future data or was cut short"
o = res[res.model == "oracle"]
assert np.allclose(o["mase"], 0) and np.allclose(o["coverage"], 1), (
    "oracle should score MASE 0 & coverage 1 -> test window is misaligned (off by one?)"
)
sp = res[res.model == "spy"]
# naive on a +1/month line: errors 1..6, MAE 3.5, scale = 12 -> 0.2917
assert np.allclose(sp["mase"], 3.5 / 12), f"naive MASE should be 0.2917, got {sp['mase'].tolist()}"
assert sp["coverage"].isna().all(), "models without lo/hi should get coverage NaN"
ok("rolling_origin_backtest: no look-ahead, test = next h months, short tails skipped")

print(f"\nAll {passed}/4 checks passed 🎉  Now run:  python {mod_name}.py")
