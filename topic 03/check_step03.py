"""
Checks for Step 03. No internet needed.

    python step03/check_step03.py              # checks YOUR step03/conformal.py
    python step03/check_step03.py --solution   # checks the reference solution
"""
import importlib.util
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
target = "solution_step03_conformal.py" if "--solution" in sys.argv else "conformal.py"
spec = importlib.util.spec_from_file_location("m", HERE / target)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(f"Checking {target}\n")


def ok(msg):
    print(f"  PASS  {msg}")


# --- TODO 1 -----------------------------------------------------------------
s = np.arange(1, 10) / 10  # 0.1 .. 0.9, n = 9
assert math.isclose(m.conformal_quantile(s, 0.1), 0.9), "n=9, alpha=0.1 -> k=9 -> 0.9"
assert math.isclose(m.conformal_quantile(s, 0.5), 0.5), "n=9, alpha=0.5 -> k=ceil(5)=5 -> 0.5"
assert m.conformal_quantile(s, 0.05) == np.inf, "n=9, alpha=0.05 -> k=10 > n -> inf"
assert m.conformal_quantile(s, 0.0) == np.inf, "alpha<=0 -> inf"
assert m.conformal_quantile(s, -0.2) == np.inf, "alpha<0 -> inf"
assert m.conformal_quantile(s, 1.0) == 0.0, "alpha>=1 -> 0"
assert math.isclose(m.conformal_quantile(s[::-1], 0.1), 0.9), "must sort the scores"
ok("TODO 1 conformal_quantile: worked examples and edge cases")

# --- synthetic data: calm (sd 0.15) -> shock (sd 0.6) -> calm ----------------
rng = np.random.default_rng(0)
sd = np.r_[np.full(300, 0.15), np.full(48, 0.60), np.full(120, 0.15)]
y = 2.0 + np.cumsum(rng.normal(0, sd))   # random-walk "inflation"
yhat = np.r_[y[0], y[:-1]]              # naive forecast: last value
W = 60
shock = slice(300, 348)

# --- TODO 2 -----------------------------------------------------------------
lo, hi, miss = m.split_conformal_intervals(y, yhat, alpha=0.1, window=W)
assert np.all(np.isnan(miss[:W])), "no intervals before the first full window"
assert not np.any(np.isnan(miss[W:])), "intervals for every t >= window"
q_manual = m.conformal_quantile(np.abs(y - yhat)[W - W:W], 0.1)
assert math.isclose(hi[W] - yhat[W], q_manual), "at t=window, calibrate on scores[0:window] (no look-ahead)"
calm_cov = 1 - miss[W:300].mean()
shock_cov_split = 1 - miss[shock].mean()
assert 0.84 <= calm_cov <= 0.96, f"calm-period coverage should be ~90%, got {calm_cov:.3f}"
assert shock_cov_split < 0.80, f"a fixed-alpha interval should under-cover in the shock, got {shock_cov_split:.3f}"
ok(f"TODO 2 split conformal: calm coverage {calm_cov:.1%}, shock coverage {shock_cov_split:.1%} (breaks!)")

# --- TODO 3 -----------------------------------------------------------------
lo2, hi2, miss2, a_path = m.aci_intervals(y, yhat, alpha=0.1, gamma=0.05, window=W)
assert math.isclose(a_path[W], 0.1), "alpha_path[window] must be the starting alpha"
e = miss2[W]
assert math.isclose(a_path[W + 1], 0.1 + 0.05 * (0.1 - e)), "ACI update: alpha + gamma*(alpha - err)"
# exact replay of the update rule
for t in range(W, len(y) - 1):
    assert math.isclose(a_path[t + 1], a_path[t] + 0.05 * (0.1 - miss2[t]), abs_tol=1e-12), f"update rule broken at t={t}"
overall = 1 - np.nanmean(miss2)
shock_cov_aci = 1 - miss2[shock].mean()
T_eff = len(y) - W
bound = (max(0.1, 0.9) + 0.05) / (0.05 * T_eff)   # Gibbs & Candès Prop. 4.1
assert abs(np.nanmean(miss2) - 0.1) <= bound + 1e-12, "long-run miss rate must satisfy the ACI bound"
assert shock_cov_aci > shock_cov_split, "ACI should recover coverage faster during the shock"
ok(f"TODO 3 ACI: overall coverage {overall:.1%} (|miss-0.1| <= {bound:.3f}), shock coverage {shock_cov_aci:.1%}")

print("\nAll Step 03 checks passed. Now run: python step03/run_step03.py")
