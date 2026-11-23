"""Check Step 05.  python check_step05.py            (tests probe.py)
                   python check_step05.py solution_probe   (tests the reference)"""
import importlib
import sys

import numpy as np

mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "probe")
rng = np.random.default_rng(42)

# --- TODO 1: logistic probe matches the closed-form optimum (via sklearn if present)
n, d = 200, 5
X = rng.standard_normal((n, d))
w_true = np.array([2.0, -1.0, 0.0, 0.5, 0.0])
y = (rng.random(n) < 1 / (1 + np.exp(-(X @ w_true + 0.3)))).astype(int)
lam = 0.05
w, b = mod.fit_logistic(X, y, lam=lam, lr=0.5, n_iter=5000)
acc = np.mean((mod.predict_proba(X, w, b) > 0.5) == y)
assert acc > 0.75, f"TODO 1: in-sample accuracy {acc:.2f} too low"
assert w[0] > 0 and w[1] < 0, "TODO 1: signs of w look wrong (check p - y vs y - p)"
try:
    from sklearn.linear_model import LogisticRegression
    sk = LogisticRegression(C=1 / (lam * n), tol=1e-10, max_iter=10000).fit(X, y)
    assert np.allclose(w, sk.coef_[0], atol=1e-3), f"TODO 1: w={w.round(3)} vs sklearn {sk.coef_[0].round(3)}"
    assert abs(b - sk.intercept_[0]) < 1e-3, "TODO 1: intercept differs from sklearn"
    print("TODO 1 ok  (matches sklearn: mean log-loss + lam/2||w||^2  <=>  C = 1/(lam*n))")
except ImportError:
    print("TODO 1 ok  (sklearn not installed, skipped exact comparison)")

# --- TODO 2: difference of means
X2 = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
y2 = np.array([1, 1, 0, 0])
dvec, t = mod.diff_of_means(X2, y2)
assert np.allclose(dvec, [2.0, -2.0]), f"TODO 2: d={dvec}, expected [2,-2]"
assert abs(t - 0.0) < 1e-12, f"TODO 2: threshold {t}, expected 0"
X3 = X2 + 5.0  # shift: d unchanged, threshold must move
_, t3 = mod.diff_of_means(X3, y2)
assert abs(t3 - 0.0) < 1e-12 and np.allclose(X3 @ dvec > t3, y2 == 1), "TODO 2: threshold/shift"
print("TODO 2 ok")

# --- TODO 3: permutation p-value
score = lambda X_, y_: float(np.mean(y_[: len(y_) // 2]))  # share of 1s in first half
yy = np.array([1] * 10 + [0] * 10)
real, null, p = mod.permutation_pvalue(score, None, yy, B=99, seed=1)
assert real == 1.0 and len(null) == 99, "TODO 3: real score / null length"
assert abs(p - (1 + np.sum(null >= real)) / 100) < 1e-12, "TODO 3: p formula"
assert p < 0.05, f"TODO 3: perfectly separated labels should give small p, got {p}"
noise = lambda X_, y_: 0.5
_, _, p2 = mod.permutation_pvalue(noise, None, yy, B=49)
assert p2 == 1.0, f"TODO 3: ties must count (null >= real) -> p=1, got {p2}"
print("TODO 3 ok")
print("All Step 05 checks passed. Now run: python run_step05.py")
