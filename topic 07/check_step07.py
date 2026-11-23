"""Check your TODOs:  python check_step07.py            (tests tabular.py)
                     python check_step07.py solution   (tests the reference)"""
import sys

import numpy as np

if len(sys.argv) > 1 and sys.argv[1] == "solution":
    import solution_tabular as T
else:
    import tabular as T

rng = np.random.default_rng(0)
ok = 0

# TODO 1 -------------------------------------------------------------
t = np.arange(20)
m = T.context_mask(t, origin=10, h=6)
assert m.dtype == bool and m.sum() == 5 and m[4] and not m[5], "row s usable iff s + h <= origin"
ok += 1; print("✓ TODO 1 context_mask (no look-ahead)")

# TODO 2 -------------------------------------------------------------
X = rng.normal(size=(30, 3)); y = rng.normal(size=30)
tau2, s2 = 0.5, 0.8
K = s2 * np.eye(30) + tau2 * X @ X.T
sign, logdet = np.linalg.slogdet(K)
ref = -0.5 * y @ np.linalg.solve(K, y) - 0.5 * logdet - 15 * np.log(2 * np.pi)
assert np.isclose(T.log_marginal_likelihood(X, y, tau2, s2), ref), "log marginal likelihood mismatch"
ok += 1; print("✓ TODO 2 log_marginal_likelihood")

# TODO 3 -------------------------------------------------------------
# (a) the 1-feature worked example from today's brief
mean, var = T.bayes_predictive(np.array([[1.0], [2.0]]), np.array([1.0, 3.0]), np.array([[3.0]]), 1.0, 1.0)
assert np.isclose(mean[0], 3.5) and np.isclose(var[0], 2.5), f"worked example: got mean={mean}, var={var}"
# (b) matches sklearn Ridge for the mean (alpha = sigma2/tau2) and var > sigma2
from sklearn.linear_model import Ridge
Xq = rng.normal(size=(5, 3))
mean, var = T.bayes_predictive(X, y, Xq, tau2, s2)
r = Ridge(alpha=s2 / tau2, fit_intercept=False).fit(X, y)
assert np.allclose(mean, r.predict(Xq)), "posterior mean should equal ridge with alpha = sigma2/tau2"
assert np.all(var > s2), "predictive var = noise + parameter uncertainty > sigma2"
# (c) more data -> less parameter uncertainty
Xb = rng.normal(size=(3000, 3)); yb = Xb @ np.array([1., -1., .5]) + rng.normal(size=3000)
_, vbig = T.bayes_predictive(Xb, yb, Xq, tau2, 1.0)
assert np.all(vbig - 1.0 < 0.01), "with 3000 rows, var should shrink to ~sigma2"
ok += 1; print("✓ TODO 3 bayes_predictive (= ridge mean, + parameter uncertainty)")

# TODO 4 -------------------------------------------------------------
from sklearn.metrics import mean_pinball_loss
yy = rng.normal(size=200); qh = rng.normal(size=200)
for q in (0.1, 0.5, 0.9):
    assert np.isclose(T.pinball_loss(yy, qh, q), mean_pinball_loss(yy, qh, alpha=q)), f"pinball q={q}"
ok += 1; print("✓ TODO 4 pinball_loss")

print(f"\nAll {ok}/4 passed. Now run:  python run_step07.py")
