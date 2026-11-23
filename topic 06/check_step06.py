"""Checks for step06. Run:  python check_step06.py   (or: python check_step06.py solution)"""
import sys, importlib
import numpy as np

mod = importlib.import_module("solution_grpo" if len(sys.argv) > 1 else "grpo")
softmax = mod.softmax

# 1) group advantages: 2 of 8 correct -> A+ = sqrt(3), A- = -1/sqrt(3)
A = mod.group_advantages(np.array([1, 1, 0, 0, 0, 0, 0, 0]))
assert np.allclose(A[:2], np.sqrt(3), atol=1e-4), A
assert np.allclose(A[2:], -1 / np.sqrt(3), atol=1e-4), A
assert np.allclose(mod.group_advantages(np.ones(8)), 0), "all-correct group must give zeros"
assert np.allclose(mod.group_advantages(np.zeros(8)), 0), "all-wrong group must give zeros"
print("ok 1  group_advantages  (2/8 correct -> +1.732 / -0.577; dead groups -> 0)")

# 2) logprob_grad vs finite differences
rng = np.random.default_rng(0)
th = rng.normal(size=5)
for a in range(5):
    num = np.array([(np.log(softmax(th + h)[a]) - np.log(softmax(th - h)[a])) / 2e-6
                    for h in np.eye(5) * 1e-6])
    assert np.allclose(mod.logprob_grad(softmax(th), a), num, atol=1e-5), a
print("ok 2  logprob_grad matches finite differences")

# 3) kl_grad vs finite differences
ref = softmax(rng.normal(size=5))
kl = lambda t: np.sum(softmax(t) * (np.log(softmax(t)) - np.log(ref)))
num = np.array([(kl(th + h) - kl(th - h)) / 2e-6 for h in np.eye(5) * 1e-6])
assert np.allclose(mod.kl_grad(softmax(th), ref), num, atol=1e-5)
assert np.allclose(mod.kl_grad(ref, ref), 0, atol=1e-12), "KL grad at pi=ref must be 0"
print("ok 3  kl_grad matches finite differences (and is 0 at pi = ref)")

# 4) success amplification on a toy prompt: 1 good action out of 5, p0 = 0.2
r_act = np.array([0, 0, 1, 0, 0])
ref_s = np.full(5, 0.2)
theta = np.zeros(5)
rng = np.random.default_rng(1)
for _ in range(300):
    theta, _ = mod.grpo_step(theta, ref_s, r_act, G=8, beta=0.02, lr=0.5, rng=rng)
p = softmax(theta)[2]
assert p > 0.8, f"success prob should be amplified well above 0.2, got {p:.3f}"
# stronger KL pull -> smaller amplification (fixed point depends on beta)
theta_b = np.zeros(5); rng = np.random.default_rng(1)
for _ in range(300):
    theta_b, _ = mod.grpo_step(theta_b, ref_s, r_act, G=8, beta=1.0, lr=0.5, rng=rng)
pb = softmax(theta_b)[2]
assert 0.2 < pb < p, f"beta=1 should amplify less: {pb:.3f} vs {p:.3f}"
print(f"ok 4  GRPO amplifies success: p0=0.20 -> {p:.3f} (beta=0.02), {pb:.3f} (beta=1.0)")
print("\nAll step06 checks passed.")
