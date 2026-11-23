"""
Step 06 - GRPO (Group Relative Policy Optimization) by hand, in numpy.

The policy is tiny: for each "prompt" (an inflation state s) it holds
K logits theta[s, :] and picks a forecasting rule with pi = softmax(theta[s]).
The reward is VERIFIABLE: 1 if the chosen rule's 3-month-ahead CPI YoY
forecast lands within `tol` percentage points of what actually happened.

GRPO update for one prompt s (DeepSeekMath, Shao et al. 2024):
    sample G actions a_1..a_G ~ pi(.|s),  rewards r_i in {0,1}
    A_i   = (r_i - mean(r)) / (std(r) + eps)          # group-relative advantage
    g     = (1/G) * sum_i A_i * grad log pi(a_i|s)     # policy-gradient part
            - beta * grad KL(pi || pi_ref)             # stay near reference
    theta[s] += lr * g                                 # gradient ASCENT

Fill in the four `# TODO (you):` blocks. Run:  python check_step06.py
"""
import numpy as np


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def group_advantages(r, eps=1e-6):
    """r: array of G rewards (0/1). Return G advantages.
    If every reward in the group is the same (std == 0) return all zeros:
    the group carries no learning signal ("dead group")."""
    r = np.asarray(r, dtype=float)
    # TODO (you): standardise rewards WITHIN the group.
    # Hint: A = (r - r.mean()) / (r.std() + eps); use population std (ddof=0).
    #       If r.std() == 0, return np.zeros_like(r).
    raise NotImplementedError


def logprob_grad(pi, a):
    """Gradient of log pi[a] w.r.t. the logits theta, for pi = softmax(theta).
    Returns a length-K vector."""
    # TODO (you): d/dtheta_k log softmax(theta)[a] = 1{k == a} - pi[k]
    # Hint: start from np.eye(len(pi))[a] and subtract pi.
    raise NotImplementedError


def kl_grad(pi, ref):
    """Gradient w.r.t. theta of KL(pi || ref) = sum_k pi_k log(pi_k / ref_k),
    where pi = softmax(theta) and ref is fixed. Returns a length-K vector."""
    # TODO (you): grad = pi * (log pi - log ref - KL)
    # Hint: compute kl = np.sum(pi * (np.log(pi) - np.log(ref))) first.
    raise NotImplementedError


def grpo_step(theta_s, ref_s, rewards_of_action, G, beta, lr, rng, eps=1e-6):
    """One GRPO update for a single prompt.
    theta_s           : (K,) logits for this prompt (NOT modified in place)
    ref_s             : (K,) reference policy probabilities for this prompt
    rewards_of_action : (K,) 0/1 verifiable reward of each action for this prompt
    Returns (new_theta_s, dead) where dead=True if the group had zero variance."""
    pi = softmax(theta_s)
    a = rng.choice(len(pi), size=G, p=pi)          # sample a group of G "answers"
    r = rewards_of_action[a]                       # verify each one
    A = group_advantages(r, eps)
    dead = bool(np.all(A == 0))
    # TODO (you): build the GRPO gradient and take one ascent step.
    # Hint: g = mean over i of A[i] * logprob_grad(pi, a[i])  minus  beta * kl_grad(pi, ref_s)
    #       return theta_s + lr * g, dead
    raise NotImplementedError
