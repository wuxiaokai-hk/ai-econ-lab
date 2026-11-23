"""
ai-econ-lab · Step 04 — STARTER (you fill in the TODOs)
Attention by hand: dense, top-k sparse (DeepSeek-style), and compressed attention,
used as an "analog forecaster" for Canadian CPI inflation.

  1. Fill the 4 TODO blocks below (numpy only).
  2. python step04/check_step04.py            -> all checks should print PASS
  3. python step04/run_step04.py              -> backtest on Bank of Canada CPI

Stuck? Peek at step04/solution_attention.py.
"""
import numpy as np


# ---------------------------------------------------------------------------
# TODO 1 — numerically stable softmax with an optional mask
# ---------------------------------------------------------------------------
def softmax(scores, mask=None):
    """scores: (n,) array. mask: (n,) bool, True = allowed to attend.
    Returns weights (n,) that are >= 0, sum to 1, and are exactly 0 where mask is False."""
    s = np.asarray(scores, dtype=float)
    # TODO (you):
    #   a) where mask is False, set the score to -np.inf      (hint: np.where)
    #   b) subtract the max score before exp, so exp(1000) can't overflow
    #   c) exp, then divide by the sum
    # Why -inf? exp(-inf) = 0, so masked keys get exactly zero weight.
    raise NotImplementedError("TODO 1: softmax")


# ---------------------------------------------------------------------------
# TODO 2 — scaled dot-product attention for ONE query
# ---------------------------------------------------------------------------
def attention(q, K, V, tau=1.0, mask=None):
    """q: (d,), K: (n, d), V: (n,) or (n, m).
    weights = softmax( K q / (tau * sqrt(d)) ),  output = weights @ V
    Returns (output, weights)."""
    d = q.shape[-1]
    # TODO (you): 3 lines.
    #   scores = ...            (n,) one similarity per key; divide by tau*sqrt(d)
    #   w = softmax(scores, mask)
    #   return w @ V, w
    # Economist's view: this is a Nadaraya–Watson kernel regression with kernel
    # exp(<q, k_i> / tau·sqrt(d)). tau -> 0: nearest neighbour. tau -> inf: plain mean.
    raise NotImplementedError("TODO 2: attention")


# ---------------------------------------------------------------------------
# TODO 3 — top-k sparse attention (the idea behind DeepSeek's lightning indexer)
# ---------------------------------------------------------------------------
def topk_attention(q, K, V, k, tau=1.0):
    """Score every key, keep only the k highest, softmax over those k.
    Returns (output, weights) with weights == 0 outside the top k."""
    d = q.shape[-1]
    scores = K @ q / (tau * np.sqrt(d))
    k = min(k, len(scores))
    # TODO (you):
    #   top  = indices of the k largest scores   (hint: np.argpartition(scores, -k)[-k:])
    #   mask = boolean array, True only at `top`
    #   reuse softmax(scores, mask) and return (w @ V, w)
    # In DeepSeek-V3.2/V4 a cheap "indexer" does this scoring, so the expensive
    # attention only touches k keys instead of all n (1M tokens -> a few thousand).
    raise NotImplementedError("TODO 3: topk_attention")


# ---------------------------------------------------------------------------
# TODO 4 — compress the KV sequence by mean-pooling blocks (CSA / HCA idea)
# ---------------------------------------------------------------------------
def compress_kv(K, V, block):
    """Mean-pool consecutive blocks of `block` keys/values.
    If n is not divisible by block, DROP THE OLDEST n % block rows (keep the most recent).
    Returns (Kc, Vc) with n // block rows."""
    n = K.shape[0]
    nb = n // block
    # TODO (you):
    #   start = n - nb * block                      (rows to drop at the front)
    #   Kc = K[start:].reshape(nb, block, *K.shape[1:]).mean(axis=1)
    #   Vc = same for V
    # DeepSeek-V4 uses learned, gated pooling (x4 in CSA, x128 in HCA); a plain mean
    # is the simplest version of the same idea: fewer, coarser memory slots.
    raise NotImplementedError("TODO 4: compress_kv")


# ---------------------------------------------------------------------------
# Scaffolding (given): turn a time series into queries, keys and values
# ---------------------------------------------------------------------------
def window_features(y, end, L):
    """Shape of the last L observations ending at index `end` (inclusive):
    re-based to the last value and L2-normalised (like QK-norm), so similarity = pattern, not level."""
    w = y[end - L + 1:end + 1] - y[end]
    nrm = np.linalg.norm(w)
    return w / nrm if nrm > 0 else w


def build_memory(y, T, L, h):
    """Keys/values that are fully OBSERVED at forecast origin T (the causal mask).
    Key for window ending at s : window_features(y, s, L)
    Value                     : y[s+h] - y[s]   (what happened next)
    Only s with s + h <= T are allowed -> no peeking at the future."""
    ends = np.arange(L - 1, T - h + 1)
    K = np.stack([window_features(y, s, L) for s in ends])
    V = np.array([y[s + h] - y[s] for s in ends])
    return K, V, ends


def sliding_window_attention(q, K, V, window, tau=1.0):
    """Given: attend only to the most recent `window` keys (the local branch in CSA)."""
    return attention(q, K[-window:], V[-window:], tau)
