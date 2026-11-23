"""
ai-econ-lab · Step 04 — REFERENCE SOLUTION
Attention by hand: dense, top-k sparse (DeepSeek-style), and compressed attention.

Try attention.py first. Check yours with:  python step04/check_step04.py
Check this file with:                     python step04/check_step04.py --solution
"""
import numpy as np


# ---------------------------------------------------------------------------
# TODO 1 — numerically stable softmax with an optional mask
# ---------------------------------------------------------------------------
def softmax(scores, mask=None):
    """scores: (n,) array. mask: (n,) bool, True = allowed to attend.
    Returns weights (n,) that are >= 0, sum to 1, and are exactly 0 where mask is False."""
    s = np.asarray(scores, dtype=float)
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    m = np.max(s)                      # subtract the max: exp never overflows
    e = np.exp(s - m)                  # exp(-inf) = 0 -> masked keys get weight 0
    return e / e.sum()


# ---------------------------------------------------------------------------
# TODO 2 — scaled dot-product attention for ONE query
# ---------------------------------------------------------------------------
def attention(q, K, V, tau=1.0, mask=None):
    """q: (d,), K: (n, d), V: (n,) or (n, m).
    weights = softmax( K q / (tau * sqrt(d)) ),  output = weights @ V
    Returns (output, weights)."""
    d = q.shape[-1]
    scores = K @ q / (tau * np.sqrt(d))
    w = softmax(scores, mask)
    return w @ V, w


# ---------------------------------------------------------------------------
# TODO 3 — top-k sparse attention (the idea behind DeepSeek's lightning indexer)
# ---------------------------------------------------------------------------
def topk_attention(q, K, V, k, tau=1.0):
    """Score every key, keep only the k highest, softmax over those k.
    Returns (output, weights) with weights == 0 outside the top k."""
    d = q.shape[-1]
    scores = K @ q / (tau * np.sqrt(d))
    k = min(k, len(scores))
    top = np.argpartition(scores, -k)[-k:]        # indices of the k largest scores
    mask = np.zeros(len(scores), dtype=bool)
    mask[top] = True
    w = softmax(scores, mask)
    return w @ V, w


# ---------------------------------------------------------------------------
# TODO 4 — compress the KV sequence by mean-pooling blocks (CSA / HCA idea)
# ---------------------------------------------------------------------------
def compress_kv(K, V, block):
    """Mean-pool consecutive blocks of `block` keys/values.
    If n is not divisible by block, DROP THE OLDEST n % block rows (keep the most recent).
    Returns (Kc, Vc) with n // block rows."""
    n = K.shape[0]
    nb = n // block
    start = n - nb * block
    Kc = K[start:].reshape(nb, block, *K.shape[1:]).mean(axis=1)
    Vc = V[start:].reshape(nb, block, *V.shape[1:]).mean(axis=1)
    return Kc, Vc


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
