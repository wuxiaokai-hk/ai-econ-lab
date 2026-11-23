"""Reference solution for step06/grpo.py - try the TODOs first!"""
import numpy as np


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def group_advantages(r, eps=1e-6):
    r = np.asarray(r, dtype=float)
    s = r.std()
    if s == 0:
        return np.zeros_like(r)
    return (r - r.mean()) / (s + eps)


def logprob_grad(pi, a):
    return np.eye(len(pi))[a] - pi


def kl_grad(pi, ref):
    kl = np.sum(pi * (np.log(pi) - np.log(ref)))
    return pi * (np.log(pi) - np.log(ref) - kl)


def grpo_step(theta_s, ref_s, rewards_of_action, G, beta, lr, rng, eps=1e-6):
    pi = softmax(theta_s)
    a = rng.choice(len(pi), size=G, p=pi)
    r = rewards_of_action[a]
    A = group_advantages(r, eps)
    dead = bool(np.all(A == 0))
    pg = np.mean([A[i] * logprob_grad(pi, a[i]) for i in range(G)], axis=0)
    g = pg - beta * kl_grad(pi, ref_s)
    return theta_s + lr * g, dead
