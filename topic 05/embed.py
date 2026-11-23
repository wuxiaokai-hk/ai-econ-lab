"""Step 05 scaffolding: turn sentences into vectors (no TODOs here).

Default: Ollama `embeddinggemma` (same model as Step 02), cached to .npy.
Fallback: a hashing bag-of-words embedder (numpy only), so everything runs
offline. Run with EMBEDDER=hash to force the fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/embed")
OLLAMA_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma")
CACHE_DIR = Path(__file__).parent / "cache"


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9%]+", text.lower())
    # unigrams + bigrams so "no longer" / "too early" carry some signal
    return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]


def hash_embed(texts: list[str], dim: int = 256) -> np.ndarray:
    """Feature-hashing BoW: each token adds +-1 to one of `dim` buckets."""
    X = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in _tokens(t):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            X[i, h % dim] += 1.0 if (h >> 64) % 2 else -1.0
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1, norms)


def ollama_embed(texts: list[str]) -> np.ndarray:
    body = json.dumps({"model": OLLAMA_MODEL, "input": texts}).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return np.asarray(json.load(r)["embeddings"], dtype=float)


def embed(texts: list[str]) -> tuple[np.ndarray, str]:
    """Return (X, name_of_embedder). Uses a cache keyed on the texts."""
    mode = os.environ.get("EMBEDDER", "auto")
    if mode != "hash":
        key = hashlib.md5(("\n".join(texts) + OLLAMA_MODEL).encode()).hexdigest()[:12]
        path = CACHE_DIR / f"{OLLAMA_MODEL}_{key}.npy"
        if path.exists():
            return np.load(path), OLLAMA_MODEL
        try:
            X = ollama_embed(texts)
            CACHE_DIR.mkdir(exist_ok=True)
            np.save(path, X)
            return X, OLLAMA_MODEL
        except Exception as e:  # Ollama not running -> fall back
            if mode == "ollama":
                raise
            print(f"[embed] Ollama unavailable ({e.__class__.__name__}); using hashing BoW")
    return hash_embed(texts), "hash-bow-256"
