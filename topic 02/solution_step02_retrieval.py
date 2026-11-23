"""ai-econ-lab · Step 02 — Hybrid retrieval over Bank of Canada rate announcements.
REFERENCE SOLUTION. Try step02_retrieval.py first!

The "R" in RAG: before an LLM can answer "when did the Bank end QT?", something
has to find the right paragraph. We build three retrievers and score them:

  1. BM25     (lexical: exact word overlap, great for numbers like "2.25%")
  2. Dense    (embeddings: meaning, great for paraphrases)
  3. Hybrid   (Reciprocal Rank Fusion of 1 + 2)

and evaluate with Recall@k and MRR on a small hand-labelled gold set.

Run (from the repo root):
    python -m lab.boc_text                                    # once: download + cache
    ollama pull embeddinggemma                                # once
    python step02/solution_step02_retrieval.py --embedder ollama
    python step02/solution_step02_retrieval.py --embedder hash   # offline fallback
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- scaffolding
_FRACTIONS = {"¼": ".25", "½": ".5", "¾": ".75"}
_TOKEN = re.compile(r"\d+(?:\.\d+)?%?|[a-z]+")
STOP = set("the a an of to in and on at for by is was be its it this that with as from has have did when which".split())


def tokenize(text: str) -> list[str]:
    """Lowercase, turn '2¼%' into '2.25%', keep numbers (with %) and words, drop stopwords.
    Why: BoC titles write 2¼% while the body writes 2.25% — lexical search needs one form."""
    for k, v in _FRACTIONS.items():
        text = text.replace(k, v)
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


# ---------------------------------------------------------------- TODO 1
def chunk_words(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    """Split text into chunks of `size` words; consecutive chunks share `overlap` words.
    step = size - overlap. The last chunk may be shorter. Never return an empty chunk."""
    assert 0 <= overlap < size
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


# ---------------------------------------------------------------- TODO 2
def bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]],
                k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Okapi BM25 score of the query against every document.

    score(q, d) = Σ_{t in q} IDF(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )
    IDF(t)      = ln( (N − n_t + 0.5) / (n_t + 0.5) + 1 )
    f(t,d) = count of t in d, |d| = len(d), avgdl = mean len, N = #docs, n_t = #docs containing t.
    """
    N = len(docs_tokens)
    lens = np.array([len(d) for d in docs_tokens], dtype=float)
    avgdl = lens.mean() if N else 0.0
    df = Counter(t for d in docs_tokens for t in set(d))
    tfs = [Counter(d) for d in docs_tokens]
    scores = np.zeros(N)
    for t in set(query_tokens):          # set(): repeated query words count once
        n_t = df.get(t, 0)
        if n_t == 0:
            continue
        idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1)
        for i, tf in enumerate(tfs):
            f = tf.get(t, 0)
            if f:
                scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * lens[i] / avgdl))
    return scores


# ---------------------------------------------------------------- TODO 3
def dense_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector (d,) and every row of doc_vecs (n, d).
    cos(q, x) = q·x / (‖q‖‖x‖). Don't assume inputs are already normalised."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    X = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
    return X @ q


# ---------------------------------------------------------------- TODO 4
def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).
    Each ranking is a list of item ids, best first; rank r starts at 1.
        RRF(d) = Σ_rankings 1 / (k + r(d))      (items missing from a ranking add 0)
    Return all item ids sorted by RRF score, highest first (ties: smaller id first)."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for r, d in enumerate(ranking, start=1):
            score[d] = score.get(d, 0.0) + 1.0 / (k + r)
    return sorted(score, key=lambda d: (-score[d], d))


# ---------------------------------------------------------------- scaffolding: embedders
class HashEmbedder:
    """Offline stand-in for a neural embedder: character 3-5-gram hashing.
    It captures spelling overlap, not meaning — useful for testing the pipeline."""
    name = "hash-char-ngrams"

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.v = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**14,
                                   alternate_sign=False, norm="l2")

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.v.transform(texts).toarray()


class OllamaEmbedder:
    """Local neural embeddings via Ollama's /api/embed (no API key)."""

    def __init__(self, model: str = "embeddinggemma", host: str = "http://localhost:11434"):
        self.model, self.host, self.name = model, host, f"ollama:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        import requests
        r = requests.post(f"{self.host}/api/embed", json={"model": self.model, "input": texts}, timeout=300)
        r.raise_for_status()
        return np.array(r.json()["embeddings"], dtype=float)


# ---------------------------------------------------------------- scaffolding: index + eval
class Index:
    def __init__(self, statements: list[dict], embedder, size=120, overlap=30):
        self.chunks, self.chunk_date = [], []
        for s in statements:
            for c in chunk_words(s["title"] + ". " + s["text"], size, overlap):
                self.chunks.append(c)
                self.chunk_date.append(s["date"])
        self.tokens = [tokenize(c) for c in self.chunks]
        self.embedder = embedder
        self.vecs = embedder.embed(self.chunks)

    def rank(self, query: str, method: str) -> list[int]:
        """Chunk ids, best first."""
        if method == "bm25":
            return list(np.argsort(-bm25_scores(tokenize(query), self.tokens), kind="stable"))
        if method == "dense":
            qv = self.embedder.embed([query])[0]
            return list(np.argsort(-dense_scores(qv, self.vecs), kind="stable"))
        if method == "hybrid":
            return rrf_fuse([self.rank(query, "bm25"), self.rank(query, "dense")])
        raise ValueError(method)

    def rank_docs(self, query: str, method: str) -> list[str]:
        """Collapse the chunk ranking to a document (date) ranking: first appearance wins."""
        seen, out = set(), []
        for i in self.rank(query, method):
            d = self.chunk_date[i]
            if d not in seen:
                seen.add(d)
                out.append(d)
        return out


def recall_at_k(ranked: list[str], relevant: str, k: int) -> float:
    return float(relevant in ranked[:k])


def reciprocal_rank(ranked: list[str], relevant: str) -> float:
    return 1.0 / (ranked.index(relevant) + 1) if relevant in ranked else 0.0


def evaluate(index: Index, gold: list[dict], methods=("bm25", "dense", "hybrid"), verbose=True):
    results = {}
    for m in methods:
        r1 = r3 = mrr = 0.0
        for g in gold:
            ranked = index.rank_docs(g["question"], m)
            r1 += recall_at_k(ranked, g["relevant_date"], 1)
            r3 += recall_at_k(ranked, g["relevant_date"], 3)
            mrr += reciprocal_rank(ranked, g["relevant_date"])
            if verbose and ranked[0] != g["relevant_date"]:
                print(f"  [{m:6s}] miss: {g['question'][:60]!r} -> top={ranked[0]}, want={g['relevant_date']}")
        n = len(gold)
        results[m] = {"recall@1": r1 / n, "recall@3": r3 / n, "MRR": mrr / n}
    return results


def main(argv=None):
    from lab.boc_text import load_statements
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["ollama", "hash"], default="ollama")
    ap.add_argument("--model", default="embeddinggemma")
    args = ap.parse_args(argv)

    statements = load_statements()
    with open(ROOT / "data" / "step02_gold.csv", newline="", encoding="utf-8") as f:
        gold = list(csv.DictReader(f))
    emb = OllamaEmbedder(args.model) if args.embedder == "ollama" else HashEmbedder()
    index = Index(statements, emb)
    print(f"{len(statements)} releases -> {len(index.chunks)} chunks; embedder = {emb.name}\n")
    res = evaluate(index, gold)
    print(f"\n{'method':8s} {'recall@1':>9s} {'recall@3':>9s} {'MRR':>6s}")
    for m, r in res.items():
        print(f"{m:8s} {r['recall@1']:9.2f} {r['recall@3']:9.2f} {r['MRR']:6.2f}")


if __name__ == "__main__":
    main()
