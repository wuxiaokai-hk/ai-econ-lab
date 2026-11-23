"""Checks for Step 02. Run from the repo root:

    python step02/check_step02.py                              # checks your step02_retrieval.py
    python step02/check_step02.py solution_step02_retrieval    # checks the reference solution

No network or Ollama needed: uses tiny hand-computed cases and a synthetic toy corpus.
"""
import importlib
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
mod_name = sys.argv[1] if len(sys.argv) > 1 else "step02_retrieval"
m = importlib.import_module(mod_name)
print(f"checking {mod_name}.py\n")


def ok(name):
    print(f"  PASS  {name}")


# ---- TODO 1: chunking
words = [f"w{i}" for i in range(10)]
ch = m.chunk_words(" ".join(words), size=4, overlap=1)
assert ch == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"], f"chunk_words gave {ch}"
assert m.chunk_words("a b c d", size=4, overlap=1) == ["a b c d"], "a text of exactly `size` words is ONE chunk"
assert m.chunk_words("", 4, 1) == [], "empty text -> []"
assert m.chunk_words("a b c d e", size=4, overlap=0) == ["a b c d", "e"], "last chunk may be shorter"
ok("TODO 1 chunk_words")

# ---- TODO 2: BM25 (hand-computed; k1=1.5, b=0.75)
docs = [["rate", "cut"], ["rate", "hold", "hold"], ["inflation"]]
s = m.bm25_scores(["cut"], docs)
# N=3, n_t=1: IDF = ln((3-1+.5)/(1+.5)+1) = ln(2.6667) = 0.98083; doc0 |d|=avgdl -> tf part = 1
assert np.allclose(s, [math.log(2.5 / 1.5 + 1), 0, 0], atol=1e-6), f"bm25 'cut' gave {s}"
s = m.bm25_scores(["rate"], docs)
# IDF = ln(1.5/2.5+1) = 0.47000; doc1 is longer (3 > avgdl 2) so its tf part is 2.5/3.0625 = 0.8163
idf = math.log(1.5 / 2.5 + 1)
assert np.allclose(s, [idf * 1.0, idf * 2.5 / 3.0625, 0], atol=1e-6), f"bm25 'rate' gave {s} (length normalisation?)"
assert np.allclose(m.bm25_scores(["gdp"], docs), 0), "unseen term scores 0"
ok("TODO 2 bm25_scores")

# ---- TODO 3: cosine
sc = m.dense_scores(np.array([1.0, 0.0]), np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]))
assert np.allclose(sc, [1.0, 0.0, 1 / math.sqrt(2)], atol=1e-6), f"dense_scores gave {sc} (normalise both sides)"
ok("TODO 3 dense_scores")

# ---- TODO 4: RRF  (the worked example from today's brief)
assert m.rrf_fuse([[0, 1, 2], [1, 2, 0]]) == [1, 0, 2], "RRF: doc 1 (ranks 2 & 1) beats doc 0 (ranks 1 & 3)"
assert m.rrf_fuse([[0, 1], [2]]) == [0, 2, 1], "RRF: missing items add 0; ties -> smaller id first"
assert m.rrf_fuse([[5, 7]], k=0) == [5, 7]
ok("TODO 4 rrf_fuse")

# ---- end-to-end on a SYNTHETIC toy corpus (not real BoC text)
try:
    emb = m.HashEmbedder()
except ImportError:
    print("  SKIP  end-to-end (pip install scikit-learn)")
else:
    toy = [
        {"date": "A", "title": "Toy bank lowers policy rate to 2¼%", "text": "The toy bank reduced its target rate to 2.25%. Growth is weak."},
        {"date": "B", "title": "Toy bank holds policy rate at 2¼%", "text": "The toy bank held its target rate at 2.25%. Inflation is near target."},
        {"date": "C", "title": "Toy bank ends balance sheet runoff", "text": "The toy bank will stop quantitative tightening and restart asset purchases."},
    ]
    idx = m.Index(toy, emb, size=30, overlap=5)
    assert idx.rank_docs("When did the bank end quantitative tightening?", "bm25")[0] == "C"
    assert idx.rank_docs("Which time did the bank holds its rate at 2.25%?", "hybrid")[0] == "B"
    res = m.evaluate(idx, [{"question": "stop quantitative tightening", "relevant_date": "C"}], verbose=False)
    assert res["hybrid"]["recall@1"] == 1.0
    ok("end-to-end toy corpus")

print("\nAll checks passed. Now run it on the real releases:")
print("  python step02/step02_retrieval.py --embedder ollama")
