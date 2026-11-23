"""Step 05 — Does the embedding 'know' hawkish vs dovish? Probe + sanity baselines.

    python run_step05.py                      # uses probe.py (your version)
    python run_step05.py --impl solution_probe
    EMBEDDER=hash python run_step05.py        # force the offline embedder
"""
import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from embed import embed

ap = argparse.ArgumentParser()
ap.add_argument("--impl", default="probe")
ap.add_argument("--B", type=int, default=100, help="permutations")
args = ap.parse_args()
P = importlib.import_module(args.impl)

here = Path(__file__).parent
df = pd.read_csv(here / "sentences.csv")
X, emb_name = embed(df.text.tolist())
y = df.label.to_numpy()
print(f"{len(y)} sentences, embedder = {emb_name}, dim = {X.shape[1]}\n")

rows = []
acc_lr = P.kfold_accuracy(P.logistic_fp, X, y)
rows.append(("Logistic probe (all dims)", acc_lr))
rows.append(("Difference-of-means direction", P.kfold_accuracy(P.dom_fp, X, y)))
for k in (1, 8):
    m, s = P.random_direction_accuracy(X, y, k=k)
    rows.append((f"Probe on {k} RANDOM direction(s)  (sd {s:.2f})", m))
rng = np.random.default_rng(7)
shuf = np.mean([P.kfold_accuracy(P.logistic_fp, X, rng.permutation(y)) for _ in range(10)])
rows.append(("Control: shuffled labels", shuf))

print(f"{'5-fold CV accuracy':<48}acc")
for name, a in rows:
    print(f"{name:<48}{a:.2f}")
print(f"\nSelectivity = probe - shuffled control = {acc_lr - shuf:+.2f}")

print(f"\nPermutation test on the logistic probe (B={args.B}) ...")
cv_score = lambda X_, y_: P.kfold_accuracy(P.logistic_fp, X_, y_)
real, null, p = P.permutation_pvalue(cv_score, X, y, B=args.B, seed=0)
print(f"  real acc {real:.2f} | null mean {null.mean():.2f}, 95th pct {np.quantile(null, .95):.2f} | p = {p:.3f}")

# Read the direction: fit on all data, score every sentence along w
w, b = P.fit_logistic(X, y)
df["score"] = X @ w + b
df = df.sort_values("score")
print("\nMost DOVISH along the probe direction:")
for t in df.text.head(3):
    print("  -", t)
print("Most HAWKISH:")
for t in df.text.tail(3)[::-1]:
    print("  -", t)

# Validation angle: what does the probe get wrong out-of-fold?
probs = np.zeros(len(y))
fold = np.arange(len(y)) % 5
for f in range(5):
    tr, te = fold != f, fold == f
    wf, bf = P.fit_logistic(X[tr], y[tr])
    probs[te] = P.predict_proba(X[te], wf, bf)
miss = pd.read_csv(here / "sentences.csv").assign(p_hawk=probs)
miss = miss[(miss.p_hawk > 0.5) != (miss.label == 1)]
print(f"\nOut-of-fold mistakes ({len(miss)}):")
for _, r in miss.iterrows():
    print(f"  [{'H' if r.label else 'D'} | p_hawk={r.p_hawk:.2f}] {r.text}")
