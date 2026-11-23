# Step 05 — Does the model "know" hawkish vs dovish? Linear probes and sanity baselines

**AI development:** interpretability is shifting from "we found a feature" to "does it beat a random baseline?"
Korznikov et al. (*Sanity Checks for Sparse Autoencoders*, arXiv 2602.14111, v2 Nov 2026) find that SAEs with
**random, frozen** directions score almost the same as fully trained SAEs (sparse probing 0.69 vs 0.72), and on synthetic data
trained SAEs recover only 9% of the true features while explaining 71% of the variance. Kantamneni, Engels et al. (arXiv 2502.16681) find SAE probes
do not reliably beat plain baselines. The lesson for validators: **a claim about a model needs a control.**

## What this step does
80 short central-bank-style sentences (`sentences.csv`, 40 hawkish / 40 dovish, written by hand for this exercise and
not taken from real statements) are embedded with Ollama `embeddinggemma` (same as Step 02). We then ask whether a *linear* direction in
embedding space separates the two classes, and whether that beats the controls:

| Method | What it tests |
|---|---|
| Logistic probe, p = σ(w·x + b) | is stance linearly readable from the embedding? |
| Difference of means, d = μ₁ − μ₀ | the no-training baseline (LDA with identity covariance) |
| Probe on 1 or 8 **random** directions | is the stance direction special, or would any direction do? |
| Shuffled labels | chance level for *this* probe on *this* data |
| Permutation p = (1 + #{null ≥ real}) / (1 + B) | is the accuracy above what chance would give? |

Selectivity = probe accuracy − shuffled-label accuracy (Hewitt & Liang's "control task" idea).

## Run
```bash
cd step05
ollama pull embeddinggemma          # if not already pulled in Step 02
python check_step05.py              # fill the 3 TODOs in probe.py until this passes
python run_step05.py                # ~1 min; embeddings cached in step05/cache/
EMBEDDER=hash python run_step05.py  # same pipeline, bag-of-words hashing "embedding"
```

## Reference result (offline hashing embedder, tested)
With the hashing bag-of-words embedder (it has no semantic knowledge), the probe is at chance:
CV accuracy 0.53, shuffled control 0.50, permutation p = 0.36. Few words repeat across
the 80 sentences, so word counts cannot carry over to new sentences. **That is the control
representation.** Compare it with your `embeddinggemma` run: the difference between the two
shows what the language model adds. _(Fill in: probe acc __, random-8 __, p = __.)_

## What I learned
- A probe is just logistic regression on hidden states. The gradient X'(p − y)/n has the same form as OLS's X'e/n.
- The difference-of-means direction is a strong baseline with no training. If it matches the probe, the "concept" is simply a mean shift.
- Accuracy means little until it is compared with shuffled labels and random directions. The same rule applies to SAE features, to attention-based "explanations", and to any interpretability claim made to a validator.
- The out-of-fold mistakes (negations like "no longer a concern", conditional forward guidance) show where the representation is weak. That is useful content for a validation note.

## Try next
Probe **real** Bank of Canada sentences from the Step 02 corpus: label 30 of them, then fit the probe on 2022 sentences and
test it on 2025 ones (a tightening year versus an easing year). Does the direction hold up across the policy regime, or did it learn "2022 vocabulary"?
