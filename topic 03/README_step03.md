## Step 03 — Conformal prediction: coverage you can monitor (split CP vs ACI)

**Question.** Step 01 showed Chronos-2's q10–q90 band can miss its 80% target when the regime changes. Can we wrap *any* forecaster (naive, Chronos, an LLM) so that its intervals hit a target coverage, and recover quickly after a shock?

**Method.**
- Score: `s_t = |y_t − ŷ_t|` (absolute forecast error).
- Split conformal: `q = k-th smallest of the last n scores`, `k = ⌈(n+1)(1−α)⌉`; interval `ŷ_t ± q`. Valid if errors are exchangeable, which fails under regime change.
- Adaptive conformal inference (Gibbs & Candès, NeurIPS 2021): `α_{t+1} = α_t + γ(α − err_t)`. A miss widens the next interval and a hit narrows it a little. The guarantee holds for **any** data path: `|(1/T)Σ err_t − α| ≤ (max(α₁,1−α₁)+γ)/(γT)`.
- Data: BoC Valet V41690973 → 12-month CPI inflation; naive h=1 forecast; 60-month rolling calibration; 90% target.

**Run.**
```bash
python step03/check_step03.py      # unit checks for your 3 TODOs
python step03/run_step03.py        # real CPI; writes data/step03_intervals.csv/.png
```

**What it shows.** Fixed-α conformal is close to 90% in calm years and collapses in the 2021–23 surge. ACI's working α_t drops after misses, so the intervals widen and long-run coverage goes back to ~90%. Sometimes α_t < 0, which gives an *infinite* interval. That is the method saying, honestly, "I can't promise coverage right now", and it is a natural monitoring flag.

**What I learned.**
- Coverage is an observable, testable monitoring metric, unlike "the model looks fine".
- Conformal is model-agnostic: the same wrapper works on Chronos, GBDT or an LLM forecaster.
- The exchangeability assumption is exactly what macro regime shifts break. ACI trades a per-period guarantee for a long-run one that holds without assumptions.
- Link to OSFI E-23 (effective May 1, 2027), Principle 3.6: monitoring should define "thresholds for breaches" and handle "the elevated potential for model drift" in AI/ML models. A rolling coverage rate and α_t are such thresholds.

**Try next.** Replace the naive forecast with Step 01's Chronos-2 median (or use its q10/q90 with the CQR score `max(q_lo − y, y − q_hi)`), and move to h=12. With h=12 you only learn `err_t` 12 months later, so update α with a 12-month lag and see how much slower the recovery is.
