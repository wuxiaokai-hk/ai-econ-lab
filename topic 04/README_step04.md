## Step 04 — Attention by hand: dense vs sparse vs compressed (DeepSeek-V4 idea) on CPI

**What it shows.** The core operation inside every transformer (and inside Chronos-2 from Step 01),
written in ~20 lines of numpy and used as an *analog forecaster* for Canadian CPI inflation:

    weights_i = softmax( <q, k_i> / (tau * sqrt(d)) )        forecast = y_T + sum_i weights_i * v_i

- query `q` = shape of the last 12 months of YoY inflation
- keys `k_i` = shapes of every past 12-month window fully observed at the origin (causal mask)
- values `v_i` = how inflation moved h months after each past window

This is Nadaraya–Watson kernel regression with an exponential kernel: `tau → 0` gives the nearest
neighbour, `tau → ∞` gives the plain average. The same backtest then compares the efficient-attention
variants that make DeepSeek-V4's 1M-token context affordable:

| variant | keys touched per query | idea in DeepSeek-V3.2 / V4 |
|---|---|---|
| dense | n | standard attention, cost grows with n |
| top-k sparse | k | "lightning indexer" scores all keys cheaply, attention reads only the top k |
| sliding window | w | local branch that keeps recent tokens uncompressed |
| compressed + top-k | k / block | CSA: pool blocks of keys (x4), then select top blocks; HCA pools x128 |

It also prints the 5 past episodes the latest window "attends to", which is a quick way to see what
the model thinks now resembles (useful as a validation and explanation tool, though attention weights
are not a full causal explanation).

**Files.** `step04/attention.py` (your TODOs) · `step04/check_step04.py` (tests) ·
`step04/run_step04.py` (backtest) · `step04/solution_attention.py` (reference)

**Run.**

    python step04/check_step04.py          # 5/5 PASS when your TODOs are right
    python step04/run_step04.py            # Bank of Canada Valet CPI, h=3
    python step04/run_step04.py --h 6 --tau 0.01

**What I learned.**
- Attention is a learned, data-dependent weighted average; softmax temperature controls how
  "nearest-neighbour" it is, and the max-subtraction trick keeps it numerically stable.
- Long context is expensive because dense attention reads every key; sparse (top-k) and compressed
  attention cut the keys read per query by orders of magnitude, trading off some accuracy.
- A pattern-matching forecaster rarely beats a no-change forecast for persistent series like inflation;
  the useful outputs are the accuracy-vs-cost trade-off and the analog episodes it retrieves.

**Try next.** Learn the projections: replace raw windows with `q = W_q x`, `k = W_k x` and fit
`W_q, W_k` (e.g. 12×4) by minimising backtest MSE with `scipy.optimize.minimize`, which turns this into
a one-head trained attention layer. Check whether it beats the fixed kernel out of sample.
