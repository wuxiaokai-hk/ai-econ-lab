## Step 06 — GRPO by hand: RL with verifiable rewards on CPI forecasting rules

**What it shows.** The training trick behind today's reasoning models (DeepSeek-R1 and successors):
sample a *group* of G answers per prompt, check each with a rule (reward 0/1), and push up the
answers that beat their own group's average. Here a "prompt" is an inflation state (level × direction,
6 states), an "answer" is one of 5 forecasting rules (naive, momentum, revert-to-2%, 12-mo average,
damped trend), and the verifier is reality: reward = 1 if the 3-month-ahead CPI YoY forecast is
within ±0.5 pp (Valet V41690973).

**Core maths.**
- Advantage: `A_i = (r_i − mean(r)) / std(r)`. With binary rewards and k of G correct (p = k/G):
  `A+ = √((1−p)/p)`, `A− = −√(p/(1−p))`. 2 of 8 right → +1.73 vs −0.58: rare successes get big weight.
- Policy gradient for a softmax: `∇ log π(a) = e_a − π`.
- KL anchor: `∇ KL(π‖π_ref) = π ⊙ (log π − log π_ref − KL)`.
- Update: `θ ← θ + lr · [ mean_i A_i ∇log π(a_i) − β ∇KL ]`.

**Run.**
```bash
cd step06
python check_step06.py           # your grpo.py: 4 checks
python run_step06.py             # train on <=2019, test on 2020+
python check_step06.py solution  # compare with the reference solution
```

**What I learned.**
- GRPO needs no value network: the group mean *is* the baseline.
- Groups where all answers score the same give zero gradient ("dead groups") — most of the
  compute can be wasted, which is why later variants (e.g. DAPO's dynamic sampling) filter them.
- In-sample, GRPO amplifies success above the reference (Mroueh, ICML 2026, proves a fixed point
  p* > p_ref). Out of sample, across the 2021–23 regime shift, a weak KL anchor (small β) can
  overfit and do *worse* than the reference; larger β trades amplification for robustness.
  That is a model-validation finding, not just an ML one.

(Offline test run used a synthetic fallback series — train 0.455 → 0.562, test 0.463 → 0.344 at
β = 0.02; your real-Valet numbers will differ.)

**Try next.** Replace the 0/1 reward with a graded verifiable reward `r = −|ŷ − y|` (pinball loss
for quantiles) and see whether dead groups disappear and test success improves.
