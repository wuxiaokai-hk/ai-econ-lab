## Step 07 — Tabular foundation models (TabPFN) vs per-dataset fitting

**What it shows.** Forecast the 6-month change in Canadian headline CPI inflation,
y_t = π_{t+6} − π_t, from 9 monthly macro features (π, its 3- and 12-month changes, CPI-trim,
trim gap, overnight rate, its 6-month change, real rate, 10y−2y slope; Bank of Canada Valet).
An expanding-window backtest (every 3 months from 2005) compares five models on
MAE, pinball loss (10/50/90% quantiles) and 80% interval coverage, by regime (2005–19, 2020–23 surge, 2024+).

| model | what it is |
|---|---|
| naive | no change; interval = past quantiles of y |
| ols | least squares + Gaussian residual band (the econometrics default) |
| gbdt | 3 gradient-boosted quantile models refit at every origin (the ML default) |
| bayes | "PFN by hand": closed-form posterior predictive, prior chosen by evidence — no gradient steps |
| tabpfn | optional (`--tabpfn`): TabPFN, one forward pass conditioned on the context table |

**The previous way.** Fit parameters to *this* dataset: θ̂ = argmin Σ L(y_i, f_θ(x_i)), then tune
hyperparameters with cross-validation. With ~250 monthly rows that means high variance, lots of tuning,
and intervals that ignore parameter uncertainty (OLS band uses σ̂² only).

**The new way.** A prior-data fitted network q_φ(y | x, D) is trained once on ~10⁸ synthetic datasets to
minimise E_{D∼prior}[−log q_φ(y* | x*, D)], so it learns to output the posterior predictive
p(y* | x*, D) = ∫ p(y* | x*, θ) p(θ | D) dθ. On your data it only *reads* the table.
`bayes` is the same idea with a linear-Gaussian prior, where the integral is exact:
mean = x*ᵀμ, var = σ² + x*ᵀΣx* (noise + parameter uncertainty).

**What I implemented (TODOs).**
1. `context_mask` — the leakage rule: row s is usable at origin t only if s + h ≤ t.
2. `log_marginal_likelihood` — evidence log N(y; 0, σ²I + τ²XXᵀ) via Cholesky, used to pick the prior.
3. `bayes_predictive` — posterior predictive mean/variance (mean = ridge with α = σ²/τ²).
4. `pinball_loss` — ρ_q(u) = max(qu, (q−1)u), the scoring rule for quantile forecasts.

**Run.**
```bash
cd step07
python check_step07.py            # 4 checks
python run_step07.py              # your version (Valet data; --offline = synthetic)
python run_step07.py solution     # reference
pip install tabpfn && python run_step07.py --tabpfn --step 6   # optional, CPU, a few minutes
```
Note: TabPFN code is Apache-2.0, but TabPFN-2.5+ **weights are non-commercial** (research/evaluation only) —
a real model-risk / vendor-licensing point for a bank.

**What I learned.** _(fill in after running: which model wins in 2005–19 vs the 2020–23 surge, and whether any
80% band actually covers ~80%.)_

**Try next.** Break the leakage rule on purpose (`t_rows <= origin`) and see how much the backtest flatters
every model — then write one paragraph on it as a validation finding.
