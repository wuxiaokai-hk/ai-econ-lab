# ai-econ-lab

Small, honest experiments applying current AI methods to Canadian macro and finance data —
built one step at a time, each step judged the way a model-validation team would judge it.

*Author: Xiaokai Wu, PhD (Economics). Everything runs on a laptop with open models and public data
(Bank of Canada Valet API); no paid API keys.*

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Repo layout

```
lab/            shared helpers (valet.py = Bank of Canada data client, cached to data/)
steps/stepNN_*  one folder per step: starter, check_*.py, solution_*.py, results
```

## Step 01 — Zero-shot CPI forecasting with a time-series foundation model

**Question.** Can a pretrained time-series foundation model (Amazon's Chronos-2), with *no training on
Canadian data*, forecast Canada's CPI 12 months ahead better than textbook baselines, and are its
uncertainty bands honest?

**What it does.**
- Pulls Total CPI (NSA, monthly, Valet series `V41690973`).
- Rolling-origin backtest: every 3 months since 2017, forecast the next 12 months using only data
  up to that month (no look-ahead).
- Compares Chronos-2 (zero-shot, in-context) with naive and seasonal-naive baselines.
- Scores point forecasts with **MASE** and the q10–q90 band with **coverage** (target 80%),
  split by regime: calm (pre-2021), surge (2021–22), disinflation (2023+).

**Run.**
```bash
cd steps/step01_chronos_cpi
python check_step01.py          # offline unit checks
python step01_chronos_cpi.py    # real data + Chronos-2 (first run downloads the model)
```

**What I learned.**
- A foundation model turns forecasting from *fitting* into *conditioning*: the model reads the
  history as context and outputs quantiles in one forward pass.
- Leaderboard gaps between models are small; the decision has to come from a backtest on your own
  series, against simple baselines, with leakage-free splits.
- Point accuracy and calibration are different questions. The 2021–22 surge is the stress test:
  check whether coverage collapses when the regime changes.

**Results.** *(fill in from `results_step01.csv`)*

| model | mean MASE | 80% band coverage |
|---|---|---|
| naive | | – |
| seasonal_naive | | – |
| chronos2 | | |

**Try next.** Forecast seasonally adjusted CPI (`V41690914`) or y/y inflation instead of the
level, and add a stronger classical benchmark (ETS or ARIMA via `statsforecast`). Does Chronos-2
still win?
