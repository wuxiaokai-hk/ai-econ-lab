"""ai-econ-lab · Step 01 — STARTER (your version)
Zero-shot forecasting of Canadian CPI with a time-series foundation model (Chronos-2),
judged against classic baselines with a leakage-free rolling-origin backtest.

Fill in the 4 `# TODO (you):` blocks (~25 min), then run the checks.

Run:
    python check_step01.py                   # offline unit checks (seconds)
    python step01_chronos_cpi.py                    # real data + Chronos-2 (first run downloads ~0.5 GB)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `lab`

Forecast = dict  # {"median": np.ndarray, "lo": np.ndarray | None, "hi": np.ndarray | None}
Forecaster = Callable[[np.ndarray, int], Forecast]


# ---------------------------------------------------------------- baselines
def naive(history: np.ndarray, h: int) -> Forecast:
    """Random walk: every future value = last observed value."""
    return {"median": np.repeat(history[-1], h), "lo": None, "hi": None}


def seasonal_naive(history: np.ndarray, h: int, m: int = 12) -> Forecast:
    """y_hat[T+k] = y[T+k-m*ceil(k/m)]: repeat the last full season. Works for h > m too."""
    history = np.asarray(history, dtype=float)
    # TODO (you): forecast = the last m observations, repeated until you have h values.
    # Hint: last_season = history[-m:]; np.tile(...) then slice [:h].
    # Check: history 1..24, m=12, h=15  ->  13,14,...,24,13,14,15
    median = ...
    return {"median": median, "lo": None, "hi": None}


# ---------------------------------------------------------------- metrics
def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 12) -> float:
    """MASE = mean|y - y_hat| / mean_t |y_t - y_{t-m}|  (scale from TRAINING data only)."""
    y_true, y_pred, y_train = (np.asarray(a, dtype=float) for a in (y_true, y_pred, y_train))
    # TODO (you): implement MASE.
    #   numerator   = mean |y_true - y_pred|                  (out-of-sample MAE)
    #   denominator = mean |y_train[t] - y_train[t-m]|        (in-sample seasonal-naive MAE)
    # Hint: the seasonal differences are y_train[m:] - y_train[:-m].
    # Why train only? The scale must be known at forecast time -> no peeking at the test set.
    raise NotImplementedError("mase")


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of actuals inside [lo, hi]. For a q10–q90 band the target is 0.80."""
    y_true, lo, hi = (np.asarray(a, dtype=float) for a in (y_true, lo, hi))
    # TODO (you): fraction of y_true with lo <= y <= hi (inclusive). One line with np.mean.
    raise NotImplementedError("interval_coverage")


# ---------------------------------------------------------------- the foundation model
class ChronosForecaster:
    """Zero-shot Chronos-2: no training, the model just reads the history (in-context)."""

    def __init__(self, model_id: str = "amazon/chronos-2", q_lo: float = 0.1, q_hi: float = 0.9):
        from chronos import Chronos2Pipeline  # pip install "chronos-forecasting>=2.1"

        self.pipe = Chronos2Pipeline.from_pretrained(model_id, device_map="cpu")
        self.q = [q_lo, 0.5, q_hi]

    def __call__(self, history: np.ndarray, h: int) -> Forecast:
        quantiles, _ = self.pipe.predict_quantiles(
            [np.asarray(history, dtype=np.float32)], prediction_length=h, quantile_levels=self.q
        )
        q = quantiles[0][0].numpy()  # (n_variates=1, h, 3) -> (h, 3)
        return {"lo": q[:, 0], "median": q[:, 1], "hi": q[:, 2]}


# ---------------------------------------------------------------- backtest
def rolling_origin_backtest(
    y: pd.Series,
    forecasters: dict[str, Forecaster],
    origins: list[pd.Timestamp],
    h: int = 12,
    m: int = 12,
) -> pd.DataFrame:
    """For each origin T: fit on y[:T] (inclusive), forecast T+1..T+h, score vs actuals.

    Returns one row per (model, origin) with columns: model, origin, mase, coverage.
    """
    rows = []
    for origin in origins:
        # TODO (you): split the series at this origin WITHOUT look-ahead.
        #   train = everything up to AND INCLUDING `origin`
        #   test  = the next h observations strictly AFTER `origin`
        # Hint: y.loc[:origin] is inclusive in pandas; for test try y.loc[origin:].iloc[1 : h + 1].
        # Off-by-one here is the #1 backtest bug in real model validation work.
        train = ...
        test = ...
        if len(test) < h:
            continue                                            # not enough future to score
        for name, f in forecasters.items():
            fc = f(train, h)
            # TODO (you, part 2): score this forecast.
            #   cov   = interval_coverage(test, lo, hi) if fc.get("lo") is not None else np.nan
            #   score = mase(test, fc["median"], train, m)     <- scale uses TRAIN only
            cov = ...
            score = ...
            rows.append(
                {
                    "model": name,
                    "origin": origin,
                    "mase": score,
                    "coverage": cov,
                }
            )
    return pd.DataFrame(rows)


def regime(ts: pd.Timestamp) -> str:
    if ts < pd.Timestamp("2021-01-01"):
        return "1_pre-2021 (calm)"
    if ts < pd.Timestamp("2023-01-01"):
        return "2_2021-22 (surge)"
    return "3_2023+ (disinflation)"


def main() -> None:
    from lab.valet import get_series

    y = get_series("V41690973")  # Total CPI, NSA, monthly
    print(f"CPI: {y.index[0]:%Y-%m} .. {y.index[-1]:%Y-%m}, n={len(y)}")

    h = 12
    origins = list(pd.date_range("2017-01-01", y.index[-1 - h], freq="3MS"))
    forecasters: dict[str, Forecaster] = {
        "naive": naive,
        "seasonal_naive": seasonal_naive,
        "chronos2": ChronosForecaster(),
    }
    res = rolling_origin_backtest(y, forecasters, origins, h=h)
    res["regime"] = res["origin"].map(regime)

    print("\nMean MASE by model (lower is better; <1 beats in-sample seasonal naive):")
    print(res.groupby("model")[["mase", "coverage"]].mean().round(3))
    print("\nMean MASE by regime (origin date):")
    print(res.pivot_table(index="regime", columns="model", values="mase").round(2))
    print("\nChronos-2 80% band coverage by regime (target 0.80):")
    print(res[res.model == "chronos2"].groupby("regime")["coverage"].mean().round(2))

    out = Path(__file__).with_name("results_step01.csv")
    res.to_csv(out, index=False)
    print(f"\nSaved {out.name}")

    try:  # optional picture of one origin in the surge
        import matplotlib.pyplot as plt

        o = pd.Timestamp("2021-04-01")
        fc = forecasters["chronos2"](y.loc[:o].to_numpy(), h)
        idx = pd.date_range(o, periods=h + 1, freq="MS")[1:]
        ax = y.loc["2018":].plot(label="CPI actual", figsize=(8, 4))
        ax.plot(idx, fc["median"], label="Chronos-2 median")
        ax.fill_between(idx, fc["lo"], fc["hi"], alpha=0.3, label="q10–q90")
        ax.plot(idx, seasonal_naive(y.loc[:o].to_numpy(), h)["median"], "--", label="seasonal naive")
        ax.axvline(o, color="grey", lw=0.8)
        ax.legend()
        ax.set_title("Forecast from April 2021: did anyone see the surge?")
        plt.tight_layout()
        plt.savefig(Path(__file__).with_name("forecast_2021-04.png"), dpi=150)
        print("Saved forecast_2021-04.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
