"""
Step 03 runner: conformal intervals around a naive forecast of Canadian CPI inflation.

    python step03/run_step03.py               # uses your step03/conformal.py
    python step03/run_step03.py --solution    # uses the reference solution
    python step03/run_step03.py --synthetic   # offline demo (no Valet call)

Data: Bank of Canada Valet, series V41690973 (CPI all-items, same as Step 01),
cached to data/cpi_V41690973.csv. Target: 12-month inflation in %, monthly.
Forecaster: naive (next month's YoY inflation = this month's). Swap in Step 01's
Chronos median later — conformal does not care what the forecaster is.
"""
import importlib.util
import json
import pathlib
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE = ROOT / "step03"
DATA = ROOT / "data"
SERIES = "V41690973"

ALPHA, GAMMA, WINDOW = 0.10, 0.02, 60   # 90% intervals, ACI step, 5-year calibration window
REGIMES = {
    "2000-2019 calm": ("2000-01-01", "2019-12-31"),
    "2020 COVID":     ("2020-01-01", "2020-12-31"),
    "2021-23 surge":  ("2021-01-01", "2023-12-31"),
    "2024+ disinfl.": ("2024-01-01", "2030-12-31"),
}


def load_module():
    name = "solution_step03_conformal.py" if "--solution" in sys.argv else "conformal.py"
    spec = importlib.util.spec_from_file_location("conf", HERE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"Using {name}")
    return mod


def fetch_cpi() -> pd.Series:
    DATA.mkdir(exist_ok=True)
    cache = DATA / f"cpi_{SERIES}.csv"
    if cache.exists():
        s = pd.read_csv(cache, parse_dates=["date"]).set_index("date")["cpi"]
        return s
    url = f"https://www.bankofcanada.ca/valet/observations/{SERIES}/json"
    with urllib.request.urlopen(url, timeout=30) as r:
        obs = json.load(r)["observations"]
    s = pd.Series({pd.Timestamp(o["d"]): float(o[SERIES]["v"]) for o in obs if o.get(SERIES)}, name="cpi")
    s.index.name = "date"
    s.sort_index().to_frame().to_csv(cache)
    return s.sort_index()


def synthetic_cpi() -> pd.Series:
    """Offline stand-in: calm 2%, a 2021–23 style surge, then disinflation."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("1995-01-01", "2026-09-01", freq="MS")
    infl = np.full(len(idx), 2.0)
    shock = (idx >= "2021-03-01") & (idx <= "2023-06-01")
    infl[shock] += np.linspace(0, 5.5, shock.sum()) * np.sin(np.linspace(0, np.pi, shock.sum()))
    m = 1 + (infl + rng.normal(0, 0.25, len(idx))) / 1200
    return pd.Series(100 * np.cumprod(m), idx, name="cpi")


def main():
    conf = load_module()
    cpi = synthetic_cpi() if "--synthetic" in sys.argv else fetch_cpi()
    infl = (100 * (cpi / cpi.shift(12) - 1)).dropna()
    infl = infl[infl.index >= "1990-01-01"]
    y = infl.values
    yhat = np.r_[np.nan, y[:-1]]        # naive h=1 forecast
    y, yhat, dates = y[1:], yhat[1:], infl.index[1:]

    lo_s, hi_s, miss_s = conf.split_conformal_intervals(y, yhat, ALPHA, WINDOW)
    lo_a, hi_a, miss_a, a_path = conf.aci_intervals(y, yhat, ALPHA, GAMMA, WINDOW)

    print(f"\nTarget 90% coverage. Window={WINDOW} months, gamma={GAMMA}. Last obs {dates[-1]:%Y-%m}.\n")
    for label, lo, hi, miss in [("Split conformal (fixed alpha)", lo_s, hi_s, miss_s),
                                ("Adaptive conformal (ACI)", lo_a, hi_a, miss_a)]:
        print(label)
        tab = conf.coverage_table(dates, y, lo, hi, miss, REGIMES)
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"  overall coverage: {1 - np.nanmean(miss):.3f}\n")

    out = pd.DataFrame({"date": dates, "infl_yoy": y, "naive_fcst": yhat,
                        "split_lo": lo_s, "split_hi": hi_s, "aci_lo": lo_a, "aci_hi": hi_a,
                        "aci_alpha_t": a_path})
    DATA.mkdir(exist_ok=True)
    out.to_csv(DATA / "step03_intervals.csv", index=False)
    print(f"Wrote {DATA / 'step03_intervals.csv'}")

    try:
        import matplotlib.pyplot as plt
        d = out[out.date >= "2018-01-01"]
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, height_ratios=[3, 1])
        ax[0].plot(d.date, d.infl_yoy, "k", lw=1.2, label="CPI inflation (YoY %)")
        ax[0].fill_between(d.date, d.split_lo, d.split_hi, alpha=0.25, label="split 90%")
        ax[0].fill_between(d.date, d.aci_lo, d.aci_hi.clip(upper=d.infl_yoy.max() + 3), alpha=0.25, label="ACI 90%")
        ax[0].legend(loc="upper left"); ax[0].set_ylabel("%")
        ax[1].plot(d.date, d.aci_alpha_t); ax[1].axhline(ALPHA, ls="--", c="grey")
        ax[1].set_ylabel("ACI alpha_t")
        fig.tight_layout(); fig.savefig(DATA / "step03_intervals.png", dpi=130)
        print(f"Wrote {DATA / 'step03_intervals.png'}")
    except ImportError:
        print("(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
