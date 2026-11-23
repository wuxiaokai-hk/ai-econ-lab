"""Step 07 data: a small monthly Canadian macro table from the Bank of Canada Valet API.

Columns (monthly):
  cpi      CPI all-items index (V41690973)        -> headline YoY inflation pi_t
  trim     CPI-trim, YoY %  (CPI_TRIM)            -> core inflation
  rate     target overnight rate, % (V39079)      -> month-end value
  y10, y2  benchmark 10y / 2y GoC yields, %       -> month average

Supervised table (one row per month s, horizon h):
  target  y_s = pi_{s+h} - pi_s     (change in headline YoY inflation over the next h months)
  features observed at s only (no look-ahead).

If Valet cannot be reached (offline, proxy), a synthetic panel with the same columns is used
so the pipeline still runs. The printout says which one you got.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

VALET = "https://www.bankofcanada.ca/valet/observations/{code}/json?start_date={start}"
SERIES = {
    "cpi": "V41690973",
    "trim": "CPI_TRIM",
    "rate": "V39079",
    "y10": "BD.CDN.10YR.DQ.YLD",
    "y2": "BD.CDN.2YR.DQ.YLD",
}
FEATURES = ["pi", "d3_pi", "d12_pi", "trim", "trim_gap", "rate", "d6_rate", "real_rate", "slope"]


def _fetch_one(code: str, start: str = "1993-01-01") -> pd.Series:
    with urllib.request.urlopen(VALET.format(code=code, start=start), timeout=20) as r:
        js = json.load(r)
    rows = [(o["d"], o.get(code, {}).get("v")) for o in js["observations"]]
    s = pd.Series({pd.Timestamp(d): float(v) for d, v in rows if v not in (None, "")}, name=code)
    return s.sort_index()


def fetch_valet_monthly() -> pd.DataFrame:
    out = {}
    for name, code in SERIES.items():
        s = _fetch_one(code)
        if name == "rate":
            out[name] = s.resample("MS").last()
        else:
            out[name] = s.resample("MS").mean()
    return pd.DataFrame(out)


def synthetic_monthly(seed: int = 7) -> pd.DataFrame:
    """Fake but plausible panel (1993-2026) incl. a 2021-22 inflation surge. Offline fallback only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1993-01-01", "2026-10-01", freq="MS")
    n = len(idx)
    pi = np.zeros(n); trim = np.zeros(n); rate = np.zeros(n); slope = np.zeros(n)
    pi[0], trim[0], rate[0], slope[0] = 2.0, 2.0, 4.0, 1.0
    for t in range(1, n):
        shock = 0.0
        if idx[t].year == 2021 or (idx[t].year == 2022 and idx[t].month <= 6):
            shock = 0.35
        if idx[t].year == 2022 and idx[t].month > 6 or idx[t].year == 2023:
            shock = -0.25
        gap = rate[t - 1] - trim[t - 1] - 1.0          # policy stance vs neutral
        trim[t] = trim[t - 1] + 0.08 * (2 - trim[t - 1]) + 0.4 * shock + rng.normal(0, 0.08)
        # nonlinear: tight policy only bites when inflation is high
        pull = -0.05 * gap * (trim[t - 1] > 3)
        pi[t] = pi[t - 1] + 0.15 * (trim[t - 1] - pi[t - 1]) + shock + pull + rng.normal(0, 0.25)
        rate[t] = max(0.25, rate[t - 1] + 0.1 * (trim[t - 1] - 2) + rng.normal(0, 0.12))
        slope[t] = 0.9 * slope[t - 1] + 0.1 * (1.2 - 0.3 * (rate[t] - 2.5)) + rng.normal(0, 0.15)
    cpi = 100 * np.exp(np.cumsum(np.r_[np.zeros(12), pi[12:] / 1200.0]))  # rough level; we pass pi directly
    df = pd.DataFrame({"cpi": cpi, "trim": trim, "rate": rate, "y10": rate + slope + 0.5, "y2": rate + 0.5}, index=idx)
    df.attrs["pi_direct"] = pd.Series(pi, index=idx)
    return df


def load_monthly(offline: bool = False) -> tuple[pd.DataFrame, str]:
    if not offline:
        try:
            return fetch_valet_monthly(), "valet"
        except Exception as e:  # noqa: BLE001
            print(f"[data] Valet not reachable ({type(e).__name__}: {e}); using SYNTHETIC data.")
    return synthetic_monthly(), "synthetic"


def build_supervised(m: pd.DataFrame, h: int = 6) -> pd.DataFrame:
    """One row per month s with features known at s and target pi_{s+h} - pi_s (NaN for the last h rows)."""
    if "pi_direct" in m.attrs:
        pi = m.attrs["pi_direct"]
    else:
        pi = 100 * (m["cpi"] / m["cpi"].shift(12) - 1)
    df = pd.DataFrame(index=m.index)
    df["pi"] = pi
    df["d3_pi"] = pi - pi.shift(3)
    df["d12_pi"] = pi - pi.shift(12)
    df["trim"] = m["trim"]
    df["trim_gap"] = m["trim"] - pi
    df["rate"] = m["rate"]
    df["d6_rate"] = m["rate"] - m["rate"].shift(6)
    df["real_rate"] = m["rate"] - m["trim"]
    df["slope"] = m["y10"] - m["y2"]
    df["target"] = pi.shift(-h) - pi
    df = df.dropna(subset=FEATURES)
    df["t"] = np.arange(len(df))  # integer month index used for the leakage rule
    return df
