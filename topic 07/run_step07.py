"""Step 07 — Tabular foundation models vs the old way, on Canadian inflation.

Task: at each forecast origin t, predict y_t = pi_{t+6} - pi_t (change in headline CPI YoY
inflation over the next 6 months) from 9 macro features, using ONLY rows whose target is
already observed (expanding window). Score the 10/50/90% quantiles.

Models
  naive    no change (median 0), interval = context quantiles of y
  ols      the econometrics default: least squares + Gaussian residual interval
  gbdt     the ML default: 3 HistGradientBoosting quantile models, refit at every origin
  bayes    "PFN by hand": closed-form posterior predictive, prior picked by evidence (no gradient steps)
  tabpfn   optional (--tabpfn): the real tabular foundation model, one forward pass per origin

Usage
  python run_step07.py              # your tabular.py (NotImplementedError until TODOs are done)
  python run_step07.py solution     # reference
  options: --offline (synthetic data)  --tabpfn  --step 3 (months between origins)
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from data import FEATURES, build_supervised, load_monthly

warnings.filterwarnings("ignore")
H = 6

ap = argparse.ArgumentParser()
ap.add_argument("which", nargs="?", default="starter", choices=["starter", "solution"])
ap.add_argument("--offline", action="store_true")
ap.add_argument("--tabpfn", action="store_true")
ap.add_argument("--step", type=int, default=3)
ap.add_argument("--first", default="2005-01-01", help="first forecast origin")
args = ap.parse_args()

if args.which == "solution":
    import solution_tabular as T
else:
    import tabular as T

QS = T.QUANTILES


# ------------------------------------------------------------------ models
def m_naive(Xc, yc, Xq):
    lo, hi = np.quantile(yc, [QS[0], QS[2]])
    return np.array([[lo, 0.0, hi]] * len(Xq))


def m_ols(Xc, yc, Xq):
    Xs, Xqs = T.standardize(Xc, Xq)
    Z = np.c_[np.ones(len(Xs)), Xs]
    beta, *_ = np.linalg.lstsq(Z, yc, rcond=None)
    resid = yc - Z @ beta
    s2 = resid @ resid / max(len(yc) - Z.shape[1], 1)
    mean = np.c_[np.ones(len(Xqs)), Xqs] @ beta
    return T.gaussian_quantiles(mean, np.full(len(mean), s2))  # ignores parameter uncertainty


def m_gbdt(Xc, yc, Xq):
    from sklearn.ensemble import HistGradientBoostingRegressor
    out = []
    for q in QS:
        g = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200,
                                          learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=10,
                                          random_state=0)
        out.append(g.fit(Xc, yc).predict(Xq))
    return np.sort(np.stack(out, 1), axis=1)  # fix quantile crossing


def m_bayes(Xc, yc, Xq):
    Xs, Xqs = T.standardize(Xc, Xq)
    ybar = yc.mean()
    tau2, s2 = T.empirical_bayes(Xs, yc - ybar)
    mean, var = T.bayes_predictive(Xs, yc - ybar, Xqs, tau2, s2)
    return T.gaussian_quantiles(mean + ybar, var)


_TABPFN = None


def m_tabpfn(Xc, yc, Xq):
    global _TABPFN
    if _TABPFN is None:
        from tabpfn import TabPFNRegressor
        _TABPFN = TabPFNRegressor(device="cpu", random_state=0)
    _TABPFN.fit(Xc, yc)                        # "fit" = store the context; no weight updates
    qs = _TABPFN.predict(Xq, output_type="quantiles", quantiles=list(QS))
    return np.sort(np.stack(qs, 1), axis=1)


MODELS = {"naive": m_naive, "ols": m_ols, "gbdt": m_gbdt, "bayes": m_bayes}
if args.tabpfn:
    try:
        import tabpfn  # noqa: F401
        MODELS["tabpfn"] = m_tabpfn
    except ImportError:
        print("[tabpfn] not installed -> pip install tabpfn   (skipping)")


# ------------------------------------------------------------------ backtest
m, source = load_monthly(args.offline)
df = build_supervised(m, H)
print(f"data: {source}, {df.index[0]:%Y-%m} .. {df.index[-1]:%Y-%m}, {len(df)} rows, {len(FEATURES)} features, h={H}")

X = df[FEATURES].to_numpy(); y = df["target"].to_numpy(); trow = df["t"].to_numpy()
origins = [int(r) for r, d, yy in zip(trow, df.index, y)
           if d >= pd.Timestamp(args.first) and not np.isnan(yy)][:: args.step]

recs = []
timing = {k: 0.0 for k in MODELS}
for o in origins:
    ctx = T.context_mask(trow, o, H) & ~np.isnan(y)
    if ctx.sum() < 60:
        continue
    q = trow == o
    for name, f in list(MODELS.items()):
        t0 = time.perf_counter()
        try:
            pred = f(X[ctx], y[ctx], X[q])[0]
        except NotImplementedError:
            raise
        except Exception as e:  # noqa: BLE001  (e.g. TabPFN weights not downloadable)
            print(f"[{name}] failed: {type(e).__name__}: {e}  -> dropping model"); MODELS.pop(name); break
        timing[name] += time.perf_counter() - t0
        recs.append(dict(date=df.index[q][0], model=name, y=y[q][0], q10=pred[0], q50=pred[1], q90=pred[2],
                         n_ctx=int(ctx.sum())))

R = pd.DataFrame(recs)


def regime(d):
    if d < pd.Timestamp("2020-01-01"):
        return "2005-19"
    if d < pd.Timestamp("2024-01-01"):
        return "2020-23 surge"
    return "2024+"


R["regime"] = R["date"].map(regime)


def score(g):
    return pd.Series({
        "MAE(q50)": np.mean(np.abs(g.y - g.q50)),
        "pinball": np.mean([T.pinball_loss(g.y.values, g[c].values, q) for c, q in zip(["q10", "q50", "q90"], QS)]),
        "cover80": np.mean((g.y >= g.q10) & (g.y <= g.q90)),
        "width80": np.mean(g.q90 - g.q10),
    })


print(f"\n{len(R.date.unique())} origins every {args.step} months from {args.first[:7]}; context = rows with s+h <= origin\n")
tab = R.groupby("model").apply(score).loc[list(MODELS)]
print("ALL ORIGINS  (lower MAE/pinball better; cover80 should be near 0.80)")
print(tab.round(3).to_string())
print("\nPINBALL BY REGIME")
print(R.groupby(["regime", "model"]).apply(score)["pinball"].unstack()[list(MODELS)].round(3).to_string())
print("\nseconds per origin:", {k: round(v / max(len(R.date.unique()), 1), 3) for k, v in timing.items()})
R.to_csv("step07_backtest.csv", index=False)
print("\nwrote step07_backtest.csv")
