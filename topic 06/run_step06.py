"""
Step 06 - Train a GRPO policy that picks a CPI forecasting rule per inflation state.

    python run_step06.py            # uses your grpo.py
    python run_step06.py solution   # uses solution_grpo.py

Data: Bank of Canada Valet, CPI all-items index V41690973 (monthly).
If Valet is unreachable, a clearly-labelled SYNTHETIC series is used so the
script still runs (numbers then mean nothing economically).
"""
import sys, json, importlib, urllib.request
import numpy as np
import pandas as pd

mod = importlib.import_module("solution_grpo" if len(sys.argv) > 1 else "grpo")
softmax = mod.softmax

H, TOL, G, BETA, LR, ITERS, BATCH = 3, 0.5, 8, 0.02, 0.5, 400, 32
SPLIT = "2020-01-01"          # train <= 2019, test 2020+ (includes the 2021-23 surge)
RULES = ["naive", "momentum", "revert_2pct", "avg_12m", "damped_trend"]


def load_cpi():
    url = "https://www.bankofcanada.ca/valet/observations/V41690973/json?start_date=1995-01-01"
    try:
        with urllib.request.urlopen(url, timeout=20) as f:
            obs = json.load(f)["observations"]
        s = pd.Series({pd.Timestamp(o["d"]): float(o["V41690973"]["v"]) for o in obs}).sort_index()
        return s, "Valet V41690973"
    except Exception as e:  # offline fallback
        print(f"[warn] Valet unreachable ({type(e).__name__}); using SYNTHETIC CPI")
        rng = np.random.default_rng(0)
        idx = pd.date_range("1995-01-01", "2025-10-01", freq="MS")
        infl = np.zeros(len(idx)); infl[0] = 2.0
        for t in range(1, len(idx)):
            target = 6.0 if idx[t].year in (2021, 2022) else 2.0
            infl[t] = infl[t-1] + 0.15 * (target - infl[t-1]) + rng.normal(0, 0.6)
        lvl = 100 * np.exp(np.cumsum(np.log1p(infl / 1200)) + rng.normal(0, 0.003, len(idx)))
        return pd.Series(lvl, idx), "SYNTHETIC"


def build_dataset(cpi):
    y = (cpi / cpi.shift(12) - 1) * 100          # YoY inflation, %
    d3 = y - y.shift(3)
    fc = pd.DataFrame({
        "naive": y,
        "momentum": y + d3,
        "revert_2pct": y + 0.5 * (2.0 - y),
        "avg_12m": y.rolling(12).mean(),
        "damped_trend": y + 0.5 * d3,
    })[RULES]
    target = y.shift(-H)
    level = np.digitize(y, [1.0, 3.0])             # 0: <1%, 1: 1-3%, 2: >3%
    state = level * 2 + (d3 > 0).astype(int)       # x rising/falling -> 6 prompts
    df = pd.concat([fc, target.rename("target"), state.rename("state")], axis=1).dropna()
    R = (np.abs(df[RULES].values - df["target"].values[:, None]) <= TOL).astype(float)
    return df, R


def success(theta, states, R):
    """Expected verifiable reward = sum_k pi(k|s_t) * r_tk, averaged over dates."""
    P = softmax(theta[states])
    return float(np.mean(np.sum(P * R, axis=1)))


def main():
    cpi, src = load_cpi()
    df, R = build_dataset(cpi)
    st = df["state"].values.astype(int)
    tr = df.index < SPLIT
    S, K = 6, len(RULES)
    ref = np.full((S, K), 1.0 / K)                 # reference policy: uniform
    idx_tr = np.where(tr)[0]

    def train(beta, seed=42):
        theta = np.zeros((S, K)); rng = np.random.default_rng(seed); dead = 0
        for it in range(ITERS):
            for t in rng.choice(idx_tr, size=BATCH):   # each date = one "prompt"
                s = st[t]
                theta[s], d = mod.grpo_step(theta[s], ref[s], R[t], G, beta, LR, rng)
                dead += d
        return theta, dead / (ITERS * BATCH)

    theta, dead = train(BETA)
    print(f"data: {src}, {df.index[0]:%Y-%m} .. {df.index[-1]:%Y-%m}, h={H}, tol=+/-{TOL}pp")
    print(f"dead groups (all-same reward, zero gradient): {dead:.1%}\n")

    rows = []
    for name, mask in [("train <=2019", tr), ("test 2020+", ~tr)]:
        row = {"sample": name,
               "ref (uniform)": success(np.zeros((S, K)), st[mask], R[mask]),
               "GRPO policy": success(theta, st[mask], R[mask])}
        for k, r in enumerate(RULES):
            row[r] = R[mask, k].mean()
        rows.append(row)
    print(pd.DataFrame(rows).set_index("sample").round(3).T.to_string())

    labels = ["<1% falling", "<1% rising", "1-3% falling", "1-3% rising", ">3% falling", ">3% rising"]
    pol = pd.DataFrame(softmax(theta), index=labels, columns=RULES).round(2)
    pol["n_train"] = [int(np.sum(st[tr] == s)) for s in range(S)]
    print("\nKL strength vs success (the validation question: does amplification survive a regime shift?)")
    for b in [0.02, 0.2, 1.0, 5.0]:
        th, _ = train(b)
        print(f"  beta={b:<5} train={success(th, st[tr], R[tr]):.3f}  test={success(th, st[~tr], R[~tr]):.3f}")

    print("\nLearned policy pi(rule | state):\n" + pol.to_string())


if __name__ == "__main__":
    main()
