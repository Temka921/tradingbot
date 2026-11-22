"""Figures for the paper and the Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014).

Crypto data are used only before 2024 (the crypto holdout is still unused). The trend figure
re-reads the ETF holdout, which was already spent on the pre-registered trend test (logged).
Writes paper/figures/*.pdf and paper/data/deflated_sharpe.csv.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from swingbot import config  # noqa: E402
from swingbot.analysis.views import universe_views  # noqa: E402
from swingbot.backtest.portfolio import simulate  # noqa: E402
from swingbot.data import foreign, store  # noqa: E402
from swingbot.strategies import carry_run, run  # noqa: E402
import trend_etf  # noqa: E402

FIG = ROOT / "paper" / "figures"
DATA = ROOT / "paper" / "data"
CUT = pd.Timestamp(config.HOLDOUT_START)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def rebase(s):
    s = s.dropna()
    return s / s.iloc[0]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def monthly(eq, start, end):
    e = eq.loc[start:end].dropna()
    return e.resample("ME").last().pct_change().dropna()


def deflated_sharpe(r: pd.Series, n_trials: int, var_sr: float) -> dict:
    """DSR of the per-period Sharpe of returns r, given the number and variance of trial Sharpes."""
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    gamma = 0.5772156649
    sr0 = np.sqrt(var_sr) * ((1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
                             + gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return {"months": T, "sharpe_annual": sr * np.sqrt(12), "sr0_annual": sr0 * np.sqrt(12),
            "skew": g3, "kurtosis": g4, "dsr": float(stats.norm.cdf(z))}


def main():
    curves = {}
    # --- crypto: C1, C3 (2018-2023), F1, F2 (2020-2023)
    panel = pd.read_parquet(config.DATA_DIR / "panel" / "crypto.parquet", columns=["date", "asset", "eligible"])
    panel = panel[panel["date"] < CUT]
    o, c, el = run.crypto_inputs(panel, allow_holdout=False)
    fng = foreign.load_fng()
    fng = fng[fng.index < CUT]
    mode = json.loads((ROOT / "paper" / "data" / "originals_c3_choice.json").read_text())["c3_mode"]
    curves["C3"], _ = run.run_crypto("C3", o, c, el, fng, mode)
    curves["C1"], _ = run.run_crypto("C1", o, c, el)
    btc = c["BTCUSDT"]
    f = carry_run.load_inputs(panel, ("F1", "F2"), allow_holdout=False)
    curves["F1"] = carry_run.run_strategy("F1", f)["equity"]
    curves["F2"] = carry_run.run_strategy("F2", f)["equity"]
    del panel, o, el, f

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for name, s in (("C3: mood + momentum", curves["C3"]), ("C1: momentum", curves["C1"]),
                    ("BTC buy-and-hold", btc.loc["2018-03-04":])):
        r = rebase(s.loc["2018-03-04":"2023-12-31"])
        ax.plot(r.index, r.values, label=name, lw=1)
    ax.axvline(pd.Timestamp("2022-01-01"), color="k", ls="--", lw=0.8)
    ax.text(pd.Timestamp("2022-02-01"), ax.get_ylim()[1] * 0.5, "validation", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("Growth of 1 (log)"); ax.legend(fontsize=8)
    ax.set_title("C3 looked excellent in development and collapsed afterwards")
    save(fig, "fig_c3")

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for name, s in (("F1: BTC+ETH carry", curves["F1"]), ("F2: rotating carry", curves["F2"])):
        r = rebase(s.loc["2020-01-05":"2023-12-31"])
        ax.plot(r.index, r.values, label=name, lw=1)
    b = rebase(btc.loc["2020-01-05":"2023-12-31"])
    ax2 = ax.twinx(); ax2.plot(b.index, b.values, color="grey", lw=0.6, alpha=0.6, label="BTC (right axis)")
    ax2.grid(False); ax2.set_ylabel("BTC growth of 1")
    ax.axvline(pd.Timestamp("2022-01-01"), color="k", ls="--", lw=0.8)
    ax.set_ylabel("Growth of 1"); ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
    ax.set_title("Funding-rate carry: smooth, but income fell after 2021")
    save(fig, "fig_carry")

    # --- US stocks: S2 and the RSI(14) portfolio (pre-2024 only)
    sp = pd.read_parquet(config.DATA_DIR / "panel" / "stock.parquet")
    bars, idx, home = run.us_stock_inputs(sp)
    curves["S2"], ewh, _ = run.s2_run(bars, idx, home, config.COSTS["stock"] / 2)
    curves["S2_ewh"] = ewh
    del bars
    view = universe_views({"stock": sp})["stock"]
    del sp
    _, eq_r = simulate(view, "rsi14_oversold", "long", "validation", 10, 10, 0)
    _, eq_b = simulate(view, "__all__", "long", "validation", 10, 10, 0)
    _, eq_rd = simulate(view, "rsi14_oversold", "long", "discovery", 10, 10, 0)
    curves["RSI_disc"] = eq_rd
    del view
    spy = store.load_bars("SPY", "stock")["close"]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for name, s in (("RSI(14) oversold portfolio", eq_r), ("Random stocks, same rules", eq_b),
                    ("SPY buy-and-hold", spy.loc["2018-01-02":"2023-12-29"])):
        r = rebase(s)
        ax.plot(r.index, r.values, label=name, lw=1)
    ax.set_ylabel("Growth of 1"); ax.legend(fontsize=8)
    ax.set_title("RSI(14) oversold on U.S. stocks, validation 2018-2023 (seed 0, 10 slots)")
    save(fig, "fig_rsi")

    # --- trend rule (ETF holdout already spent on this test)
    etfs = pd.read_csv(ROOT / "universe" / "etfs.csv")["symbol"].tolist()
    eo, ec = trend_etf.load(etfs, allow_holdout=True)
    ec = ec.iloc[:-1]; eo = eo.loc[ec.index]
    tr = trend_etf.strategy(eo, ec, trend_etf.RULE_A, require_all=True)
    hd = trend_etf.strategy(eo, ec, trend_etf.RULE_A, trend=False, require_all=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for name, s in (("Trend rule (10-month SMA)", tr), ("Hold the same 5 ETFs", hd),
                    ("SPY buy-and-hold", ec["SPY"].loc[tr.index[0]:])):
        r = rebase(s)
        ax.plot(r.index, r.values, label=name, lw=1)
    for d in ("2018-01-01", "2024-01-01"):
        ax.axvline(pd.Timestamp(d), color="k", ls="--", lw=0.8)
    ax.set_yscale("log"); ax.set_ylabel("Growth of 1 (log)"); ax.legend(fontsize=8)
    ax.set_title("Trend rule: smaller crashes, lower growth (dashed: validation, holdout)")
    save(fig, "fig_trend")
    curves["TrendA"] = tr

    # --- Deflated Sharpe Ratio for the strongest discovery results
    trial_sharpes = {"RSI(14) portfolio": None, "Trend rule A": 0.670, "Trend rule B": 0.907, "C1": 0.660,
                     "C2": 0.139, "C3 (confirming)": 1.021, "C3 (contrarian)": -0.708, "S1": 0.906,
                     "S2": 1.257, "F1": 3.663, "F2": 2.762}
    disc = {"C3 (confirming)": monthly(curves["C3"], "2018-03-04", "2021-12-31"),
            "S2": monthly(curves["S2"], "2001-01-31", "2017-12-29"),
            "F1": monthly(curves["F1"], "2020-01-05", "2021-12-31"),
            "RSI(14) portfolio": monthly(curves["RSI_disc"], "2000-01-01", "2017-12-31"),
            "Trend rule A": monthly(curves["TrendA"], "2006-12-01", "2017-12-31")}
    r = disc["RSI(14) portfolio"]
    trial_sharpes["RSI(14) portfolio"] = float(r.mean() / r.std() * np.sqrt(12))
    sr_m = np.array(list(trial_sharpes.values())) / np.sqrt(12)
    var_sr, n = float(sr_m.var(ddof=1)), len(sr_m)
    rows = []
    for name, ret in disc.items():
        d = deflated_sharpe(ret, n, var_sr)
        rows.append({"strategy": name, "n_trials": n, **d})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(DATA / "deflated_sharpe.csv", index=False)
    pd.Series(trial_sharpes).to_csv(DATA / "deflated_sharpe_trials.csv", header=["discovery_sharpe_annual"])


if __name__ == "__main__":
    main()
