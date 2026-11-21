"""Pre-registered trend-following ETF test (docs/superpowers/specs/2026-11-21-trend-etf-protocol.md).

Default: discovery + validation only (holdout locked).
--holdout: the one-time 2024+ ETF test. Run only if Rule A passed validation; every load is logged.
"""
import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from swingbot import config
from swingbot.backtest.portfolio import metrics
from swingbot.backtest.trend import run_portfolio, target_weights, trend_signals
from swingbot.data import store

RULE_A = ["SPY", "EFA", "IEF", "VNQ", "DBC"]
CASH = "SHY"
PERIODS = {"discovery": (None, "2017-12-31"), "validation": ("2018-01-01", "2023-12-31"),
           "holdout": ("2024-01-01", None)}


def load(symbols, allow_holdout):
    o, c = {}, {}
    for s in symbols:
        try:
            b = store.load_bars(s, "stock", allow_holdout=allow_holdout)
        except FileNotFoundError:
            continue
        o[s], c[s] = b["open"], b["close"]
    return pd.DataFrame(o).sort_index(), pd.DataFrame(c).sort_index()


def strategy(opens, closes, assets, trend=True, require_all=False):
    sig = trend_signals(closes[assets])
    if require_all:  # protocol: Rule A starts when every asset has 10 month-end closes
        sig = sig[sig.notna().all(axis=1)]
    if not trend:
        sig = sig.where(sig.isna(), 1.0)
    w = target_weights(sig, cash=CASH)
    return run_portfolio(opens, closes, w)


def period_stats(eq, start, end):
    e = eq.loc[start:end]
    if len(e) < 40:
        return None
    m = metrics(e / e.iloc[0])
    return {k: m[k] for k in ("total_return", "cagr", "max_drawdown", "sharpe_monthly")}


def monthly_diff_t(a, b, start, end):
    ra = a.loc[start:end].resample("ME").last().pct_change().dropna()
    rb = b.loc[start:end].resample("ME").last().pct_change().dropna()
    d = (ra - rb).dropna()
    fit = sm.OLS(d.to_numpy(), np.ones((len(d), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return float(d.mean()), float(fit.tvalues[0]), float(fit.pvalues[0])


def passes(rule, bench):
    return bool(rule and bench and rule["total_return"] > 0 and rule["sharpe_monthly"] >= bench["sharpe_monthly"]
                and abs(rule["max_drawdown"]) < abs(bench["max_drawdown"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    etfs = pd.read_csv(config._REPO_ROOT / "universe" / "etfs.csv")["symbol"].tolist()
    if args.holdout:
        rec = config.DATA_DIR / "holdout_use_etf_trend.json"
        if rec.exists():
            raise SystemExit(f"ETF holdout already used: {rec.read_text()}")
        rec.write_text(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                   "who": "user request in chat (option 1: trend-following ETF test)",
                                   "rules": "Rule A Faber GTAA-5 (primary), Rule B all ETFs (secondary)",
                                   "protocol": "docs/superpowers/specs/2026-11-21-trend-etf-protocol.md"}, indent=2))
    opens, closes = load(etfs, allow_holdout=args.holdout)
    closes = closes.iloc[:-1] if args.holdout else closes  # newest bar may be partial
    opens = opens.loc[closes.index]
    breadth = [s for s in closes.columns if s != CASH]
    curves = {
        "Rule A: trend GTAA-5": strategy(opens, closes, RULE_A, require_all=True),
        "Bench 1: hold GTAA-5": strategy(opens, closes, RULE_A, trend=False, require_all=True),
        "Rule B: trend all ETFs": strategy(opens, closes, breadth),
        "Bench 1B: hold all ETFs": strategy(opens, closes, breadth, trend=False),
    }
    spy = closes["SPY"]
    start = curves["Rule A: trend GTAA-5"].index[0]
    curves = {k: v.loc[start:] for k, v in curves.items()}
    curves["SPY buy-and-hold"] = (spy.loc[start:] / spy.loc[start])
    periods = ["holdout"] if args.holdout else ["discovery", "validation"]
    for per in periods:
        s, e = PERIODS[per]
        print(f"\n=== {per} ({s or start.date()} to {e or closes.index[-1].date()}) ===")
        rows = {k: period_stats(v, s, e) for k, v in curves.items()}
        print(pd.DataFrame(rows).T.applymap(lambda x: f"{x:+.1%}" if abs(x) < 50 else f"{x:.2f}").to_string()
              if hasattr(pd.DataFrame, "applymap") else pd.DataFrame(rows).T.round(3).to_string())
        ok_a = passes(rows["Rule A: trend GTAA-5"], rows["Bench 1: hold GTAA-5"])
        ok_b = passes(rows["Rule B: trend all ETFs"], rows["Bench 1B: hold all ETFs"])
        d, t, p = monthly_diff_t(curves["Rule A: trend GTAA-5"], curves["Bench 1: hold GTAA-5"], s, e)
        print(f"Rule A vs Bench 1 monthly return difference: {d:+.2%}/month, t {t:.2f}, p {p:.2f} (information only)")
        word = "confirmed" if per == "holdout" else "PASS"
        print(f"Rule A {per}: {word if ok_a else 'not ' + word.lower() if per == 'holdout' else 'FAIL'}"
              f" | Rule B (secondary): {'pass' if ok_b else 'fail'}")


if __name__ == "__main__":
    main()
