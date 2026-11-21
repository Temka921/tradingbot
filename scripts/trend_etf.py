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
    rec = config.DATA_DIR / "holdout_use_etf_trend.json"
    if args.holdout:
        missing = [s for s in RULE_A + [CASH] if not (config.DATA_DIR / "raw" / "stock" / f"{s}.parquet").exists()]
        if missing:
            raise SystemExit(f"missing data files, holdout NOT used: {missing}")
        if rec.exists():
            raise SystemExit(f"ETF holdout already used: {rec.read_text()}")
        rec.write_text(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                   "who": "user request in chat (option 1: trend-following ETF test)",
                                   "rules": "Rule A Faber GTAA-5 (primary), Rule B all ETFs (secondary)",
                                   "protocol": "docs/superpowers/specs/2026-11-21-trend-etf-protocol.md"}, indent=2))
    try:
        results = run(etfs, args.holdout)
    except Exception as e:  # a crash after the record still counts as the one use: leave a trace
        if args.holdout:
            rec.write_text(json.dumps({**json.loads(rec.read_text()), "error": f"{type(e).__name__}: {e}"}, indent=2))
        raise
    if args.holdout:
        rec.write_text(json.dumps({**json.loads(rec.read_text()), "results": results}, indent=2, default=float))
        print(f"\nresults saved to {rec}")


def run(etfs, holdout):
    opens, closes = load(etfs, allow_holdout=holdout)
    closes = closes.iloc[:-1] if holdout else closes  # newest bar may be partial
    opens = opens.loc[closes.index]
    if holdout:
        h = closes.loc["2024-01-01":, RULE_A + [CASH]]
        print("holdout data check: last date", h.index[-1].date(), "| NaN counts", h.isna().sum().to_dict())
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
    periods = ["holdout"] if holdout else ["discovery", "validation"]
    shy = closes[CASH] / closes[CASH].loc[start:].iloc[0]
    out = {}
    for per in periods:
        s, e = PERIODS[per]
        print(f"\n=== {per} ({s or start.date()} to {e or closes.index[-1].date()}) ===")
        rows = {k: period_stats(v, s, e) for k, v in curves.items()}
        for k, v in curves.items():  # information only: Sharpe of monthly returns in excess of SHY
            if rows[k]:
                rv = v.loc[s:e].resample("ME").last().pct_change().dropna()
                rs = shy.loc[s:e].resample("ME").last().pct_change().reindex(rv.index)
                x = (rv - rs).dropna()
                rows[k]["sharpe_vs_shy"] = float(x.mean() / x.std() * np.sqrt(12)) if x.std() > 0 else np.nan
        print(pd.DataFrame(rows).T.round(3).to_string())
        ok_a = passes(rows["Rule A: trend GTAA-5"], rows["Bench 1: hold GTAA-5"])
        ok_b = passes(rows["Rule B: trend all ETFs"], rows["Bench 1B: hold all ETFs"])
        d, t, p = monthly_diff_t(curves["Rule A: trend GTAA-5"], curves["Bench 1: hold GTAA-5"], s, e)
        print(f"Rule A vs Bench 1 monthly return difference: {d:+.2%}/month, t {t:.2f}, p {p:.2f} (information only)")
        word = "confirmed" if per == "holdout" else "PASS"
        print(f"Rule A {per}: {word if ok_a else 'not ' + word.lower() if per == 'holdout' else 'FAIL'}"
              f" | Rule B (secondary): {'pass' if ok_b else 'fail'}")
        out[per] = {"table": rows, "rule_a_pass": ok_a, "rule_b_pass": ok_b, "diff_month_t_p": [d, t, p]}
    return out


if __name__ == "__main__":
    main()
