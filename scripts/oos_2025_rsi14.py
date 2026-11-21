"""User-requested out-of-sample test: rsi14_oversold on single stocks, 2025-01-01 onward.
Fixed rule (no tuning): RSI(14) < 30 close, enter next open, exit close of t+10, ret_long_10 (net of 0.10%).
$20 paper account model: fractional shares, +$0.01 regulatory fee per sell (rounded-up TAF),
leverage L: position notional = L * equity / slots, interest 7%/yr on the borrowed part for 14 calendar days.
"""
import numpy as np
import pandas as pd

from swingbot import config
from swingbot.analysis.solo import solo_table
from swingbot.analysis.views import universe_views
from swingbot.backtest.portfolio import simulate, metrics
from swingbot.data import store

START = pd.Timestamp("2025-01-01")
CASH0, FEE_SELL, RATE, HOLD_DAYS = 20.0, 0.01, 0.07, 14

p = pd.read_parquet(config.DATA_DIR / "panel" / "stock_with_holdout.parquet")
last = p["date"].max()
p = p[p["date"] < last]  # drop the newest bar (may be a partial day)
v = universe_views({"stock": p})["stock"]          # single stocks only (ETFs excluded)
v = v[(v["split"] == "holdout") & (v["date"] >= START)].copy()
v["split"] = "holdout"
print("period", v["date"].min().date(), "->", v["date"].max().date(), "rows", len(v), "assets", v["asset"].nunique())

# 1) Same statistics as the gate, on 2025+
st = solo_table(v, horizon=10, split="holdout").set_index("signal").loc["rsi14_oversold"]
print(f"STATS rsi14_oversold 2025+: firings {st.n_all}, dates {st.n_dates}, edge {st.edge_vs_all_days*1e4:.1f} bps, "
      f"p {st.p:.3f}, mean excess {st.mean_all*1e4:.1f} bps, mean abs return {st.mean_ret*1e4:.1f} bps, "
      f"hit {st.hit_rate:.1%}, insufficient {st.insufficient}")


def dollars(trades, dates, slots, lev):
    """Re-price a simulate() trade list for a $20 account with fees, min $1 order and leverage."""
    booked = pd.Series(0.0, index=dates)
    eq = CASH0
    by_exit = {}
    skipped = 0
    for t in trades.sort_values("date").itertuples():
        # book exits up to and including the signal date
        for d in sorted(k for k in by_exit if k <= t.date):
            eq += sum(by_exit.pop(d))
        notional = lev * eq / slots
        if notional < 1.0 or eq <= 0:
            skipped += 1
            continue
        borrowed = notional - eq / slots if lev > 1 else 0.0
        pnl = notional * t.ret - FEE_SELL - borrowed * RATE * HOLD_DAYS / 365
        by_exit.setdefault(t.exit_date, []).append(pnl)
        booked[t.exit_date] += pnl
    curve = CASH0 + booked.cumsum()
    return curve, skipped


dates = pd.DatetimeIndex(sorted(v["date"].unique()))
rows = []
for rule in ("rsi14_oversold", "__all__"):
    for slots in (3, 10):
        for lev in (1, 2):
            finals, dds = [], []
            for seed in range(5):
                tr, _ = simulate(v, rule, "long", "holdout", horizon=10, max_positions=slots, seed=seed)
                curve, skipped = dollars(tr, dates, slots, lev)
                finals.append(curve.iloc[-1])
                dds.append(float((curve / curve.cummax() - 1).min()))
            rows.append({"rule": rule, "slots": slots, "leverage": lev, "trades": len(tr),
                         "end_$_median": np.median(finals), "end_$_min": min(finals), "end_$_max": max(finals),
                         "worst_drawdown_median": np.median(dds)})
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.round(2).to_string(index=False))

spy = store.load_bars("SPY", "stock", allow_holdout=True)["close"]
spy = spy[(spy.index >= v["date"].min()) & (spy.index <= v["date"].max())]
print(f"SPY buy-and-hold same period: $20 -> ${20 * spy.iloc[-1] / spy.iloc[0]:.2f}, "
      f"worst drawdown {float((spy / spy.cummax() - 1).min()):.1%}")

# Fee drag illustration
tr, eq = simulate(v, "rsi14_oversold", "long", "holdout", horizon=10, max_positions=3, seed=0)
print(f"avg trade return (net 0.10%) {tr['ret'].mean()*1e4:.1f} bps over {len(tr)} trades; "
      f"$0.01 sell fee on a ~$6.67 position = {0.01/6.67*1e4:.0f} bps")
