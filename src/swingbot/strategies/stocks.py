"""S1 dip in a healthy uptrend (fixed horizon) and S2 12-1 momentum with a per-stock crash switch."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingbot import indicators as ind
from swingbot.backtest.trend import month_end_closes

from . import CASH, own_series_change


def index_up(index_close: pd.Series, dates: pd.DatetimeIndex, n: int = 200) -> pd.Series:
    """Index close above its ``n``-day SMA (index's own bars), as of each date in ``dates``
    (last index bar on or before the date). False before the SMA exists."""
    s = index_close.dropna().sort_index()
    up = (s > ind.sma(s, n)).astype(float).where(ind.sma(s, n).notna())
    return up.reindex(pd.DatetimeIndex(dates), method="ffill").eq(1.0)


def home_regime_frame(closes: pd.DataFrame, index_close: dict[str, pd.Series],
                      home: dict[str, str]) -> pd.DataFrame:
    """Bool frame (closes.index × stocks): each stock's home index above its 200-day SMA."""
    cache = {k: index_up(v, closes.index) for k, v in index_close.items()}
    return pd.DataFrame({c: cache[home[c]].to_numpy() for c in closes.columns},
                        index=closes.index, columns=closes.columns).astype(bool)


def s1_frame(bars: dict[str, pd.DataFrame], index_close: dict[str, pd.Series],
             home: dict[str, str], cost: float, horizon: int = 10) -> pd.DataFrame:
    """Long frame for ``portfolio.simulate``: date, asset, s1, ret_long_10, split (None).

    s1 at close t: RSI(14) < 30, close > SMA200, home index close > its SMA200 (as of t).
    ret_long_{horizon}: close[t+horizon] / open[t+1] − 1 − cost (NaN near the end).
    Not clipped here; the runner clips pooled returns and fills split.
    """
    parts = []
    for sym, df in bars.items():
        c = df["close"]
        sig = (ind.rsi(c, 14) < 30) & (c > ind.sma(c, 200)) & index_up(index_close[home[sym]], df.index)
        parts.append(pd.DataFrame({
            "date": df.index, "asset": sym, "s1": sig.to_numpy(bool),
            f"ret_long_{horizon}": (c.shift(-horizon) / df["open"].shift(-1) - 1 - cost).to_numpy(),
            "split": None,
        }))
    cols = ["date", "asset", "s1", f"ret_long_{horizon}", "split"]
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)[cols]


def _momentum(closes: pd.DataFrame) -> pd.DataFrame:
    """12-1 momentum close[t-21] / close[t-252] − 1 on each stock's own bars, at month ends."""
    ends = month_end_closes(closes).index
    return own_series_change(closes, 21, 252).loc[ends]


def s2_weights(closes: pd.DataFrame, home_regime: pd.DataFrame, min_valid: int = 50,
               top_frac: float = 0.10, min_top: int = 5) -> pd.DataFrame:
    """Monthly (union-calendar month-end) target weights.

    With n stocks holding a valid 12-1 momentum value (n < ``min_valid`` → 100% CASH), the slot
    count is k = max(min_top, floor(top_frac × n)). The k best-ranked stocks among those that are
    valid **and** whose own home index is above its 200-day SMA get 1/k each; unfilled slots
    go to CASH.
    """
    mom = _momentum(closes)
    reg = home_regime.reindex(index=mom.index, columns=closes.columns).fillna(False).astype(bool)
    w = pd.DataFrame(0.0, index=mom.index, columns=[*closes.columns, CASH])
    for d in mom.index:
        m = mom.loc[d].dropna()
        if len(m) < min_valid:
            w.loc[d, CASH] = 1.0
            continue
        k = max(min_top, int(np.floor(top_frac * len(m) + 1e-9)))
        picks = m[reg.loc[d, m.index]].sort_values(ascending=False, kind="mergesort").index[:k]
        w.loc[d, picks] = 1.0 / k
        w.loc[d, CASH] = 1.0 - len(picks) / k
    return w


def equal_weight_hold(closes: pd.DataFrame, min_valid: int = 50) -> pd.DataFrame:
    """S2 benchmark: at each S2 rebalance date, equal weight in every stock with a valid 12-1
    momentum value (the S2 universe on that date); 100% CASH while fewer than ``min_valid``."""
    mom = _momentum(closes)
    valid = mom.notna()
    n = valid.sum(axis=1)
    w = valid.astype(float).div(n.where(n >= min_valid), axis=0).fillna(0.0)
    w[CASH] = (n < min_valid).astype(float)
    return w
