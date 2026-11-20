"""Pullback-in-uptrend signals."""
from __future__ import annotations

from .. import indicators as ind
from ._swings import last_pivot
from .registry import register


@register("pullback_ma20_uptrend", "pullback")
def pullback_ma20_uptrend(df):
    """State: low <= SMA20 < close with SMA20 > SMA50 > SMA200."""
    c = df["close"]
    m20, m50, m200 = ind.sma(c, 20), ind.sma(c, 50), ind.sma(c, 200)
    return (df["low"] <= m20) & (c > m20) & (m20 > m50) & (m50 > m200)


@register("rsi14_pullback_uptrend", "pullback")
def rsi14_pullback_uptrend(df):
    """Event: RSI14 drops from above 50 into [40, 50] while close > SMA200."""
    c = df["close"]
    r = ind.rsi(c, 14)
    return (r.shift(1) > 50) & (r >= 40) & (r <= 50) & (c > ind.sma(c, 200))


@register("rsi2_oversold", "pullback")
def rsi2_oversold(df):
    """State: RSI(2) < 10 while close > SMA200."""
    c = df["close"]
    return (ind.rsi(c, 2) < 10) & (c > ind.sma(c, 200))


@register("fib_retrace_50_618", "pullback")
def fib_retrace_50_618(df):
    """State: close inside the 50%-61.8% retracement of the last confirmed swing
    low -> swing high leg (the high's pivot bar after the low's), close > SMA200.

    Retracement is measured down from the high: zone = [H - 0.618R, H - 0.5R].
    """
    c = df["close"]
    lo, lo_pos, _ = last_pivot(df["low"], 2, low=True)
    hi, hi_pos, _ = last_pivot(df["high"], 2, low=False)
    rng = hi - lo
    leg = (hi_pos > lo_pos) & (rng > 0)
    zone = (c >= hi - 0.618 * rng) & (c <= hi - 0.5 * rng)
    return leg & zone & (c > ind.sma(c, 200))
