"""Bullish candlestick patterns (all evaluated at the close of the last bar)."""
from __future__ import annotations

from .. import indicators as ind
from .registry import register


@register("bullish_engulfing", "candles")
def bullish_engulfing(df):
    """Prior bar bearish, today bullish, open <= prior close, close >= prior open,
    today's body > prior body."""
    o, c = df["open"], df["close"]
    po, pc = o.shift(1), c.shift(1)
    return (
        (pc < po) & (c > o) & (o <= pc) & (c >= po) & ((c - o) > (po - pc))
    )


@register("hammer", "candles")
def hammer(df):
    """Lower wick >= 2 x body (and > 0), upper wick <= body, after 3 consecutive
    down closes (close[t-1] < close[t-2] < close[t-3] < close[t-4])."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    top = o.where(o > c, c)
    bot = o.where(o < c, c)
    lower = bot - l
    upper = h - top
    d = c.diff()
    downs = (d.shift(1) < 0) & (d.shift(2) < 0) & (d.shift(3) < 0)
    return (lower >= 2 * body) & (lower > 0) & (upper <= body) & downs


@register("inside_bar_breakout", "candles")
def inside_bar_breakout(df):
    """Bar t-1 inside bar t-2 (high <=, low >=); today closes above bar t-2's high."""
    h, l, c = df["high"], df["low"], df["close"]
    inside = (h.shift(1) <= h.shift(2)) & (l.shift(1) >= l.shift(2))
    return inside & (c > h.shift(2))


@register("morning_star", "candles")
def morning_star(df):
    """t-2 bearish with body > 0.5 x ATR14(t-2); t-1 body < 0.3 x ATR14(t-1) with its
    body bottom below t-2's close; t bullish closing above the midpoint of t-2's body."""
    o, c = df["open"], df["close"]
    a = ind.atr(df, 14)
    o2, c2, a2 = o.shift(2), c.shift(2), a.shift(2)
    o1, c1, a1 = o.shift(1), c.shift(1), a.shift(1)
    first = (c2 < o2) & ((o2 - c2) > 0.5 * a2)
    star_bot = o1.where(o1 < c1, c1)
    star = ((c1 - o1).abs() < 0.3 * a1) & (star_bot < c2)
    third = (c > o) & (c > (o2 + c2) / 2.0)
    return first & star & third
