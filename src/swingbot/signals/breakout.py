"""Breakout signals. Channel levels exclude today (indicators.donchian_*)."""
from __future__ import annotations

from .. import indicators as ind
from .registry import register


@register("donchian20_breakout", "breakout")
def donchian20_breakout(df):
    """State: close > highest high of the previous 20 bars."""
    return df["close"] > ind.donchian_high(df["high"], 20)


@register("donchian20_breakdown", "breakout", "short")
def donchian20_breakdown(df):
    """State: close < lowest low of the previous 20 bars."""
    return df["close"] < ind.donchian_low(df["low"], 20)


@register("donchian55_breakout", "breakout")
def donchian55_breakout(df):
    return df["close"] > ind.donchian_high(df["high"], 55)


@register("squeeze_release_up", "breakout")
def squeeze_release_up(df):
    """BB(20,2) inside Keltner(20,1.5) on the previous bar, not today, close > BB mid."""
    mid, bu, bl = ind.bollinger(df["close"], 20, 2.0)
    _, ku, kl = ind.keltner(df, 20, 1.5)
    valid = bu.notna() & ku.notna()
    squeeze = (bu < ku) & (bl > kl)
    prev_sq = squeeze.shift(1, fill_value=False).astype(bool)
    prev_valid = valid.shift(1, fill_value=False).astype(bool)
    return prev_sq & prev_valid & valid & ~squeeze & (df["close"] > mid)


@register("volume_breakout", "breakout")
def volume_breakout(df):
    """State: close > prior 20-bar high and volume > 1.5 x SMA20(volume) (incl. today)."""
    v = df["volume"]
    return (df["close"] > ind.donchian_high(df["high"], 20)) & (v > 1.5 * ind.sma(v, 20))


@register("high_52w", "breakout")
def high_52w(df):
    """State: close > highest high of the previous 252 bars."""
    return df["close"] > ind.donchian_high(df["high"], 252)
