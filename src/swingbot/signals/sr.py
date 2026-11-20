"""Support / resistance signals."""
from __future__ import annotations

from .. import indicators as ind
from .registry import register


@register("swing_low_bounce", "sr")
def swing_low_bounce(df):
    """State: low within 1% (either side) of the last confirmed swing low, close > open."""
    sl = ind.swing_low(df["low"], 2)
    return ((df["low"] / sl - 1.0).abs() <= 0.01) & (df["close"] > df["open"])


@register("pivot_support_hold", "sr")
def pivot_support_hold(df):
    """Classic pivot from the prior bar: P = (H+L+C)/3, S1 = 2P - H.
    Fires when today's low <= S1 and close > S1."""
    h, l, c = df["high"].shift(1), df["low"].shift(1), df["close"].shift(1)
    p = (h + l + c) / 3.0
    s1 = 2.0 * p - h
    return (df["low"] <= s1) & (df["close"] > s1)
