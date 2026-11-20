"""Mean-reversion signals."""
from __future__ import annotations

from .. import indicators as ind
from ._swings import cross_above, cross_below
from .registry import register


@register("bb_lower_close", "meanrev")
def bb_lower_close(df):
    """State: close below the lower Bollinger band (20, 2)."""
    _, _, lower = ind.bollinger(df["close"], 20, 2.0)
    return df["close"] < lower


@register("bb_upper_close", "meanrev", "short")
def bb_upper_close(df):
    _, upper, _ = ind.bollinger(df["close"], 20, 2.0)
    return df["close"] > upper


@register("rsi14_oversold", "meanrev")
def rsi14_oversold(df):
    """Event: RSI14 crosses below 30."""
    return cross_below(ind.rsi(df["close"], 14), 30.0)


@register("rsi14_overbought", "meanrev", "short")
def rsi14_overbought(df):
    """Event: RSI14 crosses above 70."""
    return cross_above(ind.rsi(df["close"], 14), 70.0)


@register("zscore_below_2", "meanrev")
def zscore_below_2(df):
    """State: 20-bar z-score of close < -2."""
    return ind.zscore(df["close"], 20) < -2.0


@register("stoch_oversold_cross", "meanrev")
def stoch_oversold_cross(df):
    """Event: %K(14) crosses above %D(3) with both below 20 on the cross bar."""
    k, d = ind.stochastic(df, 14, 3)
    return cross_above(k, d) & (k < 20) & (d < 20)


@register("gap_down_fill_setup", "meanrev")
def gap_down_fill_setup(df):
    """Open below prior low by more than 0.5 x ATR14 (ATR as of prior close), close > open."""
    a = ind.atr(df, 14).shift(1)
    gap = df["low"].shift(1) - df["open"]
    return (gap > 0.5 * a) & (df["close"] > df["open"])
