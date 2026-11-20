"""Trend-following signals (crosses and direction flips are events)."""
from __future__ import annotations

from .. import indicators as ind
from ._swings import cross_above, cross_below, flip
from .registry import register


@register("ma20_50_cross_up", "trend")
def ma20_50_cross_up(df):
    c = df["close"]
    return cross_above(ind.sma(c, 20), ind.sma(c, 50))


@register("ma20_50_cross_down", "trend", "short")
def ma20_50_cross_down(df):
    c = df["close"]
    return cross_below(ind.sma(c, 20), ind.sma(c, 50))


@register("ma50_200_cross_up", "trend")
def ma50_200_cross_up(df):
    c = df["close"]
    return cross_above(ind.sma(c, 50), ind.sma(c, 200))


@register("ma50_200_cross_down", "trend", "short")
def ma50_200_cross_down(df):
    c = df["close"]
    return cross_below(ind.sma(c, 50), ind.sma(c, 200))


@register("close_above_ma200_cross", "trend")
def close_above_ma200_cross(df):
    c = df["close"]
    return cross_above(c, ind.sma(c, 200))


@register("macd_cross_up", "trend")
def macd_cross_up(df):
    line, sig = ind.macd(df["close"])
    return cross_above(line, sig)


@register("macd_cross_down", "trend", "short")
def macd_cross_down(df):
    line, sig = ind.macd(df["close"])
    return cross_below(line, sig)


@register("adx_trend_up", "trend")
def adx_trend_up(df):
    """+DI crosses above -DI on a bar where ADX(14) > 25."""
    a, pdi, mdi = ind.adx(df)
    return cross_above(pdi, mdi) & (a > 25)


@register("supertrend_flip_up", "trend")
def supertrend_flip_up(df):
    return flip(ind.supertrend_dir(df), 1.0)


@register("supertrend_flip_down", "trend", "short")
def supertrend_flip_down(df):
    return flip(ind.supertrend_dir(df), -1.0)


@register("psar_flip_up", "trend")
def psar_flip_up(df):
    return flip(ind.psar_dir(df), 1.0)
