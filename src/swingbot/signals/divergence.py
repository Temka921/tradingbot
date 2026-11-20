"""Bullish divergence between confirmed swing lows and an oscillator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ._swings import confirmed_pivots
from .registry import register

K = 2  # fractal half-width, same as indicators.swing_low


def _bull_divergence(df: pd.DataFrame, osc: pd.Series) -> pd.Series:
    """Fires on the confirmation bar t of a new swing low (pivot at t-K) whose low
    is below the previous confirmed swing low while the oscillator at the new
    pivot bar is above its value at the previous pivot bar. Uses only pivot-bar
    values, which are in the past at t."""
    low = df["low"].to_numpy(float)
    o = osc.to_numpy(float)
    conf = confirmed_pivots(df["low"], K, low=True).to_numpy()
    out = np.zeros(len(df), dtype=bool)
    prev = None
    for t in np.flatnonzero(conf):
        p = t - K
        if prev is not None:
            ok = (
                low[p] < low[prev]
                and not np.isnan(o[p])
                and not np.isnan(o[prev])
                and o[p] > o[prev]
            )
            out[t] = ok
        prev = p
    return pd.Series(out, index=df.index)


@register("rsi_bull_divergence", "divergence")
def rsi_bull_divergence(df):
    return _bull_divergence(df, ind.rsi(df["close"], 14))


@register("macd_bull_divergence", "divergence")
def macd_bull_divergence(df):
    line, _ = ind.macd(df["close"])
    return _bull_divergence(df, line)
