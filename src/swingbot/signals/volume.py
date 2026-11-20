"""Volume-based signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ._swings import cross_above, last_pivot
from .registry import register


@register("obv_leads_price", "volume")
def obv_leads_price(df):
    """State: OBV makes a new 20-bar high (strictly above the previous 19 bars)
    while close is below its own 20-bar high (max close incl. today)."""
    o = ind.obv(df)
    c = df["close"]
    obv_high = o > o.shift(1).rolling(19).max()
    return obv_high & (c < c.rolling(20).max())


def anchored_vwap_swinglow(df: pd.DataFrame, k: int = 2) -> pd.Series:
    """VWAP anchored at the pivot bar of the last confirmed swing low.

    At bar t the anchor is the pivot bar p of the latest swing low confirmed at or
    before t; VWAP = sum(tp*v)/sum(v) over bars p..t, tp = (h+l+c)/3. Resets when
    a new swing low is confirmed. NaN before the first confirmation.
    """
    _, ppos, _ = last_pivot(df["low"], k, low=True)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).fillna(0.0).to_numpy()
    v = df["volume"].where(tp.notna()).fillna(0.0).to_numpy()
    cpv = np.concatenate([[0.0], np.cumsum(pv)])
    cv = np.concatenate([[0.0], np.cumsum(v)])
    pos = ppos.to_numpy()
    out = np.full(len(df), np.nan)
    ok = ~np.isnan(pos)
    t = np.arange(len(df))[ok]
    p = pos[ok].astype(int)
    num = cpv[t + 1] - cpv[p]
    den = cv[t + 1] - cv[p]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return pd.Series(out, index=df.index)


@register("close_above_avwap_swinglow", "volume")
def close_above_avwap_swinglow(df):
    """Event: close crosses above the swing-low anchored VWAP (vs previous bar's VWAP)."""
    return cross_above(df["close"], anchored_vwap_swinglow(df))


@register("ad_rising", "volume")
def ad_rising(df):
    """Event: A/D line crosses above its SMA20 while close > SMA50."""
    ad = ind.ad_line(df)
    c = df["close"]
    return cross_above(ad, ind.sma(ad, 20)) & (c > ind.sma(c, 50))
