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


def _avwap_same_anchor(df: pd.DataFrame, k: int = 2) -> tuple[pd.Series, pd.Series]:
    """(AVWAP at t, AVWAP at t-1) both computed with the anchor in force at t.

    The anchor at t is the pivot bar p of the latest swing low confirmed at or
    before t; AVWAP[t] = sum(tp*v)/sum(v) over bars p..t, tp = (h+l+c)/3, and the
    same-anchor previous value covers bars p..t-1 (non-empty since p <= t-k).
    Everything uses bars <= t, so it is causal. NaN before the first confirmation.
    """
    _, ppos, _ = last_pivot(df["low"], k, low=True)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).fillna(0.0).to_numpy()
    v = df["volume"].where(tp.notna()).fillna(0.0).to_numpy()
    cpv = np.concatenate([[0.0], np.cumsum(pv)])
    cv = np.concatenate([[0.0], np.cumsum(v)])
    pos = ppos.to_numpy()
    cur = np.full(len(df), np.nan)
    prev = np.full(len(df), np.nan)
    ok = ~np.isnan(pos)
    t = np.arange(len(df))[ok]
    p = pos[ok].astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        for out, end in ((cur, t + 1), (prev, t)):
            num = cpv[end] - cpv[p]
            den = cv[end] - cv[p]
            out[ok] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return pd.Series(cur, index=df.index), pd.Series(prev, index=df.index)


def anchored_vwap_swinglow(df: pd.DataFrame, k: int = 2) -> pd.Series:
    """VWAP anchored at the pivot bar of the last confirmed swing low (resets on
    each new confirmation). NaN before the first confirmation."""
    return _avwap_same_anchor(df, k)[0]


@register("close_above_avwap_swinglow", "volume")
def close_above_avwap_swinglow(df):
    """Event: close crosses above the swing-low anchored VWAP *within one anchor*:
    close[t] > AVWAP_p[t] and close[t-1] <= AVWAP_p[t-1], with p the anchor in
    force at t. An anchor reset alone never creates a cross; on a confirmation
    bar it fires only if price genuinely crossed the new-anchor VWAP."""
    c = df["close"]
    cur, prev = _avwap_same_anchor(df)
    return ((c > cur) & (c.shift(1) <= prev)).fillna(False).astype(bool)


@register("ad_rising", "volume")
def ad_rising(df):
    """Event: A/D line crosses above its SMA20 while close > SMA50."""
    ad = ind.ad_line(df)
    c = df["close"]
    return cross_above(ad, ind.sma(ad, 20)) & (c > ind.sma(c, 50))
