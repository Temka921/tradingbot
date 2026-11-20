"""Shared causal helpers for signal modules.

Swing helpers expose the *confirmation* events of the fractal pivots used by
``indicators.swing_low/swing_high`` (same strict rule: pivot at t-k is lower /
higher than the k bars on each side, confirmed at bar t). Everything returned
here is indexed by the confirmation bar, so it is known at the close of t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def confirmed_pivots(x: pd.Series, k: int = 2, low: bool = True) -> pd.Series:
    """Bool Series: True at bar t when the bar t-k is confirmed as a swing pivot."""
    v = x.to_numpy(float)
    out = np.zeros(len(v), dtype=bool)
    for t in range(2 * k, len(v)):
        w = v[t - 2 * k : t + 1]
        if np.isnan(w).any():
            continue
        p = w[k]
        others = np.delete(w, k)
        out[t] = (p < others.min()) if low else (p > others.max())
    return pd.Series(out, index=x.index)


def last_pivot(x: pd.Series, k: int = 2, low: bool = True):
    """For each bar t: (value, pivot bar position) of the last pivot confirmed by t.

    Both forward-filled; NaN before the first confirmation. Also returns the
    confirmation mask.
    """
    conf = confirmed_pivots(x, k, low)
    pos = pd.Series(np.arange(len(x), dtype=float) - k, index=x.index)
    val = x.shift(k).where(conf).ffill()
    ppos = pos.where(conf).ffill()
    return val, ppos, conf


def event(cond: pd.Series, valid: pd.Series | None = None) -> pd.Series:
    """True on the bar `cond` becomes true (false/defined on the previous bar).

    `valid` marks bars where `cond` is defined; an undefined previous bar never
    produces an event (avoids spurious 'crosses' at the end of warm-up).
    """
    c = cond.fillna(False).astype(bool)
    if valid is None:
        v = pd.Series(True, index=c.index)
    else:
        v = valid.fillna(False).astype(bool)
    prev_c = c.shift(1, fill_value=False).astype(bool)
    prev_v = v.shift(1, fill_value=False).astype(bool)
    return c & v & prev_v & ~prev_c


def cross_above(a: pd.Series, b) -> pd.Series:
    """a crosses above b at t: a[t] > b[t] and a[t-1] <= b[t-1] (all defined)."""
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return ((a > b) & (a.shift(1) <= b_prev)).fillna(False).astype(bool)


def cross_below(a: pd.Series, b) -> pd.Series:
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return ((a < b) & (a.shift(1) >= b_prev)).fillna(False).astype(bool)


def flip(direction: pd.Series, to: float) -> pd.Series:
    """Direction series (+1/-1/NaN) flips to `to` at t: prev == -to and cur == to."""
    return ((direction.shift(1) == -to) & (direction == to)).astype(bool)
