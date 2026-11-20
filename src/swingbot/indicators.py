"""Hand-written, causal technical indicators (no TA-Lib).

Every function is pure: output is aligned to the input index and the value at
bar t depends only on bars <= t. Warm-up values are NaN (direction indicators
return 0 until initialised). NaN inputs never raise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = _wilder(d.clip(lower=0.0), n)
    loss = _wilder((-d).clip(lower=0.0), n)
    out = 100.0 - 100.0 / (1.0 + gain / loss)
    out = out.where(~((loss == 0) & (gain > 0)), 100.0)
    out = out.where(~((loss == 0) & (gain == 0)), 50.0)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)  # first bar has no prior close: high - low
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _wilder(true_range(df), n)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def keltner(df: pd.DataFrame, n: int = 20, k: float = 1.5):
    mid = ema(df["close"], n)
    width = k * atr(df, n)
    return mid, mid + width, mid - width


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig


def adx(df: pd.DataFrame, n: int = 14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    nan = up.isna() | down.isna()
    plus_dm = plus_dm.mask(nan)
    minus_dm = minus_dm.mask(nan)
    a = atr(df, n).replace(0.0, np.nan)
    plus_di = 100.0 * _wilder(plus_dm, n) / a
    minus_di = 100.0 * _wilder(minus_dm, n) / a
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return _wilder(dx, n), plus_di, minus_di


def stochastic(df: pd.DataFrame, n: int = 14, d: int = 3):
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    k = 100.0 * (df["close"] - ll) / (hh - ll).replace(0.0, np.nan)
    return k, k.rolling(d).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def ad_line(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    return (clv.fillna(0.0) * df["volume"]).cumsum()


def zscore(close: pd.Series, n: int = 20) -> pd.Series:
    sd = close.rolling(n).std(ddof=0).replace(0.0, np.nan)
    return (close - sma(close, n)) / sd


def donchian_high(high: pd.Series, n: int) -> pd.Series:
    """Max of the previous n bars (today excluded)."""
    return high.rolling(n).max().shift(1)


def donchian_low(low: pd.Series, n: int) -> pd.Series:
    """Min of the previous n bars (today excluded)."""
    return low.rolling(n).min().shift(1)


def supertrend_dir(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    """+1 uptrend / -1 downtrend; 0 until ATR is available. Bars with NaN carry state."""
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    a = atr(df, n).to_numpy()
    size = len(df)
    out = np.zeros(size, dtype=int)
    fu = fl = np.nan
    direction = 0
    for i in range(size):
        if np.isnan(a[i]) or np.isnan(c[i]) or np.isnan(h[i]) or np.isnan(l[i]):
            out[i] = direction
            continue
        mid = (h[i] + l[i]) / 2.0
        bu, bl = mid + mult * a[i], mid - mult * a[i]
        if direction == 0:
            fu, fl = bu, bl
            direction = 1 if c[i] >= mid else -1
        else:
            pc = c[i - 1]
            new_fu = bu if (bu < fu or pc > fu or np.isnan(pc)) else fu
            new_fl = bl if (bl > fl or pc < fl or np.isnan(pc)) else fl
            fu, fl = new_fu, new_fl
            if direction == -1 and c[i] > fu:
                direction = 1
            elif direction == 1 and c[i] < fl:
                direction = -1
        out[i] = direction
    return pd.Series(out, index=df.index)


def psar_dir(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """Parabolic SAR trend side: +1 (SAR below price) / -1; 0 until initialised."""
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    size = len(df)
    out = np.zeros(size, dtype=int)
    start = None
    for i in range(1, size):
        if not (np.isnan(h[i]) or np.isnan(l[i]) or np.isnan(h[i - 1]) or np.isnan(l[i - 1])):
            start = i
            break
    if start is None:
        return pd.Series(out, index=df.index)
    up = h[start] + l[start] >= h[start - 1] + l[start - 1]
    sar = l[start - 1] if up else h[start - 1]
    ep = h[start - 1] if up else l[start - 1]
    af = step
    out[start] = 1 if up else -1
    # process bar start properly: treat it as the first stepped bar
    prev_h, prev_l = h[start - 1], l[start - 1]
    prev2_h, prev2_l = prev_h, prev_l
    for i in range(start, size):
        if np.isnan(h[i]) or np.isnan(l[i]):
            out[i] = out[i - 1] if i > start else (1 if up else -1)
            continue
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, prev_l, prev2_l)
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            sar = max(sar, prev_h, prev2_h)
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, max_step)
        out[i] = 1 if up else -1
        prev2_h, prev2_l = prev_h, prev_l
        prev_h, prev_l = h[i], l[i]
    return pd.Series(out, index=df.index)


def _swing(x: pd.Series, k: int, low: bool) -> pd.Series:
    v = x.to_numpy(float)
    size = len(v)
    out = np.full(size, np.nan)
    for t in range(2 * k, size):
        w = v[t - 2 * k : t + 1]
        if np.isnan(w).any():
            continue
        p = w[k]
        others = np.delete(w, k)
        if (p < others.min()) if low else (p > others.max()):
            out[t] = p
    return pd.Series(out, index=x.index).ffill()


def swing_low(low: pd.Series, k: int = 2) -> pd.Series:
    """Last confirmed fractal swing low (pivot at t-k, confirmed at t, strict)."""
    return _swing(low, k, True)


def swing_high(high: pd.Series, k: int = 2) -> pd.Series:
    return _swing(high, k, False)
