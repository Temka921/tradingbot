"""Forward outcomes (the labels signals are scored on).

Entry is always the open of bar t+1 (never the close of t). Outcomes look
forward by design; the no-lookahead rule applies to signals/regimes only.
"""
import numpy as np
import pandas as pd

from swingbot import config
from swingbot.indicators import atr

ATR_EXIT_BARS = 20
ATR_STOP_MULT = 2.0
ATR_TARGET_MULT = 3.0


def _fwd_window(s: pd.Series, h: int, fn: str) -> pd.Series:
    """fn over s[t+1..t+h]; NaN when the window is incomplete."""
    rev = s.iloc[::-1].rolling(h)
    agg = rev.max() if fn == "max" else rev.min()
    return agg.iloc[::-1].shift(-1)  # window s[t..t+h-1] -> shift to s[t+1..t+h]


def _bench_gross(df: pd.DataFrame, bench: pd.DataFrame, h: int, delisted: bool = False) -> pd.Series:
    """Benchmark close(first date >= asset date[t+h]) / open(first date >= asset date[t+1]) - 1.

    ``delisted``: the exit bar is capped at the asset's last bar L, and rows with
    t+1 <= L < t+h are filled too.
    """
    n = len(df)
    out = np.full(n, np.nan)
    if n == 0 or len(bench) == 0:
        return pd.Series(out, index=df.index)
    bidx = bench.index.to_numpy().astype("datetime64[ns]")
    adates = df.index.to_numpy().astype("datetime64[ns]")
    bopen = bench["open"].to_numpy(dtype=float)
    bclose = bench["close"].to_numpy(dtype=float)
    if delisted:
        t = np.arange(n - 1)
    else:
        t = np.arange(n - h) if n > h else np.array([], dtype=int)
    if len(t):
        ex = np.minimum(t + h, n - 1)
        i_in = np.searchsorted(bidx, adates[t + 1], side="left")
        i_out = np.searchsorted(bidx, adates[ex], side="left")
        ok = (i_in < len(bidx)) & (i_out < len(bidx))
        vals = np.full(len(t), np.nan)
        vals[ok] = bclose[i_out[ok]] / bopen[i_in[ok]] - 1
        out[t] = vals
    return pd.Series(out, index=df.index)


def _atr_exits(df: pd.DataFrame, cost: float, delisted: bool = False) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    o = df["open"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    a = atr(df, 14).to_numpy(float)
    long_r = np.full(n, np.nan)
    short_r = np.full(n, np.nan)
    last = n - 1
    for t in range(n - 1 if delisted else n - ATR_EXIT_BARS):
        if np.isnan(a[t]):
            continue
        entry = o[t + 1]
        l_stop, l_tgt = entry - ATR_STOP_MULT * a[t], entry + ATR_TARGET_MULT * a[t]
        s_stop, s_tgt = entry + ATR_STOP_MULT * a[t], entry - ATR_TARGET_MULT * a[t]
        l_exit = s_exit = None
        end_i = min(t + ATR_EXIT_BARS, last)  # == t + 20 unless a delisted window runs past L
        for k in range(t + 1, end_i + 1):
            if l_exit is None:
                if o[k] <= l_stop:
                    l_exit = o[k]
                elif lo[k] <= l_stop:
                    l_exit = l_stop
                elif o[k] >= l_tgt:
                    l_exit = o[k]
                elif hi[k] >= l_tgt:
                    l_exit = l_tgt
            if s_exit is None:
                if o[k] >= s_stop:
                    s_exit = o[k]
                elif hi[k] >= s_stop:
                    s_exit = s_stop
                elif o[k] <= s_tgt:
                    s_exit = o[k]
                elif lo[k] <= s_tgt:
                    s_exit = s_tgt
            if l_exit is not None and s_exit is not None:
                break
        end = c[end_i]
        l_exit = end if l_exit is None else l_exit
        s_exit = end if s_exit is None else s_exit
        long_r[t] = l_exit / entry - 1 - cost
        short_r[t] = -(s_exit / entry - 1) - cost
    return long_r, short_r


def compute_outcomes(df: pd.DataFrame, bench: pd.DataFrame, market: str,
                     delisted: bool = False) -> pd.DataFrame:
    """Forward outcomes. With ``delisted=True`` a window running past the last bar L
    exits at close[L] (so a dying asset's crash stays in the statistics instead of
    becoming NaN); mfe/mae stay NaN there."""
    cost = config.COSTS[market]
    entry = df["open"].shift(-1)
    out = pd.DataFrame(index=df.index)
    n = len(df)
    t = np.arange(n)
    for h in config.HORIZONS:
        gross = df["close"].shift(-h) / entry - 1
        if delisted and n > 1:
            exit_idx = np.minimum(t + h, n - 1)
            close_exit = df["close"].to_numpy(float)[exit_idx]
            filled = close_exit / entry.to_numpy(float) - 1  # NaN at t = L (no entry bar)
            gross = pd.Series(filled, index=df.index)
        bg = _bench_gross(df, bench, h, delisted)
        out[f"gross_{h}"] = gross
        out[f"ret_long_{h}"] = gross - cost
        out[f"ret_short_{h}"] = -gross - cost
        out[f"bench_gross_{h}"] = bg
        out[f"excess_long_{h}"] = out[f"ret_long_{h}"] - bg
        out[f"excess_short_{h}"] = out[f"ret_short_{h}"] + bg
        out[f"mfe_{h}"] = _fwd_window(df["high"], h, "max") / entry - 1
        out[f"mae_{h}"] = _fwd_window(df["low"], h, "min") / entry - 1
    long_r, short_r = _atr_exits(df, cost, delisted)
    out["atr_exit_long"] = long_r
    out["atr_exit_short"] = short_r
    return out
