"""C1 momentum rotation, C2 weakness hedge, C3 Fear & Greed gated C1 (weekly, Sunday close)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import CASH, own_series_change

GATES = {"contrarian": lambda v: v <= 25, "confirming": lambda v: v >= 55}


def weekly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Signal dates: the Sundays present in ``index``."""
    index = pd.DatetimeIndex(index)
    return index[index.day_of_week == 6]


def _inputs(closes: pd.DataFrame, eligible: pd.DataFrame, lookback: int):
    dates = weekly_dates(closes.index)
    ret = own_series_change(closes, 0, lookback).loc[dates]
    elig = eligible.reindex(index=dates, columns=closes.columns).fillna(False).astype(bool)
    return dates, ret, elig & ret.notna()


def _empty(dates: pd.DatetimeIndex, closes: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=dates, columns=[*closes.columns, CASH])


def btc_regime(closes: pd.DataFrame, btc: str = "BTCUSDT", sma: int = 200) -> pd.Series:
    """BTC close above its ``sma``-day simple average (own bars), on ``closes.index``; else False."""
    b = closes[btc].dropna()
    on = b > b.rolling(sma, min_periods=sma).mean()
    return on.reindex(closes.index, fill_value=False).astype(bool)


def c1_weights(closes: pd.DataFrame, eligible: pd.DataFrame, btc: str = "BTCUSDT", top: int = 5,
               lookback: int = 30, sma: int = 200) -> pd.DataFrame:
    """Equal weight in the top ``top`` eligible coins by ``lookback``-day return while BTC is
    above its ``sma``-day SMA; otherwise (or with no candidate) 100% CASH."""
    dates, ret, cand = _inputs(closes, eligible, lookback)
    regime = btc_regime(closes, btc, sma).loc[dates]
    w = _empty(dates, closes)
    for d in dates:
        r = ret.loc[d][cand.loc[d]]
        if not regime.loc[d] or r.empty:
            w.loc[d, CASH] = 1.0
            continue
        picks = r.sort_values(ascending=False, kind="mergesort").index[:top]
        w.loc[d, picks] = 1.0 / len(picks)
    return w


def c2_weights(closes: pd.DataFrame, eligible: pd.DataFrame, btc: str = "BTCUSDT", bottom: int = 5,
               lookback: int = 30) -> pd.DataFrame:
    """BTC +0.5; the bottom ``bottom`` eligible non-BTC coins by ``lookback``-day return short
    -0.5 in total (−0.1 each for 5, −0.5/n for fewer); CASH = 1 − 0.5 − shorts (1.0 with shorts,
    0.5 if there is no short candidate)."""
    dates, ret, cand = _inputs(closes, eligible, lookback)
    w = _empty(dates, closes)
    w[btc] = 0.5
    for d in dates:
        r = ret.loc[d][cand.loc[d]].drop(btc, errors="ignore")
        picks = r.sort_values(ascending=True, kind="mergesort").index[:bottom]
        if len(picks):
            w.loc[d, picks] = -0.5 / len(picks)
    w[CASH] = 1.0 - w.drop(columns=CASH).sum(axis=1)
    return w


def c3_weights(closes: pd.DataFrame, eligible: pd.DataFrame, fng: pd.Series, mode: str,
               btc: str = "BTCUSDT", top: int = 5, lookback: int = 30,
               sma: int = 200) -> pd.DataFrame:
    """C1 where the Fear & Greed gate is open, else 100% CASH.

    The gate at signal date t reads the index value dated t − 1 calendar day ("contrarian":
    ≤ 25, "confirming": ≥ 55). A missing value closes the gate.
    """
    if mode not in GATES:
        raise ValueError(f"mode must be one of {sorted(GATES)}, got {mode!r}")
    w = c1_weights(closes, eligible, btc, top, lookback, sma)
    f = fng.copy()
    f.index = pd.DatetimeIndex(f.index).normalize()
    f = f[~f.index.duplicated(keep="last")]
    lagged = f.reindex(w.index - pd.Timedelta(days=1)).to_numpy(float)
    gate = np.where(np.isfinite(lagged), GATES[mode](np.nan_to_num(lagged, nan=50.0)), False)
    closed = w.index[~gate.astype(bool)]
    w.loc[closed] = 0.0
    w.loc[closed, CASH] = 1.0
    return w
