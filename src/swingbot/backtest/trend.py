"""Monthly trend-following (Faber 10-month SMA) portfolio backtest.

Protocol: docs/superpowers/specs/2026-11-21-trend-etf-protocol.md (pre-registered).
Signal at a month-end close uses only month-end closes up to that date; trades happen at the
next trading day's open; costs 0.05% per unit of weight traded (0.10% per round trip).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingbot import config

SMA_MONTHS = 10
COST_PER_SIDE = config.COSTS["stock"] / 2


def month_end_closes(closes: pd.DataFrame) -> pd.DataFrame:
    """Close on the last trading day of each month (index = that actual trading date)."""
    last_days = closes.index.to_series().groupby(closes.index.to_period("M")).max()
    return closes.loc[last_days.to_numpy()]


def trend_signals(closes: pd.DataFrame, months: int = SMA_MONTHS) -> pd.DataFrame:
    """True where the month-end close is above its `months`-month SMA; NaN before enough history."""
    me = month_end_closes(closes)
    sma = me.rolling(months, min_periods=months).mean()
    sig = (me > sma).astype(float)
    return sig.where(sma.notna() & me.notna())


def target_weights(signals: pd.DataFrame, cash: str = "SHY") -> pd.DataFrame:
    """Equal slots over assets with a defined signal; an 'off' slot goes to `cash`.

    Returns weights indexed by signal date, columns = signal assets + cash. Rows where no asset
    has a defined signal are dropped.
    """
    defined = signals.notna()
    n = defined.sum(axis=1)
    w = signals.fillna(0.0).div(n.replace(0, np.nan), axis=0)
    w[cash] = (defined & (signals == 0)).sum(axis=1) / n.replace(0, np.nan)
    return w[n > 0].fillna(0.0)


def run_portfolio(opens: pd.DataFrame, closes: pd.DataFrame, weights: pd.DataFrame,
                  cost_per_side: float = COST_PER_SIDE) -> pd.Series:
    """Daily equity (start 1.0). Weights dated at a signal close are traded at the next day's open.

    Between trades holdings drift with prices. Missing prices are forward-filled (no return).
    """
    cols = list(weights.columns)
    o = opens.reindex(columns=cols).ffill().to_numpy(float)
    c = closes.reindex(columns=cols).ffill().to_numpy(float)
    days = closes.index
    trade_at = {}
    for d, row in weights.iterrows():
        i = days.searchsorted(d, side="right")  # first trading day after the signal close
        if i < len(days):
            trade_at[i] = row.to_numpy(float)
    first = min(trade_at) if trade_at else len(days)
    eq = np.full(len(days), np.nan)
    hold = np.zeros(len(cols))  # value held per asset at the previous close
    value = 1.0
    for i in range(first, len(days)):
        if i in trade_at:
            if i > first:  # mark holdings from previous close to today's open
                ratio = np.where(np.isfinite(o[i]) & np.isfinite(c[i - 1]) & (c[i - 1] > 0), o[i] / c[i - 1], 1.0)
                hold = hold * ratio
                value = hold.sum()
            target = trade_at[i] * value
            turnover = np.abs(target - hold).sum() if i > first else target.sum()
            value -= turnover * cost_per_side
            hold = trade_at[i] * value
            ratio = np.where(np.isfinite(c[i]) & np.isfinite(o[i]) & (o[i] > 0), c[i] / o[i], 1.0)
        else:
            ratio = np.where(np.isfinite(c[i]) & np.isfinite(c[i - 1]) & (c[i - 1] > 0), c[i] / c[i - 1], 1.0)
        hold = hold * ratio
        value = hold.sum()
        eq[i] = value
    return pd.Series(eq, index=days).dropna()
