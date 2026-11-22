"""General target-weight rebalance engine (longs, shorts, free cash, short funding).

Timing: a weight row dated at a signal close is traded at the next bar of ``closes.index``
(the union calendar) at its open. Between trades holdings drift with prices. Missing prices are
forward-filled (design rule): a position whose asset has no bar (a delisted coin, or a stock on
a local holiday) is carried at its last close, and a rebalance on such a day trades it at that
stale last price. Cash columns have a constant price of 1.0 and no trading cost.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    return np.where(ok, num / np.where(ok, den, 1.0), 1.0)


def run_weights(opens: pd.DataFrame, closes: pd.DataFrame, weights: pd.DataFrame,
                cost_per_side: float, cash_cols: Iterable[str] = ("CASH",),
                short_funding_annual: float = 0.0) -> pd.Series:
    """Daily equity (start 1.0) of trading to ``weights`` at the open after each signal date.

    - Negative weights are shorts; a short loses when its price rises (holdings drift).
    - Turnover cost ``cost_per_side`` per unit of value traded, on non-cash columns only.
    - Funding: every close, ``short_funding_annual / 365 * |current short notional|`` is paid.
    - ``equity.attrs`` (scalars only): ``worst_short_loss`` = the worst price move of any short
      within one holding window between consecutive rebalances (entry open to the next
      rebalance open, or the last close; reset at each rebalance even if the short is kept), as
      a fraction (positive = loss for the short); NaN if no short was ever held.
    - Every weight row must sum to 1 (within 1e-9), cash included; otherwise ValueError.
    """
    sums = weights.fillna(0.0).sum(axis=1)
    bad = (sums - 1.0).abs() > 1e-9
    if bad.any():
        raise ValueError(f"weight rows must sum to 1; first bad row {sums.index[bad][0]}: "
                         f"{sums[bad].iloc[0]!r}")
    cols = list(weights.columns)
    cash = np.array([c in set(cash_cols) for c in cols])
    days = closes.index
    c = closes.reindex(columns=cols).ffill().to_numpy(float, copy=True)
    o_raw = opens.reindex(index=days, columns=cols).to_numpy(float)
    prev_c = np.vstack([np.full((1, len(cols)), np.nan), c[:-1]])
    o = np.where(np.isfinite(o_raw), o_raw, prev_c)  # no bar today -> last known close
    c[:, cash] = 1.0
    o[:, cash] = 1.0

    trade_at = {}
    for d, row in weights.fillna(0.0).iterrows():
        i = days.searchsorted(d, side="right")  # first bar after the signal close
        if i < len(days):
            trade_at[i] = row.to_numpy(float)
    first = min(trade_at) if trade_at else len(days)
    daily_fund = short_funding_annual / 365.0

    eq = np.full(len(days), np.nan)
    hold = np.zeros(len(cols))  # signed value per asset at the previous close
    resid = 0.0  # funding paid since the last rebalance (not attached to any column)
    entry = np.full(len(cols), np.nan)  # entry price of each open short
    short_moves: list[float] = []
    for i in range(first, len(days)):
        if i in trade_at:
            hold = hold * _ratio(o[i], c[i - 1]) if i > first else hold
            short_moves += list(o[i][np.isfinite(entry)] / entry[np.isfinite(entry)] - 1)
            value = hold.sum() + resid if i > first else 1.0
            target = trade_at[i] * value
            turnover = np.abs(target - hold)[~cash].sum()
            value -= turnover * cost_per_side
            hold = trade_at[i] * value
            resid = 0.0
            entry = np.where(trade_at[i] < 0, o[i], np.nan)
            hold = hold * _ratio(c[i], o[i])
        else:
            hold = hold * _ratio(c[i], c[i - 1])
        resid -= daily_fund * np.abs(hold[hold < 0]).sum()
        eq[i] = hold.sum() + resid
    if trade_at and np.isfinite(entry).any():
        short_moves += list(c[-1][np.isfinite(entry)] / entry[np.isfinite(entry)] - 1)
    out = pd.Series(eq, index=days).dropna()
    out.attrs = {"worst_short_loss": float(max(short_moves)) if short_moves else float("nan")}
    return out
