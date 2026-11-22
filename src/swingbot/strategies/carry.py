"""Funding-rate carry: F1 static BTC+ETH, F2 rotating top-5 (design:
docs/superpowers/specs/2026-11-22-funding-carry-design.md), and the carry engine ``run_carry``.

All inputs are wide frames (dates x coins) keyed by the spot pair name (e.g. "BTCUSDT"); the perp
frames use the same column names. ``funding_daily`` holds the sum of that UTC day's funding rates
(received by the short perp leg; NaN = no funding that day, booked as 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import CASH
from .crypto import weekly_dates

F1_COINS = ("BTCUSDT", "ETHUSDT")


def f1_weights(dates) -> pd.DataFrame:
    """F1: carry in BTC and ETH at 0.5 each on every Sunday of ``dates`` (weekly back to 50/50)."""
    d = weekly_dates(pd.DatetimeIndex(dates))
    w = pd.DataFrame(0.0, index=d, columns=[*F1_COINS, CASH])
    w[list(F1_COINS)] = 0.5
    return w


def f2_weights(funding_daily: pd.DataFrame, eligible: pd.DataFrame, top: int = 5,
               lookback: int = 7) -> pd.DataFrame:
    """F2: every Sunday t, rank coins by mean daily funding over calendar days t-lookback+1..t.

    Sunday's own funding is known by its 24:00 UTC close (last payment 16:00). A coin qualifies if
    it has all ``lookback`` days of funding (non-NaN), a positive mean and ``eligible`` True on t
    (missing = False). The top ``top`` qualifiers get ``1/top`` each; unfilled slots are CASH.
    """
    f = funding_daily.sort_index()
    if len(f):
        f = f.reindex(pd.date_range(f.index[0], f.index[-1], freq="D", name=f.index.name))
    mean = f.rolling(lookback, min_periods=lookback).mean()
    dates = weekly_dates(f.index)
    mean = mean.loc[dates]
    elig = eligible.reindex(index=dates, columns=f.columns)
    elig = elig.astype("boolean").fillna(False).astype(bool)
    cand = elig & (mean > 0)
    w = pd.DataFrame(0.0, index=dates, columns=[*f.columns, CASH])
    for d in dates:
        m = mean.loc[d][cand.loc[d]]
        picks = m.sort_values(ascending=False, kind="mergesort").index[:top]
        w.loc[d, picks] = 1.0 / top
        w.loc[d, CASH] = 1.0 - len(picks) / top
    return w


def _ok(*arrs: np.ndarray) -> np.ndarray:
    out = np.ones(arrs[0].shape, dtype=bool)
    for a in arrs:
        out &= np.isfinite(a) & (a > 0)
    return out


def run_carry(spot_open: pd.DataFrame, spot_close: pd.DataFrame, perp_open: pd.DataFrame,
              perp_close: pd.DataFrame, funding_daily: pd.DataFrame, weights: pd.DataFrame,
              spot_cost: float = 0.001, perp_cost: float = 0.0005) -> pd.Series:
    """Daily equity (start 1.0) of carry positions traded to ``weights``.

    Timing: a weight row dated at a signal close trades at the open of the next day of
    ``spot_close.index``. Equity starts on that first trade day.

    Accounting, per coin with weight w entered at equity V: spot leg ``s = w*V/2`` (drifts with
    spot), perp short notional ``n = w*V/2`` (drifts with the perp price), and collateral
    ``k = w*V/2`` held at 0%; the short's PnL ``-n*perp_ret`` and the funding ``funding_day*n``
    (on the start-of-day notional) are booked into ``k``. Equity = cash + sum(s + k). So the day's
    change is ``sum (w/2)(spot_ret - perp_ret + funding_day)`` with w the drifted weight; on a
    trade day returns run from the open and funding accrues to the post-trade positions.

    Costs: ``spot_cost`` per unit of spot-leg notional traded and ``perp_cost`` per unit of
    perp-leg notional traded; collateral and cash moves are free. As in ``run_weights``, traded
    notional is measured against targets set on the pre-cost value, and the legs are then set on
    the post-cost value.

    Missing prices: a coin needs spot and perp opens on the trade day to be entered; otherwise its
    weight stays in cash. A held coin whose spot or perp price (close, or open on a trade day) is
    missing is closed at the last available prices (the previous closes, or the trade-day open if
    only that day's close is missing), with exit costs charged; the value goes to cash until the
    next rebalance and that day's funding is not earned. Weights must be >= 0 (long carry only).

    ``attrs`` (scalars only): ``negative_funding_share`` (share of position-days with negative
    funding; NaN if none), ``avg_gross_funding_yield`` (mean over days of funding earned / start
    of day equity, x365), ``turnover_per_year`` (sum over trades of traded spot + perp notional
    / pre-trade equity, divided by years = days / 365; a full entry from cash counts 1.0).
    Every weight row must sum to 1 (within 1e-9), CASH included; otherwise ValueError.
    """
    w_all = weights.fillna(0.0)
    sums = w_all.sum(axis=1)
    bad = (sums - 1.0).abs() > 1e-9
    if bad.any():
        raise ValueError(f"weight rows must sum to 1; first bad row {sums.index[bad][0]}: "
                         f"{sums[bad].iloc[0]!r}")
    if (w_all < 0).any().any():
        raise ValueError("carry weights must be >= 0")
    coins = [c for c in w_all.columns if c != CASH]
    days = pd.DatetimeIndex(spot_close.index)

    def arr(df: pd.DataFrame) -> np.ndarray:
        return df.reindex(index=days, columns=coins).to_numpy(float)

    so, sc, po, pc = arr(spot_open), arr(spot_close), arr(perp_open), arr(perp_close)
    fund = np.nan_to_num(arr(funding_daily), nan=0.0)

    trade_at: dict[int, np.ndarray] = {}
    for d, row in w_all.iterrows():
        i = days.searchsorted(d, side="right")  # first day after the signal close
        if i < len(days):
            trade_at[i] = row.reindex(coins).fillna(0.0).to_numpy(float)
    first = min(trade_at) if trade_at else len(days)

    m = len(coins)
    s = np.zeros(m)  # spot leg value
    n = np.zeros(m)  # perp short notional (positive)
    k = np.zeros(m)  # collateral incl. perp PnL and funding
    cash = 1.0
    eq = np.full(len(days), np.nan)
    pos_days = neg_days = 0
    yields: list[float] = []
    turnover = 0.0

    def close_out(mask: np.ndarray) -> None:
        nonlocal cash, turnover
        if not mask.any():
            return
        value = cash + (s + k).sum()
        cost = s[mask].sum() * spot_cost + n[mask].sum() * perp_cost
        if value > 0:
            turnover += (s[mask].sum() + n[mask].sum()) / value
        cash += (s[mask] + k[mask]).sum() - cost
        s[mask] = n[mask] = k[mask] = 0.0

    for i in range(first, len(days)):
        held = n > 0
        if i in trade_at:
            # overnight move to the open; a held coin without both opens exits at last closes
            gone = held & ~_ok(so[i], po[i], sc[i - 1], pc[i - 1])
            close_out(gone)
            held = n > 0
            if held.any():
                rs, rp = so[i][held] / sc[i - 1][held], po[i][held] / pc[i - 1][held]
                k[held] -= n[held] * (rp - 1)
                s[held] *= rs
                n[held] *= rp
            tw = trade_at[i]
            tw = np.where(_ok(so[i], po[i]), tw, 0.0)  # entry needs both opens
            value = cash + (s + k).sum()
            target = tw * value / 2
            ds, dn = np.abs(target - s).sum(), np.abs(target - n).sum()
            if value > 0:
                turnover += (ds + dn) / value
            value -= ds * spot_cost + dn * perp_cost
            s = tw * value / 2
            n = tw * value / 2
            k = tw * value / 2
            cash = value - tw.sum() * value
            # no close on the trade day: exit at the open (the last available price)
            close_out((n > 0) & ~_ok(sc[i], pc[i]))
            held = n > 0
            if held.any():  # open to close
                rs, rp = sc[i][held] / so[i][held], pc[i][held] / po[i][held]
        else:
            gone = held & ~_ok(sc[i], pc[i])
            close_out(gone)
            held = n > 0
            if held.any():
                rs, rp = sc[i][held] / sc[i - 1][held], pc[i][held] / pc[i - 1][held]
        start_value = cash + (s + k).sum()
        if held.any():
            earned = fund[i][held] * n[held]  # funding on the start-of-day notional
            k[held] += earned
            k[held] -= n[held] * (rp - 1)
            s[held] *= rs
            n[held] *= rp
            pos_days += int(held.sum())
            neg_days += int((fund[i][held] < 0).sum())
            yields.append(earned.sum() / start_value if start_value > 0 else 0.0)
        else:
            yields.append(0.0)
        eq[i] = cash + (s + k).sum()

    out = pd.Series(eq, index=days).iloc[first:]
    years = len(out) / 365.0
    out.attrs = {
        "negative_funding_share": float(neg_days / pos_days) if pos_days else float("nan"),
        "avg_gross_funding_yield": float(np.mean(yields) * 365) if yields else float("nan"),
        "turnover_per_year": float(turnover / years) if years > 0 else float("nan"),
    }
    return out
