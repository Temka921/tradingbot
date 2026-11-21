"""Capacity-limited, fixed-horizon portfolio simulator for a signal rule."""
from __future__ import annotations

import numpy as np
import pandas as pd


ALL_RULE = "__all__"


def rule_mask(panel: pd.DataFrame, rule: str) -> np.ndarray:
    """"a+b" -> A and B both fire; otherwise the single signal column.

    The reserved rule ``ALL_RULE`` ("__all__") selects every row: it is the same-model universe
    baseline (enter any eligible row, same slots and random fill as the strategy).
    """
    mask = np.ones(len(panel), dtype=bool)
    if rule == ALL_RULE:
        return mask
    cols = rule.split("+")
    for c in cols:
        mask &= panel[c].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return mask


def simulate(panel: pd.DataFrame, rule: str, direction: str, split: str, horizon: int = 10,
             max_positions: int = 10, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate ``rule`` as ``max_positions`` equal slots over the panel rows of ``split``.

    A firing on (asset, t) enters at the next open and exits at the close of the asset's
    ``horizon``-th subsequent panel row; the trade return is ``ret_{direction}_{horizon}``
    (already net of costs). Trades with NaN return or no such row are skipped. A trade holds a
    slot from the first panel date after t through its exit date inclusive. Each signal date the
    free slots are filled from that day's candidates in random order (one default_rng(seed) per
    call, used in date order). Each trade is sized ``slot_value = equity / max_positions`` with
    the equity booked through the signal date (known when the decision is made); its P&L
    ``slot_value * ret`` is booked on the exit date. No leverage, no compounding inside a trade.

    Approximation: views are filtered to eligible rows, so an asset's subsequent rows can have
    holes where it was ineligible. The outcome ret was computed on the asset's full bar series,
    while the exit date is the panel date ``horizon`` rows later, which can be later than the true
    exit when holes exist. That is acceptable for capacity accounting.

    Returns ``(trades, equity)``. ``trades`` has date (signal day), asset, entry_idx (position of
    the entry date in the split's sorted dates), exit_date, ret. ``equity`` starts at 1.0, is
    indexed by every distinct date of the split, and carries scalar ``attrs`` only (so curves
    concat cleanly): exposure (mean busy slots / max_positions), max_positions and n_trades.

    Only trades with a known ret and exit date reserve slots (skipped trades never occupy one).
    One asset may hold overlapping positions: each firing is its own trade, as in the statistics.
    """
    ret_col = f"ret_{direction}_{horizon}"
    p = panel.loc[(panel["split"] == split).to_numpy()]
    dates = pd.DatetimeIndex(np.sort(p["date"].unique()), name="date")
    n_dates = len(dates)
    di = dates.get_indexer(p["date"])
    asset = p["asset"].to_numpy()
    ret = p[ret_col].to_numpy(dtype=float)
    cand = rule_mask(p, rule)

    # Exit date index: the asset's horizon-th subsequent row.
    tmp = pd.DataFrame({"a": asset, "d": di.astype(float)})
    order = np.lexsort((di, pd.factorize(asset)[0]))
    exit_di = np.full(len(p), np.nan)
    shifted = tmp.iloc[order].groupby("a", sort=False)["d"].shift(-horizon).to_numpy()
    exit_di[order] = shifted

    ok = cand & ~np.isnan(ret) & ~np.isnan(exit_di)
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(di[idx], kind="stable")]
    sig_day = di[idx]
    bounds = np.searchsorted(sig_day, np.arange(n_dates + 1))

    rng = np.random.default_rng(seed)
    pnl = np.zeros(n_dates)
    release = np.zeros(n_dates + 1, dtype=int)  # trades whose last busy date is d
    busy_diff = np.zeros(n_dates + 2, dtype=int)
    equity = np.empty(n_dates)
    eq = 1.0
    open_n = 0
    taken: list[np.ndarray] = []
    for d in range(n_dates):
        eq += pnl[d]
        equity[d] = eq
        open_n -= release[d]  # trades with exit on d are free for entries after d
        lo, hi = bounds[d], bounds[d + 1]
        # Slots busy on d+1 = open trades (exit >= d+1) -> open_n after releasing exits on d.
        if hi > lo and d + 1 < n_dates:
            free = max_positions - open_n
            if free > 0:
                rows = idx[lo:hi]
                if len(rows) > free:
                    rows = rows[rng.permutation(len(rows))[:free]]
                sv = eq / max_positions
                ex = exit_di[rows].astype(int)
                np.add.at(pnl, ex, sv * ret[rows])
                np.add.at(release, ex, 1)
                busy_diff[d + 1] += len(rows)
                np.add.at(busy_diff, ex + 1, -1)
                open_n += len(rows)
                taken.append(rows)

    rows = np.concatenate(taken) if taken else np.array([], dtype=int)
    trades = pd.DataFrame({
        "date": dates[di[rows]],
        "asset": asset[rows],
        "entry_idx": di[rows] + 1,
        "exit_date": dates[exit_di[rows].astype(int)] if len(rows) else pd.DatetimeIndex([]),
        "ret": ret[rows],
    })
    out = pd.Series(equity, index=dates, name="equity")
    busy = np.cumsum(busy_diff)[:n_dates]
    out.attrs["exposure"] = float(busy.mean() / max_positions) if n_dates else float("nan")
    out.attrs["max_positions"] = max_positions
    out.attrs["n_trades"] = len(trades)
    return trades, out


def _max_dd(s: pd.Series) -> float:
    return float((s / s.cummax() - 1.0).min())


def _cagr(s: pd.Series) -> float:
    years = (s.index[-1] - s.index[0]).days / 365.25
    if years <= 0 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1)


def _sharpe_monthly(s: pd.Series) -> float:
    """Annualised Sharpe of month-end returns (zero rate), first partial month included."""
    me = s.resample("ME").last().dropna()
    prev = me.shift(1)
    prev.iloc[0] = s.iloc[0]  # include the first partial month
    monthly = (me / prev - 1.0)
    sd = monthly.std()
    return float(monthly.mean() / sd * np.sqrt(12)) if len(monthly) > 1 and sd > 0 else float("nan")


def metrics(equity: pd.Series, bench_close: pd.Series | None = None) -> dict:
    """Summary stats of an equity curve; bench values are NaN without ``bench_close``.

    ``sharpe_monthly`` and ``bench_sharpe_monthly`` use the same month-end method.
    """
    nan = float("nan")
    sharpe = _sharpe_monthly(equity)
    exposure = float(equity.attrs.get("exposure", nan))
    out = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": _cagr(equity),
        "max_drawdown": _max_dd(equity),
        "sharpe_monthly": sharpe,
        "n_trades": equity.attrs.get("n_trades", nan),
        "exposure": exposure,
        "bench_total_return": nan, "bench_cagr": nan, "bench_max_drawdown": nan,
        "bench_sharpe_monthly": nan,
    }
    if bench_close is not None:
        b = bench_close.dropna()
        b = b.loc[(b.index >= equity.index[0]) & (b.index <= equity.index[-1])]
        if len(b) > 1:
            out["bench_total_return"] = float(b.iloc[-1] / b.iloc[0] - 1)
            out["bench_cagr"] = _cagr(b)
            out["bench_max_drawdown"] = _max_dd(b)
            out["bench_sharpe_monthly"] = _sharpe_monthly(b)
    return out


def yearly_returns(equity: pd.Series) -> pd.Series:
    """Calendar-year return of the equity curve (a year with no change is 0.0)."""
    last = equity.groupby(equity.index.year).last()
    prev = last.shift(1)
    prev.iloc[0] = equity.iloc[0]
    return (last / prev - 1.0).rename("return")
