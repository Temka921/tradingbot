"""Momentum signals, including the cross-sectional 12-1 momentum ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._swings import event
from .registry import register

MOM_LOOKBACK = 252  # ~12 months of trading days
MOM_SKIP = 21  # ~1 month skipped
TOP_PCT = 0.9  # rank percentile strictly above this = top decile


@register("roc20_accel", "momentum")
def roc20_accel(df):
    """Event: ROC20 > 0 and ROC20 - ROC20[t-5] > 0 becomes true."""
    c = df["close"]
    roc = c / c.shift(20) - 1.0
    accel = roc - roc.shift(5)
    return event((roc > 0) & (accel > 0), valid=accel.notna())


def cross_sectional_signals(closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cross-sectional signals over a wide close panel (rows = dates, columns = assets).

    ``mom_12_1_top_decile``: on the first trading day of each month (first row of
    a new calendar month in ``closes.index``) assets are ranked by the 12-1 return
    close[p-21] / close[p-252] - 1 measured at p = the previous row (last bar of
    the prior month), so the ranking uses only closes known before the
    rebalance day. Lags are counted in each asset's own bars. Assets without a
    value at p (fewer than 253 bars of history, or not trading at p) are
    ineligible. Assets with ascending rank percentile > 0.9 among the eligible
    are flagged (20 assets -> top 2) and the flag is held until the next
    rebalance. Days where an asset has no close are False.
    """
    idx = closes.index
    mom = {}
    for col in closes.columns:
        s = closes[col].dropna()
        mom[col] = (s.shift(MOM_SKIP) / s.shift(MOM_LOOKBACK) - 1.0).reindex(idx)
    mom = pd.DataFrame(mom, index=idx, columns=closes.columns)

    flags = np.full(closes.shape, np.nan)
    month = idx.to_period("M")
    for i in range(1, len(idx)):
        if month[i] == month[i - 1]:
            continue
        r = mom.iloc[i - 1]
        ranks = r.rank(pct=True)  # NaN stays NaN -> ineligible
        flags[i] = (ranks > TOP_PCT).to_numpy(bool)
    held = pd.DataFrame(flags, index=idx, columns=closes.columns).ffill()
    out = held.where(held.notna(), 0.0).astype(bool) & closes.notna()
    return {"mom_12_1_top_decile": out}
