"""Pre-registered original strategies (design: docs/superpowers/specs/2026-11-21-original-strategies-design.md).

Builders return target weights dated at a signal close (traded at the next open by
``backtest.rebalance.run_weights``) or, for S1, the long frame for ``backtest.portfolio.simulate``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CASH = "CASH"
MAX_STALE_BARS = 5


def own_series_change(closes: pd.DataFrame, num_lag: int, den_lag: int,
                      max_stale: int = MAX_STALE_BARS) -> pd.DataFrame:
    """``close[t-num_lag] / close[t-den_lag] - 1`` counted in each asset's own bars.

    On every date of ``closes.index`` (the union calendar) the value is taken as of the asset's
    last own bar on or before that date, so a local holiday on a union signal date does not drop
    the asset. NaN if that last own bar is more than ``max_stale`` union bars back (delisted or
    suspended), if the asset has no bar yet, or if it had fewer than ``den_lag`` prior own bars.
    """
    pos = np.arange(len(closes.index))
    out = {}
    for col in closes.columns:
        own = closes[col].notna().to_numpy()
        s = closes[col].dropna()
        val = (s.shift(num_lag) / s.shift(den_lag) - 1).reindex(closes.index).to_numpy(float)
        last = np.maximum.accumulate(np.where(own, pos, -1))
        ok = (last >= 0) & (pos - last <= max_stale)
        out[col] = np.where(ok, val[np.maximum(last, 0)], np.nan)
    return pd.DataFrame(out, index=closes.index, columns=closes.columns)
