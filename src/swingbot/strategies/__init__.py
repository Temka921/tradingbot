"""Pre-registered original strategies (design: docs/superpowers/specs/2026-11-21-original-strategies-design.md).

Builders return target weights dated at a signal close (traded at the next open by
``backtest.rebalance.run_weights``) or, for S1, the long frame for ``backtest.portfolio.simulate``.
"""
from __future__ import annotations

import pandas as pd

CASH = "CASH"


def own_series_change(closes: pd.DataFrame, num_lag: int, den_lag: int) -> pd.DataFrame:
    """``close[t-num_lag] / close[t-den_lag] - 1`` counted in each asset's own bars.

    NaN where the asset has no bar at t or fewer than ``den_lag`` prior bars.
    """
    out = {}
    for col in closes.columns:
        s = closes[col].dropna()
        out[col] = s.shift(num_lag) / s.shift(den_lag) - 1
    return pd.DataFrame(out, index=closes.index, columns=closes.columns)
