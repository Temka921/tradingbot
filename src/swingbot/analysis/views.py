"""Universe views: ETFs and crypto are primary results, single stocks are suspect (survivorship)."""
from __future__ import annotations

import numpy as np
import pandas as pd

VIEWS = ("etf", "crypto", "stock")
PRIMARY = ("etf", "crypto")
SUSPECT = ("stock",)
SEGMENT_KEY = {"etf": "sector", "crypto": "cap_bucket", "stock": "sector"}


def _bool_col(panel: pd.DataFrame, col: str, default: bool) -> np.ndarray:
    if col not in panel:
        return np.full(len(panel), default, dtype=bool)
    return panel[col].astype("boolean").fillna(default).to_numpy(dtype=bool)


def universe_views(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Split market panels into ``VIEWS`` (non-empty only, in VIEWS order).

    A missing ``is_etf`` column means all non-ETF; a missing ``eligible`` column means all eligible.
    Rows with ``eligible == False`` are dropped. ``delisted`` is never used (it is hindsight).
    """
    raw: dict[str, pd.DataFrame] = {}
    for market, panel in panels.items():
        found = sorted(map(str, panel["market"].dropna().unique()))
        if found != [market]:
            raise ValueError(f"panel for market {market!r} must contain only that market; found {found}")
        panel = panel[_bool_col(panel, "eligible", True)]
        if market == "stock":
            etf = _bool_col(panel, "is_etf", False)
            raw["etf"], raw["stock"] = panel[etf], panel[~etf]
        elif market == "crypto":
            raw["crypto"] = panel
        else:
            raw[market] = panel
    out = {}
    for v in VIEWS:
        df = raw.get(v)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["view"] = pd.Categorical([v] * len(df), categories=list(VIEWS))
        out[v] = df
    return out
