"""Regime labels. Every label at date t uses only data up to the close of t."""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from swingbot import config
from swingbot.indicators import adx, atr, sma

VOL_WINDOW = 504
VOL_MIN = 252
TREND_SLOPE_LAG = 20


def _trailing_pct_rank(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Fraction of the trailing `window` values (incl. current) that are <= current."""
    v = x.to_numpy(dtype=float)
    if len(v) == 0:
        return pd.Series([], index=x.index, dtype=float)
    padded = np.concatenate([np.full(window - 1, np.nan), v])
    win = sliding_window_view(padded, window)
    count = (~np.isnan(win)).sum(axis=1)
    rank = (win <= v[:, None]).sum(axis=1) / np.maximum(count, 1)
    ok = (count >= min_periods) & ~np.isnan(v)
    return pd.Series(np.where(ok, rank, np.nan), index=x.index)


def label_regimes(df: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    out = pd.DataFrame(index=df.index)

    s200 = sma(c, 200)
    prev = s200.shift(TREND_SLOPE_LAG)
    trend = np.where((c > s200) & (s200 > prev), "up",
                     np.where((c < s200) & (s200 < prev), "down", "sideways"))
    out["trend_regime"] = pd.Series(trend, index=df.index).where(s200.notna(), "unknown")

    rank = _trailing_pct_rank(atr(df, 14) / c, VOL_WINDOW, VOL_MIN)
    vol = np.where(rank < 1 / 3, "low", np.where(rank > 2 / 3, "high", "mid"))
    out["vol_regime"] = pd.Series(vol, index=df.index).where(rank.notna(), "unknown")

    bc = bench["close"]
    bs200 = sma(bc, 200)
    bull = pd.Series(np.where(bc > bs200, 1.0, 0.0), index=bench.index).where(bs200.notna())
    # forward-fill only: an asset date takes the latest benchmark value at or before it
    aligned = bull.reindex(bull.index.union(df.index)).ffill().reindex(df.index)
    out["market_regime"] = pd.Series(np.where(aligned == 1.0, "bull", "bear"), index=df.index) \
        .where(aligned.notna(), "unknown")

    a, _, _ = adx(df, 14)
    adx_lab = np.where(a > 25, "trending", np.where(a < 20, "ranging", "neutral"))
    out["adx_regime"] = pd.Series(adx_lab, index=df.index).where(a.notna(), "unknown")
    return out


def liquidity_eligible(closes: pd.DataFrame, volumes: pd.DataFrame, top_n: int,
                       window: int = config.LIQUIDITY_WINDOW) -> pd.DataFrame:
    """Point-in-time top-``top_n`` liquidity flag (dates x assets, bool).

    Dollar volume = close x volume; trailing ``window``-day median (``min_periods=window``)
    is taken on each asset's own bars, then aligned to the union of dates. Per date, assets
    rank descending (``method="first"``); eligible when rank <= ``top_n``. Missing -> False.
    Uses data up to the close of t only.
    """
    dv = closes * volumes
    med = pd.DataFrame(index=dv.index, columns=dv.columns, dtype=float)
    for col in dv.columns:
        own = dv[col].dropna()
        med[col] = own.rolling(window, min_periods=window).median().reindex(dv.index)
    rank = med.rank(axis=1, ascending=False, method="first")
    return (rank <= top_n).astype(bool)
