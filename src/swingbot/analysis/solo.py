"""Solo edge of each signal: does it beat all days, alone and without help from other families?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingbot import config
from swingbot.analysis.stats import (bh_qvalues, date_matched_edge_test, min_dates_for_test,
                                     per_date_means)
from swingbot.signals import SIGNAL_INFO

METRICS = ("excess", "ret")
STAT_COLS = ["n_all", "n_solo", "n_dates", "mean_all", "mean_ret", "mean_solo", "edge_vs_all_days", "median_all",
             "hit_rate", "profit_factor", "t", "p", "q", "occ_per_year", "insufficient"]


CLIP_PCT = (0.1, 99.9)


def clip_bounds(values: np.ndarray) -> tuple[float, float]:
    """Pooled (CLIP_PCT) percentiles of the non-NaN values; NaN bounds when there are none."""
    v = np.asarray(values, dtype="float64")
    v = v[~np.isnan(v)]
    if not len(v):
        return (np.nan, np.nan)
    lo, hi = np.percentile(v, CLIP_PCT)
    return float(lo), float(hi)


def clip_values(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Symmetric winsorisation at ``bounds``; NaN stays NaN."""
    lo, hi = bounds
    v = np.asarray(values, dtype="float64")
    return v if np.isnan(lo) else np.clip(v, lo, hi)


def _info(signal_info: dict[str, tuple[str, str]] | None) -> dict[str, tuple[str, str]]:
    return SIGNAL_INFO if signal_info is None else signal_info


def _check_columns(panel: pd.DataFrame, info: dict) -> None:
    missing = [s for s in info if s not in panel.columns]
    if missing:
        raise KeyError(f"signal columns missing from panel: {missing}")


def _family_any(panel: pd.DataFrame, info: dict, rows: np.ndarray | None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, (fam, _) in info.items():
        col = panel[name].to_numpy(dtype=bool)
        if rows is not None:
            col = col[rows]
        out[fam] = col.copy() if fam not in out else (out[fam] | col)
    return out


def _other_family_any(fam_any: dict[str, np.ndarray], family: str, n: int) -> np.ndarray:
    other = np.zeros(n, dtype=bool)
    for f, arr in fam_any.items():
        if f != family:
            other |= arr
    return other


def solo_mask(panel: pd.DataFrame, signal: str,
              signal_info: dict[str, tuple[str, str]] | None = None) -> pd.Series:
    """True where ``signal`` fires and no signal of a *different* family fires on that row.

    Signals of the same family firing together do not break solo status.
    """
    info = _info(signal_info)
    _check_columns(panel, info)
    family = info[signal][0]
    fam_any = _family_any(panel, info, None)
    fires = panel[signal].to_numpy(dtype=bool)
    solo = fires & ~_other_family_any(fam_any, family, len(panel))
    return pd.Series(solo, index=panel.index, name=signal)


def _group_codes(sub: pd.DataFrame, group_by: list[str]) -> tuple[np.ndarray, list[tuple]]:
    """Integer group code per row (-1 where a key is NaN) and the key tuple per code."""
    if not group_by:
        return np.zeros(len(sub), dtype=np.int64), [()]
    gb = sub.groupby(group_by, observed=True, dropna=True, sort=True)
    raw = gb.ngroup().to_numpy(dtype="float64")  # NaN for rows with a NaN key
    codes = np.where(np.isnan(raw), -1, raw).astype(np.int64)
    idx = gb.size().index  # same order as ngroup() (sort=True)
    keys = list(idx) if isinstance(idx, pd.MultiIndex) else [(k,) for k in idx]
    return codes, keys


def _baseline_daily(vals: np.ndarray, dates: np.ndarray, codes: np.ndarray,
                    n_groups: int) -> list[pd.Series]:
    """Per group: date-sorted per-date mean of the metric over all non-NaN rows."""
    out = [pd.Series([], dtype="float64") for _ in range(n_groups)]
    ok = (codes >= 0) & ~np.isnan(vals)
    if ok.any():
        df = pd.DataFrame({"g": codes[ok], "d": dates[ok], "v": vals[ok]})
        m = df.groupby(["g", "d"], sort=True)["v"].mean()
        g = m.index.get_level_values("g").to_numpy()
        d = m.index.get_level_values("d")
        bounds = np.searchsorted(g, np.arange(n_groups + 1))
        for k in range(n_groups):
            lo, hi = bounds[k], bounds[k + 1]
            out[k] = pd.Series(m.to_numpy()[lo:hi], index=d[lo:hi])
    return out


def _span_years(dates: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.full(n_groups, np.nan)
    ok = codes >= 0
    if ok.any():
        s = pd.Series(dates[ok]).groupby(codes[ok])
        span = (s.max() - s.min()).dt.days.to_numpy() / 365.25
        out[s.max().index.to_numpy()] = np.where(span > 0, span, np.nan)
    return out


def date_weighted_mean(v: np.ndarray, d: np.ndarray) -> float:
    """Mean of per-date means of the non-NaN values; NaN when there are none."""
    ok = ~np.isnan(v)
    return float(per_date_means(v[ok], d[ok]).mean()) if ok.any() else np.nan


def _row_stats(v: np.ndarray, d: np.ndarray, solo: np.ndarray, base_daily: pd.Series,
               horizon: int, r: np.ndarray) -> dict:
    mean_ret = date_weighted_mean(r, d)
    ok = ~np.isnan(v)
    v, d, solo = v[ok], d[ok], solo[ok]
    n = len(v)
    daily = per_date_means(v, d)
    mean_all = float(daily.mean()) if n else np.nan
    res = date_matched_edge_test(daily, base_daily, horizon, n)
    n_solo = int(solo.sum())
    mean_solo = float(per_date_means(v[solo], d[solo]).mean()) if n_solo else np.nan
    if n:
        gains = float(v[v > 0].sum())
        losses = float(-v[v < 0].sum())
        if losses > 0:
            pf = gains / losses
        else:
            pf = np.inf if gains > 0 else np.nan
        median, hit = float(np.median(v)), float((v > 0).mean())
    else:
        pf = median = hit = np.nan
    return {"n_all": n, "n_solo": n_solo, "n_dates": int(len(daily)), "mean_all": mean_all,
            "mean_ret": mean_ret, "mean_solo": mean_solo, "edge_vs_all_days": res.mean, "median_all": median,
            "hit_rate": hit, "profit_factor": pf, "t": res.t, "p": res.p}


def clipped_or_nan(raw: np.ndarray | None, n: int) -> np.ndarray:
    """``raw`` clipped at its own pooled CLIP_PCT bounds; ``n`` NaNs when the column is absent (None)."""
    if raw is None:
        return np.full(n, np.nan)
    raw = np.asarray(raw, dtype="float64")
    return clip_values(raw, clip_bounds(raw))


def solo_table(panel: pd.DataFrame, horizon: int = 10, metric: str = "excess",
               group_by: list[str] | None = None, split: str | None = "discovery",
               signal_info: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """One row per signal (per present group) with its standalone edge.

    The outcome column is ``f"{metric}_{direction}_{horizon}"`` using each
    signal's direction. Only rows of ``split`` are used (``None`` = all rows).
    Before any statistic the metric is clipped at its CLIP_PCT (0.1, 99.9)
    percentiles pooled over every row of the split (pass one market per call);
    signal rows and the baseline use the same clipped values. The bounds are in
    ``table.attrs["clip_bounds"]``.
    Rows with a NaN group key are dropped; NaN metric values are ignored, so
    ``n_all``/``n_solo`` count occurrences with a known outcome and
    ``n_dates`` counts the distinct dates among them.

    - ``mean_all``/``mean_solo``: date-weighted means (mean of per-date means,
      equal-weighting dates, not occurrences) of the metric itself.
    - ``mean_ret``: date-weighted mean of the absolute return net of costs,
      ``f"ret_{direction}_{horizon}"``, on the signal's rows with a known return,
      clipped like the metric (pooled CLIP_PCT bounds over the split, per
      direction). Equals ``mean_all`` when ``metric == "ret"``; NaN when the
      panel has no ``ret_*`` column. For shorts, ``excess`` is a short-asset /
      long-benchmark pair trade, so it can be positive while the plain short
      loses money; ``mean_ret`` shows the latter.
    - ``edge_vs_all_days``: date-matched edge. For each signal date,
      d = mean of the signal's values that date - mean of all non-NaN rows
      (signal or not) that date in the same group/split; the column is the
      date-weighted mean of d.
    - ``t``/``p``: HAC test (see ``stats.hac_test``) on the date-sorted d
      series, so p tests whether the signal beats the same-day all-rows
      average, not whether its return is non-zero. NaN with < 5*horizon dates.
    - ``profit_factor``: sum of gains / |sum of losses|; inf with gains but no
      losses; NaN with no rows (or all-zero values).
    - ``occ_per_year``: n_all / span in years of all dates in the group/split.
    - ``q``: Benjamini-Hochberg across every row of the returned table.
    - ``insufficient``: n_all or n_dates below MIN_N (no groups) or
      MIN_N_SLICE (groups), or n_dates below 5*horizon (where p is NaN);
      such rows must never be ranked as winners.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    info = _info(signal_info)
    _check_columns(panel, info)
    group_by = list(group_by or [])
    min_n = config.MIN_N_SLICE if group_by else config.MIN_N
    # no row may be "sufficient" without a p-value (hac_test needs 5*horizon dates)
    min_dates = max(min_n, min_dates_for_test(horizon))

    rows = np.flatnonzero((panel["split"] == split).to_numpy()) if split is not None else None
    sub = panel[["date", *group_by]] if rows is None else panel[["date", *group_by]].iloc[rows]
    dates = sub["date"].to_numpy()
    codes, keys = _group_codes(sub, group_by)
    n_groups = len(keys)
    span = _span_years(dates, codes, n_groups)

    def col(name: str) -> np.ndarray:
        a = panel[name].to_numpy()
        return a if rows is None else a[rows]

    directions = sorted({d for _, d in info.values()})
    raw = {d: col(f"{metric}_{d}_{horizon}").astype("float64") for d in directions}
    # winsorise once, over every row of this split, before any statistic: signal rows and the
    # same-day baseline see the same clipped values (guards against bad data splices)
    clip = {d: clip_bounds(raw[d]) for d in directions}
    vals = {d: clip_values(raw[d], clip[d]) for d in directions}
    del raw
    # absolute return net of costs, clipped the same way (pooled over the split, per direction)
    rets = {d: vals[d] if metric == "ret" else
            clipped_or_nan(col(f"ret_{d}_{horizon}") if f"ret_{d}_{horizon}" in panel.columns else None,
                           len(dates))
            for d in directions}
    base = {d: _baseline_daily(vals[d], dates, codes, n_groups) for d in directions}
    fam_any = _family_any(panel, info, rows)

    out = []
    for name, (family, direction) in info.items():
        fires = col(name).astype(bool) & (codes >= 0)
        solo = fires & ~_other_family_any(fam_any, family, len(fires))
        idx = np.flatnonzero(fires)
        g, v, d, s, r = codes[idx], vals[direction][idx], dates[idx], solo[idx], rets[direction][idx]
        order = np.argsort(g, kind="stable")
        g, v, d, s, r = g[order], v[order], d[order], s[order], r[order]
        bounds = np.searchsorted(g, np.arange(n_groups + 1))
        for k in range(n_groups):
            lo, hi = bounds[k], bounds[k + 1]
            st = _row_stats(v[lo:hi], d[lo:hi], s[lo:hi], base[direction][k], horizon, r[lo:hi])
            st["occ_per_year"] = st["n_all"] / span[k] if np.isfinite(span[k]) else np.nan
            st["insufficient"] = (st["n_all"] < min_n) or (st["n_dates"] < min_dates)
            out.append({"signal": name, "family": family, "direction": direction,
                        **dict(zip(group_by, keys[k])), **st})

    table = pd.DataFrame(out, columns=["signal", "family", "direction", *group_by,
                                       *[c for c in STAT_COLS if c != "q"]])
    table.insert(table.columns.get_loc("p") + 1, "q", bh_qvalues(table["p"].to_numpy(dtype="float64")))
    table["n_all"] = table["n_all"].astype(np.int64)
    table["n_solo"] = table["n_solo"].astype(np.int64)
    table["n_dates"] = table["n_dates"].astype(np.int64)
    table["insufficient"] = table["insufficient"].astype(bool)
    table.attrs["clip_bounds"] = {f"{metric}_{d}_{horizon}": clip[d] for d in directions}
    return table
