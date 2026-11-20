"""Solo edge of each signal: does it beat all days, alone and without help from other families?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingbot import config
from swingbot.analysis.stats import bh_qvalues, hac_test, per_date_means
from swingbot.signals import SIGNAL_INFO

METRICS = ("excess", "ret")
STAT_COLS = ["n_all", "n_solo", "mean_all", "mean_solo", "edge_vs_all_days", "median_all",
             "hit_rate", "profit_factor", "t", "p", "q", "occ_per_year", "insufficient"]


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


def _baseline_means(vals: np.ndarray, dates: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Date-clustered mean of the metric over all rows, per group."""
    ok = (codes >= 0) & ~np.isnan(vals)
    out = np.full(n_groups, np.nan)
    if ok.any():
        df = pd.DataFrame({"g": codes[ok], "d": dates[ok], "v": vals[ok]})
        m = df.groupby(["g", "d"], sort=False)["v"].mean().groupby(level="g").mean()
        out[m.index.to_numpy()] = m.to_numpy()
    return out


def _span_years(dates: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.full(n_groups, np.nan)
    ok = codes >= 0
    if ok.any():
        s = pd.Series(dates[ok]).groupby(codes[ok])
        span = (s.max() - s.min()).dt.days.to_numpy() / 365.25
        out[s.max().index.to_numpy()] = np.where(span > 0, span, np.nan)
    return out


def _row_stats(v: np.ndarray, d: np.ndarray, solo: np.ndarray, horizon: int) -> dict:
    ok = ~np.isnan(v)
    v, d, solo = v[ok], d[ok], solo[ok]
    n = len(v)
    res = hac_test(per_date_means(v, d), horizon, n)
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
    return {"n_all": n, "n_solo": n_solo, "mean_all": res.mean, "mean_solo": mean_solo,
            "median_all": median, "hit_rate": hit, "profit_factor": pf,
            "t": res.t, "p": res.p}


def solo_table(panel: pd.DataFrame, horizon: int = 10, metric: str = "excess",
               group_by: list[str] | None = None, split: str | None = "discovery",
               signal_info: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """One row per signal (per present group) with its standalone edge.

    The outcome column is ``f"{metric}_{direction}_{horizon}"`` using each
    signal's direction. Only rows of ``split`` are used (``None`` = all rows).
    Rows with a NaN group key are dropped; NaN metric values are ignored, so
    ``n_all``/``n_solo`` count occurrences with a known outcome.

    - ``mean_all``/``mean_solo``: date-clustered means (mean of per-date means,
      equal-weighting dates). ``t``/``p`` come from the HAC test of mean_all.
    - ``edge_vs_all_days``: mean_all minus the same clustered mean over every
      row of the group/split, signal or not.
    - ``profit_factor``: sum of gains / |sum of losses|; inf with gains but no
      losses; NaN with no rows (or all-zero values).
    - ``occ_per_year``: n_all / span in years of all dates in the group/split.
    - ``q``: Benjamini-Hochberg across every row of the returned table.
    - ``insufficient``: n_all < MIN_N (no groups) or < MIN_N_SLICE (groups);
      such rows must never be ranked as winners.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    info = _info(signal_info)
    _check_columns(panel, info)
    group_by = list(group_by or [])
    min_n = config.MIN_N_SLICE if group_by else config.MIN_N

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
    vals = {d: col(f"{metric}_{d}_{horizon}").astype("float64") for d in directions}
    base = {d: _baseline_means(vals[d], dates, codes, n_groups) for d in directions}
    fam_any = _family_any(panel, info, rows)

    out = []
    for name, (family, direction) in info.items():
        fires = col(name).astype(bool) & (codes >= 0)
        solo = fires & ~_other_family_any(fam_any, family, len(fires))
        idx = np.flatnonzero(fires)
        g, v, d, s = codes[idx], vals[direction][idx], dates[idx], solo[idx]
        order = np.argsort(g, kind="stable")
        g, v, d, s = g[order], v[order], d[order], s[order]
        bounds = np.searchsorted(g, np.arange(n_groups + 1))
        for k in range(n_groups):
            lo, hi = bounds[k], bounds[k + 1]
            st = _row_stats(v[lo:hi], d[lo:hi], s[lo:hi], horizon)
            st["edge_vs_all_days"] = st["mean_all"] - base[direction][k]
            st["occ_per_year"] = st["n_all"] / span[k] if np.isfinite(span[k]) else np.nan
            st["insufficient"] = st["n_all"] < min_n
            out.append({"signal": name, "family": family, "direction": direction,
                        **dict(zip(group_by, keys[k])), **st})

    table = pd.DataFrame(out, columns=["signal", "family", "direction", *group_by,
                                       *[c for c in STAT_COLS if c != "q"]])
    table.insert(table.columns.get_loc("p") + 1, "q", bh_qvalues(table["p"].to_numpy(dtype="float64")))
    table["n_all"] = table["n_all"].astype(np.int64)
    table["n_solo"] = table["n_solo"].astype(np.int64)
    table["insufficient"] = table["insufficient"].astype(bool)
    return table
