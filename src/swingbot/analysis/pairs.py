"""Pair analysis: does firing together beat the better signal firing alone?

For every unordered pair of same-direction signals (A, B) the rows of a split
are cut into three disjoint groups: A and B both fire (AB), only A, only B.
Each group's edge is date-matched against the same-day all-rows average (as in
``solo_table``); the lift is the AB edge minus the better single-only edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from swingbot import config
from swingbot.analysis.solo import METRICS, clip_bounds, clip_values
from swingbot.analysis.stats import (TestResult, bh_qvalues, diff_test, hac_test,
                                     min_dates_for_test)
from swingbot.signals import SIGNAL_INFO

PAIR_VERDICTS = ("insufficient", "synergy", "conflict", "redundant", "not significant")
PAIR_COLS = [
    "a", "b", "family_a", "family_b", "direction", "jaccard",
    "n_ab", "n_dates_ab", "n_a_only", "n_b_only",
    "mean_ab", "edge_ab", "t_ab", "p_ab", "q_ab",
    "edge_a_only", "edge_b_only", "cond_a_given_b", "cond_b_given_a",
    "lift", "t_lift", "p_lift", "q_lift",
    "insufficient", "verdict",
]
Q_SIGNIFICANT = 0.10
NAN = float("nan")


@dataclass
class EdgeContext:
    horizon: int
    date_code: np.ndarray                  # int per selected row, codes in ascending date order
    n_dates_total: int
    vals: dict[str, np.ndarray]            # clipped metric per direction (NaN kept)
    base: dict[str, np.ndarray]            # per-date all-rows mean per direction, by date code
    fires: dict[str, np.ndarray]           # bool per signal over the selected rows
    info: dict[str, tuple[str, str]]


@dataclass
class GroupStats:
    n: int                                 # rows with a known (non-NaN) outcome
    n_dates: int                           # distinct dates among them
    mean: float                            # date-weighted mean of the metric
    edge: TestResult                       # date-matched edge vs ctx.base


def build_context(panel: pd.DataFrame, horizon: int = 10, metric: str = "excess",
                  split: str | None = "discovery",
                  signal_info: dict[str, tuple[str, str]] | None = None,
                  rows_mask: np.ndarray | None = None) -> EdgeContext:
    """Arrays shared by every group statistic on the selected rows (split ∩ ``rows_mask``).

    The metric ``f"{metric}_{direction}_{horizon}"`` is clipped at the
    ``solo.CLIP_PCT`` percentiles pooled over every selected row per direction,
    exactly as ``solo_table`` does; the baseline is the per-date mean of all
    non-NaN selected rows of the clipped values.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    info = SIGNAL_INFO if signal_info is None else signal_info
    missing = [s for s in info if s not in panel.columns]
    if missing:
        raise KeyError(f"signal columns missing from panel: {missing}")

    sel = np.ones(len(panel), dtype=bool)
    if split is not None:
        sel &= (panel["split"] == split).to_numpy(dtype=bool)
    if rows_mask is not None:
        sel &= np.asarray(rows_mask, dtype=bool)
    rows = np.flatnonzero(sel)

    date_code, uniq = pd.factorize(panel["date"].to_numpy()[rows], sort=True)
    date_code = date_code.astype(np.int64)
    n_dates_total = len(uniq)

    directions = sorted({d for _, d in info.values()})
    vals: dict[str, np.ndarray] = {}
    base: dict[str, np.ndarray] = {}
    for d in directions:
        raw = panel[f"{metric}_{d}_{horizon}"].to_numpy(dtype="float64")[rows]
        v = clip_values(raw, clip_bounds(raw))
        ok = ~np.isnan(v)
        sums = np.bincount(date_code[ok], weights=v[ok], minlength=n_dates_total)
        cnt = np.bincount(date_code[ok], minlength=n_dates_total)
        with np.errstate(invalid="ignore", divide="ignore"):
            base[d] = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
        vals[d] = v
    fires = {s: panel[s].to_numpy(dtype=bool)[rows] for s in info}
    return EdgeContext(int(horizon), date_code, n_dates_total, vals, base, fires, dict(info))


def group_stats(ctx: EdgeContext, mask: np.ndarray, direction: str) -> GroupStats:
    """Count, date-weighted mean and date-matched edge of the rows in ``mask``.

    NaN metric values are ignored. Per date, d = mean of the group's values -
    ctx.base; the edge is ``hac_test`` on the date-sorted d series.
    """
    v = ctx.vals[direction]
    ok = np.asarray(mask, dtype=bool) & ~np.isnan(v)
    n = int(ok.sum())
    if n == 0:
        return GroupStats(0, 0, NAN, TestResult(NAN, NAN, NAN, NAN, 0, 0))
    dc = ctx.date_code[ok]
    sums = np.bincount(dc, weights=v[ok], minlength=ctx.n_dates_total)
    cnt = np.bincount(dc, minlength=ctx.n_dates_total)
    has = np.flatnonzero(cnt > 0)          # ascending date codes = date order
    daily = sums[has] / cnt[has]
    d = daily - ctx.base[direction][has]
    edge = hac_test(pd.Series(d), ctx.horizon, n)
    return GroupStats(n, int(len(has)), float(daily.mean()), edge)


def combo_sufficient(n_ab, n_dates_ab, horizon: int):
    """The single sufficiency rule for a combo group: n >= MIN_N and n_dates >= max(MIN_N, 5h).

    Works on scalars and on Series/arrays alike.
    """
    min_dates = max(config.MIN_N, min_dates_for_test(horizon))
    return (n_ab >= config.MIN_N) & (n_dates_ab >= min_dates)


def add_pair_verdicts(t: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Set ``verdict`` (one of ``PAIR_VERDICTS``), first matching rule wins.

    - insufficient (but "redundant" when the combo itself is sufficient, see
      ``combo_sufficient``, and an only-group is empty: one signal never fires
      without the other);
    - synergy: q_lift < 0.10, lift > 0 and mean_ab > 0;
    - conflict: q_lift < 0.10 and lift < 0;
    - redundant: not significant (q_lift NaN or >= 0.10) and
      jaccard >= config.REDUNDANT_JACCARD;
    - not significant otherwise (including q_lift < 0.10 with lift > 0 but mean_ab <= 0).
    """
    t = t.copy()
    q = t["q_lift"].astype("float64")
    lift = t["lift"].astype("float64")
    sig = q < Q_SIGNIFICANT
    insufficient = t["insufficient"].astype(bool)
    only_empty = (t["n_a_only"] == 0) | (t["n_b_only"] == 0)
    conds = [
        insufficient & only_empty & combo_sufficient(t["n_ab"], t["n_dates_ab"], horizon),
        insufficient,
        sig & (lift > 0) & (t["mean_ab"].astype("float64") > 0),
        sig & (lift < 0),
        ~sig & (t["jaccard"].astype("float64") >= config.REDUNDANT_JACCARD),
    ]
    choices = ["redundant", "insufficient", "synergy", "conflict", "redundant"]
    t["verdict"] = np.select(conds, choices, default="not significant") if len(t) else []
    t["verdict"] = t["verdict"].astype(object)
    return t


def pair_table(panel: pd.DataFrame, horizon: int = 10, metric: str = "excess",
               split: str | None = "discovery",
               signal_info: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """One row per unordered same-direction pair (a before b in registry order).

    Groups are disjoint: AB = both fire, A-only, B-only. ``edge_*`` are
    date-matched edges vs the same-day all-rows average (as ``solo_table``'s
    ``edge_vs_all_days``); ``mean_ab`` is the date-weighted mean of the metric
    on AB rows. Counts are rows with a known outcome.

    - ``jaccard`` = |A and B| / |A or B| over the split rows (NaN if neither fires);
    - ``cond_a_given_b`` = edge_ab - edge_b_only (what A adds when B fires);
    - ``lift`` = edge_ab - max(edge_a_only, edge_b_only); ``t_lift``/``p_lift``
      from ``stats.diff_test(AB, best only-group)``, conservative; NaN when
      either only-group's edge is NaN;
    - ``q_ab``/``q_lift``: Benjamini-Hochberg across this table's rows, separately;
    - ``insufficient``: n_ab < MIN_N, n_dates_ab < max(MIN_N, 5*horizon), or
      either only-group is empty or has no edge p-value (too few dates).
    """
    ctx = build_context(panel, horizon, metric, split, signal_info)
    names = list(ctx.info)
    out = []
    for i, a in enumerate(names):
        fam_a, dir_a = ctx.info[a]
        fa = ctx.fires[a]
        for b in names[i + 1:]:
            fam_b, dir_b = ctx.info[b]
            if dir_b != dir_a:
                continue
            fb = ctx.fires[b]
            both = fa & fb
            n_union = int((fa | fb).sum())
            jac = int(both.sum()) / n_union if n_union else NAN
            ab = group_stats(ctx, both, dir_a)
            ao = group_stats(ctx, fa & ~fb, dir_a)
            bo = group_stats(ctx, fb & ~fa, dir_a)
            ea, eb = ao.edge.mean, bo.edge.mean
            if np.isnan(ea) or np.isnan(eb):
                lift_t = TestResult(NAN, NAN, NAN, NAN, 0, 0)
            else:
                best = ao if ea >= eb else bo
                lift_t = diff_test(ab.edge, best.edge, horizon)
            insufficient = (not combo_sufficient(ab.n, ab.n_dates, horizon)
                            or ao.n == 0 or bo.n == 0
                            or np.isnan(ao.edge.p) or np.isnan(bo.edge.p))
            out.append({
                "a": a, "b": b, "family_a": fam_a, "family_b": fam_b, "direction": dir_a,
                "jaccard": jac, "n_ab": ab.n, "n_dates_ab": ab.n_dates,
                "n_a_only": ao.n, "n_b_only": bo.n,
                "mean_ab": ab.mean, "edge_ab": ab.edge.mean, "t_ab": ab.edge.t, "p_ab": ab.edge.p,
                "q_ab": NAN, "edge_a_only": ea, "edge_b_only": eb,
                "cond_a_given_b": ab.edge.mean - eb, "cond_b_given_a": ab.edge.mean - ea,
                "lift": lift_t.mean,
                "t_lift": lift_t.t, "p_lift": lift_t.p, "q_lift": NAN,
                "insufficient": bool(insufficient), "verdict": "",
            })
    t = pd.DataFrame(out, columns=PAIR_COLS)
    t["q_ab"] = bh_qvalues(t["p_ab"].to_numpy(dtype="float64"))
    t["q_lift"] = bh_qvalues(t["p_lift"].to_numpy(dtype="float64"))
    for c in ("n_ab", "n_dates_ab", "n_a_only", "n_b_only"):
        t[c] = t[c].astype(np.int64)
    t["insufficient"] = t["insufficient"].astype(bool)
    return add_pair_verdicts(t, horizon)
