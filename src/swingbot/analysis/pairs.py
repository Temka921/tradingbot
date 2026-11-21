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
TRIPLE_COLS = [
    "a", "b", "c", "direction", "n_abc", "n_dates_abc", "mean_abc", "edge_abc",
    "best_pair", "edge_best_pair_only", "lift", "t_lift", "p_lift", "q_lift",
    "insufficient", "verdict",
]
REGIME_COLS = [
    "a", "b", "direction", "regime", "n_ab", "n_dates_ab", "mean_ab", "edge_ab",
    "t_ab", "p_ab", "q_ab", "insufficient",
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

    Frames without ``jaccard`` / ``n_a_only`` / ``n_b_only`` (triples) are
    accepted: both "redundant" rules then never apply. Triples pass their
    combo columns as ``n_ab``, ``n_dates_ab`` and ``mean_ab``.
    """
    t = t.copy()
    q = t["q_lift"].astype("float64")
    lift = t["lift"].astype("float64")
    sig = q < Q_SIGNIFICANT
    insufficient = t["insufficient"].astype(bool)
    if "n_a_only" in t and "n_b_only" in t:
        only_empty = (t["n_a_only"] == 0) | (t["n_b_only"] == 0)
    else:
        only_empty = pd.Series(False, index=t.index)
    jac = t["jaccard"].astype("float64") if "jaccard" in t else pd.Series(NAN, index=t.index)
    conds = [
        insufficient & only_empty & combo_sufficient(t["n_ab"], t["n_dates_ab"], horizon),
        insufficient,
        sig & (lift > 0) & (t["mean_ab"].astype("float64") > 0),
        sig & (lift < 0),
        ~sig & (jac >= config.REDUNDANT_JACCARD),
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


def _triple_candidates(pairs: pd.DataFrame, info: dict[str, tuple[str, str]]
                       ) -> list[tuple[str, str, str, str]]:
    """Same-direction triples (registry order) whose three pairs all have lift > 0 and are sufficient.

    Ranked by the smallest constituent lift, descending (ties keep registry
    order), and cut to ``config.MAX_TRIPLES``.
    """
    ok = pairs[(pairs["lift"].astype("float64") > 0) & ~pairs["insufficient"].astype(bool)]
    lift = {frozenset((a, b)): float(l) for a, b, l in zip(ok["a"], ok["b"], ok["lift"])}
    names = [s for s in info if any(s in k for k in lift)]
    cands = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            lab = lift.get(frozenset((a, b)))
            if lab is None or info[a][1] != info[b][1]:
                continue
            for c in names[j + 1:]:
                if info[c][1] != info[a][1]:
                    continue
                lac, lbc = lift.get(frozenset((a, c))), lift.get(frozenset((b, c)))
                if lac is None or lbc is None:
                    continue
                cands.append((min(lab, lac, lbc), a, b, c, info[a][1]))
    cands.sort(key=lambda x: -x[0])                    # stable: ties stay in registry order
    return [(a, b, c, d) for _, a, b, c, d in cands[:max(int(config.MAX_TRIPLES), 0)]]


def triple_table(panel: pd.DataFrame, pairs: pd.DataFrame, horizon: int = 10,
                 metric: str = "excess", split: str | None = "discovery",
                 signal_info: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Does a third signal add to a pair? One row per candidate triple (see ``_triple_candidates``).

    Groups: ABC = all three fire; the three pair-only groups A∧B∧¬C, A∧C∧¬B,
    B∧C∧¬A. ``edge_best_pair_only`` is the largest of their date-matched edges
    and ``best_pair`` names it ("a+b"); ``lift = edge_abc - edge_best_pair_only``,
    tested with ``stats.diff_test`` (conservative). As in ``pair_table``, if any
    pair-only group is empty or has no edge p-value, the best pair-only edge
    and the lift are NaN.

    ``insufficient``: ABC fails ``combo_sufficient`` or the best pair-only edge
    is NaN. ``q_lift`` is BH across this table. Verdicts via
    ``add_pair_verdicts``; triples have no Jaccard, so "redundant" never applies.
    """
    info = SIGNAL_INFO if signal_info is None else signal_info
    cands = _triple_candidates(pairs, info)
    if not cands:
        return pd.DataFrame(columns=TRIPLE_COLS)
    ctx = build_context(panel, horizon, metric, split, info)
    out = []
    for a, b, c, d in cands:
        fa, fb, fc = ctx.fires[a], ctx.fires[b], ctx.fires[c]
        abc = group_stats(ctx, fa & fb & fc, d)
        only = {f"{a}+{b}": group_stats(ctx, fa & fb & ~fc, d),
                f"{a}+{c}": group_stats(ctx, fa & fc & ~fb, d),
                f"{b}+{c}": group_stats(ctx, fb & fc & ~fa, d)}
        valid = all(g.n > 0 and not np.isnan(g.edge.mean) and not np.isnan(g.edge.p)
                    for g in only.values())
        if valid:
            best_pair = max(only, key=lambda k: only[k].edge.mean)   # first max on ties
            best = only[best_pair]
            lift_t = diff_test(abc.edge, best.edge, horizon)
            e_best = best.edge.mean
        else:
            best_pair, e_best = "", NAN
            lift_t = TestResult(NAN, NAN, NAN, NAN, 0, 0)
        insufficient = not combo_sufficient(abc.n, abc.n_dates, horizon) or np.isnan(e_best)
        out.append({
            "a": a, "b": b, "c": c, "direction": d, "n_abc": abc.n, "n_dates_abc": abc.n_dates,
            "mean_abc": abc.mean, "edge_abc": abc.edge.mean, "best_pair": best_pair,
            "edge_best_pair_only": e_best, "lift": lift_t.mean, "t_lift": lift_t.t,
            "p_lift": lift_t.p, "q_lift": NAN, "insufficient": bool(insufficient), "verdict": "",
        })
    t = pd.DataFrame(out, columns=TRIPLE_COLS)
    t["q_lift"] = bh_qvalues(t["p_lift"].to_numpy(dtype="float64"))
    for col in ("n_abc", "n_dates_abc"):
        t[col] = t[col].astype(np.int64)
    t["insufficient"] = t["insufficient"].astype(bool)
    v = add_pair_verdicts(t.rename(columns={"n_abc": "n_ab", "n_dates_abc": "n_dates_ab",
                                            "mean_abc": "mean_ab"}), horizon)
    t["verdict"] = v["verdict"].to_numpy()
    return t[TRIPLE_COLS]


def pair_regime_table(panel: pd.DataFrame, pairs: pd.DataFrame, horizon: int = 10,
                      regime_col: str = "trend_regime", split: str | None = "discovery",
                      signal_info: dict[str, tuple[str, str]] | None = None,
                      metric: str = "excess") -> pd.DataFrame:
    """Edge of every synergy pair's AB group inside each regime of ``regime_col``.

    Each regime gets its own context (``rows_mask = regime == value``), so the
    baseline is the same-day average of rows in the same regime. Regime
    "unknown" (and missing labels) are skipped; regimes are listed in sorted
    order. ``q_ab`` is BH across this table. ``insufficient``: n_ab < MIN_N_SLICE
    or n_dates_ab < max(MIN_N_SLICE, 5*horizon).
    """
    info = SIGNAL_INFO if signal_info is None else signal_info
    syn = pairs[pairs["verdict"] == "synergy"]
    if syn.empty:
        return pd.DataFrame(columns=REGIME_COLS)
    lab = panel[regime_col].astype(object).to_numpy()
    in_split = np.ones(len(panel), dtype=bool) if split is None else \
        (panel["split"] == split).to_numpy(dtype=bool)
    regimes = sorted({r for r in lab[in_split] if isinstance(r, str) and r != "unknown"})
    min_dates = max(config.MIN_N_SLICE, min_dates_for_test(horizon))
    out = []
    for reg in regimes:
        ctx = build_context(panel, horizon, metric, split, info, rows_mask=lab == reg)
        for a, b, d in zip(syn["a"], syn["b"], syn["direction"]):
            g = group_stats(ctx, ctx.fires[a] & ctx.fires[b], d)
            out.append({
                "a": a, "b": b, "direction": d, "regime": reg, "n_ab": g.n,
                "n_dates_ab": g.n_dates, "mean_ab": g.mean, "edge_ab": g.edge.mean,
                "t_ab": g.edge.t, "p_ab": g.edge.p, "q_ab": NAN,
                "insufficient": bool(g.n < config.MIN_N_SLICE or g.n_dates < min_dates),
            })
    t = pd.DataFrame(out, columns=REGIME_COLS)
    t["q_ab"] = bh_qvalues(t["p_ab"].to_numpy(dtype="float64"))
    for col in ("n_ab", "n_dates_ab"):
        t[col] = t[col].astype(np.int64)
    t["insufficient"] = t["insufficient"].astype(bool)
    return t
