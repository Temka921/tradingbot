"""Render the solo-signal research report (static HTML + CSVs)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from swingbot import config
from swingbot.analysis.solo import clip_bounds, clip_values, solo_table
from swingbot.analysis.views import PRIMARY, SEGMENT_KEY, SUSPECT, universe_views
from swingbot.signals import SIGNAL_INFO

CDN_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"
CAVEAT = ("Discovery split only. Survivorship bias: single stocks are current index members only, so delisted stocks are missing. "
          "Costs: stocks 0.10%, crypto 0.30% per round trip. "
          "Results with q ≥ 0.10 or flagged insufficient are not evidence of an edge.")
CAVEAT_EXTRA = ("p-values test the edge versus the same-day average of all assets in the same view "
                "(signal included), using date-clustered HAC errors with conservative effective degrees "
                "of freedom; q-values correct for multiple testing within each table only.")
SUSPECT_LABEL = "Suspect: survivorship-biased (today's index members only)"
SURVIVORSHIP = (
    "Survivorship is large for single stocks, not a footnote. (a) The stock view is today's S&P 500/400 "
    "constituents, so historical returns are inflated and dip-buying signals are favoured: stocks that fell and "
    "kept falling were dropped from the indexes and are missing here. Stocks are therefore suspect. The all-rows "
    "mean excess over SPY for the stock view (ETFs excluded) in discovery is {stock_mean} (date-weighted, after "
    "clipping); that is mostly survivorship, not skill. (b) Crypto includes delisted Binance pairs and uses a "
    "point-in-time top-100 liquidity filter; delisted coins exit at their last close, which is an approximation "
    "(post-delisting value may be lower). (c) ETFs: currently listed funds only, so closed funds are missing "
    "(a smaller bias). ETFs and crypto are the primary results.")
DUP_NOTE = ("<code>bb_lower_close</code> and <code>zscore_below_2</code> are the same test: close below the "
            "20-day mean minus 2 standard deviations. They fire on the same rows and give identical statistics; "
            "count them once.")
_TEMPLATES = Path(__file__).parent / "templates"

STAT_FIELDS = {"mean_all", "mean_solo", "edge_vs_all_days", "t", "p", "q", "hit_rate", "profit_factor",
               "occ_per_year"}
LEADER_COLS = [
    ("signal", "Signal"), ("family", "Family"), ("direction", "Dir"), ("verdict", "Verdict"), ("n_all", "n"),
    ("n_solo", "n solo"), ("n_dates", "Dates"), ("mean_all", "Mean excess (date-weighted, bps)"),
    ("mean_solo", "Mean excess solo (date-weighted, bps)"),
    ("edge_vs_all_days", "Edge vs same-day avg (bps)"), ("t", "t"), ("p", "p"),
    ("q", "q (BH, this table)"), ("hit_rate", "Hit rate"), ("profit_factor", "Profit factor"),
    ("occ_per_year", "Occurrences / yr"),
]


SIG_Q = 0.10
VERDICTS = ["candidate", "significant negative edge (avoid)",
            "beats same-day avg but loses net of costs", "not significant", "insufficient"]


def add_verdicts(t: pd.DataFrame) -> pd.DataFrame:
    """Add a ``verdict`` column (after ``direction`` when present).

    candidate: sufficient, q < 0.10, edge > 0 AND mean excess net of costs > 0.
    Edge and its p/q are relative to the same-day average and cost-neutral, so a
    significant positive edge alone is not enough.
    """
    t = t.drop(columns="verdict", errors="ignore").copy()
    insuf = t["insufficient"].astype(bool).to_numpy()
    q = t["q"].to_numpy(dtype="float64")
    edge = t["edge_vs_all_days"].to_numpy(dtype="float64")
    mean = t["mean_all"].to_numpy(dtype="float64")
    sig = ~insuf & (q < SIG_Q)
    v = np.select([insuf, sig & (edge > 0) & (mean > 0), sig & (edge < 0), sig & (edge > 0) & (mean <= 0)],
                  [VERDICTS[4], VERDICTS[0], VERDICTS[1], VERDICTS[2]], default=VERDICTS[3])
    pos = t.columns.get_loc("direction") + 1 if "direction" in t.columns else len(t.columns)
    t.insert(pos, "verdict", v)
    return t


def verdict_counts(t: pd.DataFrame) -> dict[str, int]:
    vc = t["verdict"].value_counts()
    return {v: int(vc.get(v, 0)) for v in VERDICTS}


def sort_leaderboard(t: pd.DataFrame) -> pd.DataFrame:
    """Verdict order (candidate, avoid, loses net, not significant, insufficient);
    within each: q ascending (NaN last), then edge descending."""
    t = add_verdicts(t)
    t["_rank"] = t["verdict"].map({v: i for i, v in enumerate(VERDICTS)})
    return (t.sort_values(["_rank", "q", "edge_vs_all_days"], ascending=[True, True, False],
                          na_position="last", kind="stable")
            .drop(columns="_rank").reset_index(drop=True))


def _bps(v) -> str:
    return "" if pd.isna(v) else f"{v * 1e4:.1f}"


_TEXT_COLS = {"signal", "family", "direction", "verdict", "a", "b", "c", "family_a", "family_b", "regime",
              "best_pair"}
_INT_COLS = {"n_all", "n_solo", "n_dates", "n_ab", "n_dates_ab", "n_a_only", "n_b_only", "n_abc", "n_dates_abc"}
_BPS_COLS = {"mean_all", "edge_vs_all_days", "mean_solo", "median_all", "mean_ab", "edge_ab", "edge_a_only",
             "edge_b_only", "lift", "mean_abc", "edge_abc", "edge_best_pair_only", "cond_a_given_b",
             "cond_b_given_a"}
_P_COLS = {"p", "q", "p_ab", "q_ab", "p_lift", "q_lift"}


def _fmt(col: str, v) -> str:
    if col in _TEXT_COLS:
        return str(v)
    if col in _INT_COLS:
        return str(int(v))
    if pd.isna(v):
        return ""
    if col in _BPS_COLS:
        return _bps(v)
    if col in ("t", "t_ab", "t_lift", "jaccard"):
        return f"{v:.2f}"
    if col in _P_COLS:
        return f"{v:.3f}"
    if col == "hit_rate":
        return f"{v * 100:.1f}%"
    if col == "profit_factor":
        return "inf" if np.isinf(v) else f"{v:.2f}"
    if col == "occ_per_year":
        return f"{v:.1f}"
    return str(v)


def _rows(t: pd.DataFrame, cols: list[tuple[str, str]], blank_insufficient: bool = False) -> list[dict]:
    keep_always = {"signal", "family", "direction", "verdict", "n_all", "n_solo", "n_dates"}
    out = []
    for r in t.to_dict("records"):
        insuf = bool(r["insufficient"])
        cells = []
        for c, _ in cols:
            blank = blank_insufficient and insuf and c not in keep_always and c in STAT_FIELDS
            cells.append("" if blank else _fmt(c, r[c]))
        out.append({"insufficient": insuf, "cells": cells})
    return out


def heatmap_matrix(table: pd.DataFrame, regime_col: str, significant_only: bool = False) -> pd.DataFrame:
    """signal x (view | regime) matrix of edge_vs_all_days in bps; insufficient cells are NaN.

    With ``significant_only`` cells with q >= SIG_Q (or NaN q) are NaN too.
    """
    t = table.copy()
    blank = t["insufficient"].to_numpy(dtype=bool)
    if significant_only:
        blank = blank | ~(t["q"].to_numpy(dtype="float64") < SIG_Q)
    t["edge_bps"] = np.where(blank, np.nan, t["edge_vs_all_days"] * 1e4)
    t["col"] = t["view"].astype(str) + " | " + t[regime_col].astype(str)
    m = t.pivot_table(index="signal", columns="col", values="edge_bps", aggfunc="first", dropna=False)
    return m.reindex(list(SIGNAL_INFO)).sort_index(axis=1)


def _json_z(a: np.ndarray) -> list[list[float | None]]:
    """Plain nested lists (None for NaN) so the rendered figure JSON is readable."""
    return [[None if np.isnan(x) else float(x) for x in row] for row in a]


def _heatmap_html(tables: list[pd.DataFrame], regime_col: str, title: str) -> str:
    full = pd.concat(tables, ignore_index=True)
    full["col"] = full["view"].astype(str) + " | " + full[regime_col].astype(str)
    z = heatmap_matrix(full, regime_col)
    z_sig = heatmap_matrix(full, regime_col, significant_only=True).reindex(index=z.index, columns=z.columns)
    all_z = z.to_numpy(dtype=float)
    sig_z = z_sig.to_numpy(dtype=float)
    grey_z = np.where(~np.isnan(all_z) & np.isnan(sig_z), 0.0, np.nan)
    idx = full.set_index(["signal", "col"])
    text = []
    for s in z.index:
        row = []
        for c in z.columns:
            try:
                r = idx.loc[(s, c)]
                if r.insufficient:
                    row.append(f"{s}<br>{c}<br>insufficient data (n={int(r.n_all)}, dates={int(r.n_dates)})")
                else:
                    tag = "" if r.q < SIG_Q else "<br>not significant (q ≥ 0.10)"
                    row.append(f"{s}<br>{c}<br>n={int(r.n_all)}, dates={int(r.n_dates)}<br>"
                               f"edge={_bps(r.edge_vs_all_days)} bps, mean excess={_bps(r.mean_all)} bps<br>"
                               f"t={_fmt('t', r.t)}, q={_fmt('q', r.q)}{tag}")
            except KeyError:
                row.append(f"{s}<br>{c}<br>no rows")
        text.append(row)
    common = {"x": list(z.columns), "y": list(z.index), "text": text, "hoverongaps": False,
              "hovertemplate": "%{text}<extra></extra>"}
    fig = go.Figure([
        go.Heatmap(z=_json_z(grey_z), name="not significant", showscale=False,
                   colorscale=[[0, "#c9cdd3"], [1, "#c9cdd3"]], **common),
        go.Heatmap(z=_json_z(sig_z), name="significant", colorscale="RdBu", zmid=0,
                   colorbar={"title": "Edge (bps)"}, **common),
    ])
    fig.update_layout(title=title, height=max(400, 22 * len(z) + 160), margin={"l": 170, "r": 20, "t": 50, "b": 80},
                      yaxis={"autorange": "reversed"}, xaxis={"tickangle": -30})
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def _quality_count(quality: pd.DataFrame | None, symbols: set[str], market: str) -> int | None:
    """Distinct assets of this market with a non-empty quality issue (market filter where populated)."""
    if quality is None or "symbol" not in quality:
        return None
    q = quality
    if "market" in q:
        q = q[q["market"].isna() | (q["market"] == market)]
    if "issues" in q:
        q = q[q["issues"].fillna("").astype(str).str.strip() != ""]
    return int(q.loc[q["symbol"].isin(symbols), "symbol"].nunique())


def _pct_range(b: tuple[float, float]) -> str:
    lo, hi = b
    return "n/a" if np.isnan(lo) else f"{lo * 100:.2f}% to {hi * 100:.2f}%"


def universe_mean_excess(panel: pd.DataFrame, horizon: int) -> float:
    """Date-weighted mean of excess_long_h over every discovery row (clipped as in solo_table)."""
    disc = panel[(panel["split"] == "discovery").to_numpy()]
    v = disc[f"excess_long_{horizon}"].to_numpy(dtype="float64")
    v = clip_values(v, clip_bounds(v))
    ok = ~np.isnan(v)
    if not ok.any():
        return np.nan
    return float(pd.Series(v[ok]).groupby(disc["date"].to_numpy()[ok]).mean().mean())


def stock_mean_text(panel: pd.DataFrame, horizon: int) -> str:
    """Survivorship-text figure: the stock view's universe mean excess, in bps per horizon."""
    x = universe_mean_excess(panel, horizon)
    return "n/a" if np.isnan(x) else f"{x * 1e4:+.1f} bps per {horizon} days"


def build_report(panels: dict[str, pd.DataFrame], out_path: Path, horizon: int = 10,
                 quality: pd.DataFrame | None = None, as_of: str | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = as_of or dt.date.today().isoformat()
    if quality is None:
        qp = config.DATA_DIR / "quality_report.csv"
        quality = pd.read_csv(qp) if qp.exists() else None

    def save(name: str, df: pd.DataFrame) -> None:
        df.to_csv(out_path.parent / f"solo_{name}_{stamp}.csv", index=False)

    coverage, leaders, tr_tabs, vol_tabs, segments = [], [], [], [], []
    stock_mean = "not computed (no stock panel)"
    for view, panel in universe_views(panels).items():
        market = str(panel["market"].iloc[0])
        disc = panel[panel["split"] == "discovery"]
        syms = set(panel["asset"].astype(str).unique())
        qn = _quality_count(quality, syms, market)

        lb_raw = solo_table(panel, horizon=horizon)
        clip = lb_raw.attrs.get("clip_bounds", {})
        lb = sort_leaderboard(lb_raw)
        save(f"leaderboard_{view}", lb)
        leaders.append({"view": view, "suspect": view in SUSPECT, "primary": view in PRIMARY,
                        "rows": _rows(lb, LEADER_COLS), "counts": verdict_counts(lb)})
        if view == "stock":
            stock_mean = stock_mean_text(panel, horizon)
        coverage.append({
            "view": view, "assets": len(syms), "rows": len(panel),
            "start": str(pd.to_datetime(panel["date"]).min().date()),
            "end": str(pd.to_datetime(panel["date"]).max().date()),
            "disc_rows": len(disc), "issues": "n/a" if qn is None else qn,
            "clip": "; ".join(f"{k}: {_pct_range(b)}" for k, b in clip.items())})

        tr = solo_table(panel, horizon=horizon, group_by=["view", "trend_regime"])
        vo = solo_table(panel, horizon=horizon, group_by=["view", "vol_regime"])
        save(f"heatmap_trend_{view}", tr)
        save(f"heatmap_vol_{view}", vo)
        tr_tabs.append(tr)
        vol_tabs.append(vo)

        key = SEGMENT_KEY.get(view)
        if key and key in panel.columns:
            seg = sort_leaderboard(solo_table(panel, horizon=horizon, group_by=["view", key]))
            save(f"segment_{view}", seg)
            cols = [("signal", "Signal"), (key, key.replace("_", " ").title())] + LEADER_COLS[2:]
            segments.append({"view": view, "rows": _rows(seg, cols, blank_insufficient=True),
                             "headers": [h for _, h in cols]})

    env = Environment(loader=FileSystemLoader(_TEMPLATES), autoescape=select_autoescape(["j2", "html"]))
    page = env.get_template("report.html.j2").render(
        caveat=CAVEAT, survivorship=SURVIVORSHIP.format(stock_mean=stock_mean), caveat_extra=CAVEAT_EXTRA,
        cdn=CDN_URL, horizon=horizon, stamp=stamp, coverage=coverage, leaders=leaders,
        suspect_label=SUSPECT_LABEL, dup_note=DUP_NOTE, headers=[h for _, h in LEADER_COLS],
        heat_trend=_heatmap_html(tr_tabs, "trend_regime", "Edge vs same-day avg (bps) by trend regime"),
        heat_vol=_heatmap_html(vol_tabs, "vol_regime", "Edge vs same-day avg (bps) by volatility regime"),
        segments=segments)
    out_path.write_text(page, encoding="utf-8")
    return out_path
