"""Render the solo-signal research report (static HTML + CSVs)."""
from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from swingbot import config
from swingbot.analysis.solo import solo_table
from swingbot.signals import SIGNAL_INFO

CDN_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"
CAVEAT = ("Discovery split only. Survivorship bias: delisted stocks and some delisted coins are missing. "
          "Costs: stocks 0.10%, crypto 0.30% per round trip. "
          "Results with q ≥ 0.10 or flagged insufficient are not evidence of an edge.")
CAVEAT_EXTRA = ("p-values test the edge versus the same-day average of all assets in the same market "
                "(signal included), using date-clustered HAC errors with conservative effective degrees "
                "of freedom; q-values correct for multiple testing within each table only.")
SEGMENT_KEY = {"stock": "sector", "crypto": "cap_bucket"}
_TEMPLATES = Path(__file__).parent / "templates"

STAT_FIELDS = {"mean_all", "edge_vs_all_days", "t", "p", "q", "hit_rate", "profit_factor", "occ_per_year"}
LEADER_COLS = [
    ("signal", "Signal"), ("family", "Family"), ("direction", "Dir"), ("n_all", "n"),
    ("n_solo", "n solo"), ("n_dates", "Dates"), ("mean_all", "Mean excess (date-weighted, bps)"),
    ("edge_vs_all_days", "Edge vs same-day avg (bps)"), ("t", "t"), ("p", "p"),
    ("q", "q (BH, this table)"), ("hit_rate", "Hit rate"), ("profit_factor", "Profit factor"),
    ("occ_per_year", "Occurrences / yr"),
]


def sort_leaderboard(t: pd.DataFrame) -> pd.DataFrame:
    """Insufficient rows last; then q ascending (NaN last); then edge descending."""
    return t.sort_values(["insufficient", "q", "edge_vs_all_days"], ascending=[True, True, False],
                         na_position="last", kind="stable").reset_index(drop=True)


def _bps(v) -> str:
    return "" if pd.isna(v) else f"{v * 1e4:.1f}"


def _fmt(col: str, v) -> str:
    if col in ("signal", "family", "direction"):
        return str(v)
    if col in ("n_all", "n_solo", "n_dates"):
        return str(int(v))
    if pd.isna(v):
        return ""
    if col in ("mean_all", "edge_vs_all_days", "mean_solo", "median_all"):
        return _bps(v)
    if col == "t":
        return f"{v:.2f}"
    if col in ("p", "q"):
        return f"{v:.3f}"
    if col == "hit_rate":
        return f"{v * 100:.1f}%"
    if col == "profit_factor":
        return "inf" if np.isinf(v) else f"{v:.2f}"
    if col == "occ_per_year":
        return f"{v:.1f}"
    return str(v)


def _rows(t: pd.DataFrame, cols: list[tuple[str, str]], blank_insufficient: bool = False) -> list[dict]:
    keep_always = {"signal", "family", "direction", "n_all", "n_solo", "n_dates"}
    out = []
    for r in t.to_dict("records"):
        insuf = bool(r["insufficient"])
        cells = []
        for c, _ in cols:
            blank = blank_insufficient and insuf and c not in keep_always and c in STAT_FIELDS
            cells.append("" if blank else _fmt(c, r[c]))
        out.append({"insufficient": insuf, "cells": cells})
    return out


def heatmap_matrix(table: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    """signal x (market | regime) matrix of edge_vs_all_days in bps; insufficient cells are NaN."""
    t = table.copy()
    t["edge_bps"] = np.where(t["insufficient"], np.nan, t["edge_vs_all_days"] * 1e4)
    t["col"] = t["market"].astype(str) + " | " + t[regime_col].astype(str)
    m = t.pivot_table(index="signal", columns="col", values="edge_bps", aggfunc="first", dropna=False)
    return m.reindex(list(SIGNAL_INFO)).sort_index(axis=1)


def _heatmap_html(tables: list[pd.DataFrame], regime_col: str, title: str) -> str:
    full = pd.concat(tables, ignore_index=True)
    full["col"] = full["market"].astype(str) + " | " + full[regime_col].astype(str)
    z = heatmap_matrix(full, regime_col)
    idx = full.set_index(["signal", "col"])
    text = []
    for s in z.index:
        row = []
        for c in z.columns:
            try:
                r = idx.loc[(s, c)]
                row.append(f"{s}<br>{c}<br>n={int(r.n_all)}, dates={int(r.n_dates)}<br>"
                           f"mean excess={_bps(r.mean_all)} bps<br>t={_fmt('t', r.t)}, q={_fmt('q', r.q)}"
                           + ("<br>insufficient" if r.insufficient else ""))
            except KeyError:
                row.append(f"{s}<br>{c}<br>no rows")
        text.append(row)
    fig = go.Figure(go.Heatmap(
        z=z.to_numpy(dtype=float), x=list(z.columns), y=list(z.index), text=text,
        hovertemplate="%{text}<extra></extra>", colorscale="RdBu", zmid=0,
        colorbar={"title": "Edge (bps)"}))
    fig.update_layout(title=title, height=max(400, 22 * len(z) + 160), margin={"l": 170, "r": 20, "t": 50, "b": 80},
                      yaxis={"autorange": "reversed"}, xaxis={"tickangle": -30})
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def _quality_count(quality: pd.DataFrame | None, symbols: set[str]) -> int | None:
    if quality is None or "symbol" not in quality:
        return None
    return int(quality["symbol"].isin(symbols).sum())


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
    for market, panel in panels.items():
        disc = panel[panel["split"] == "discovery"]
        syms = set(panel["asset"].astype(str).unique())
        qn = _quality_count(quality, syms)
        coverage.append({
            "market": market, "assets": len(syms), "rows": len(panel),
            "start": str(pd.to_datetime(panel["date"]).min().date()),
            "end": str(pd.to_datetime(panel["date"]).max().date()),
            "disc_rows": len(disc), "issues": "n/a" if qn is None else qn})

        lb = sort_leaderboard(solo_table(panel, horizon=horizon))
        save(f"leaderboard_{market}", lb)
        leaders.append({"market": market, "rows": _rows(lb, LEADER_COLS),
                        "n_sufficient": int((~lb["insufficient"]).sum()),
                        "n_sig": int((lb["q"] < 0.10).sum())})

        tr = solo_table(panel, horizon=horizon, group_by=["market", "trend_regime"])
        vo = solo_table(panel, horizon=horizon, group_by=["market", "vol_regime"])
        save(f"heatmap_trend_{market}", tr)
        save(f"heatmap_vol_{market}", vo)
        tr_tabs.append(tr)
        vol_tabs.append(vo)

        key = SEGMENT_KEY.get(market)
        if key and key in panel.columns:
            seg = sort_leaderboard(solo_table(panel, horizon=horizon, group_by=["market", key]))
            save(f"segment_{market}", seg)
            cols = [("signal", "Signal"), (key, key.replace("_", " ").title())] + LEADER_COLS[2:]
            segments.append({"market": market, "rows": _rows(seg, cols, blank_insufficient=True),
                             "headers": [h for _, h in cols]})

    env = Environment(loader=FileSystemLoader(_TEMPLATES), autoescape=select_autoescape(["j2", "html"]))
    page = env.get_template("report.html.j2").render(
        caveat=CAVEAT, caveat_extra=CAVEAT_EXTRA, cdn=CDN_URL, horizon=horizon, stamp=stamp,
        coverage=coverage, leaders=leaders, headers=[h for _, h in LEADER_COLS],
        heat_trend=_heatmap_html(tr_tabs, "trend_regime", "Edge vs same-day avg (bps) by trend regime"),
        heat_vol=_heatmap_html(vol_tabs, "vol_regime", "Edge vs same-day avg (bps) by volatility regime"),
        segments=segments)
    out_path.write_text(page, encoding="utf-8")
    return out_path
