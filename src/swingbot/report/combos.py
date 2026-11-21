"""Render the combo report: pairs, triples, regime slices and redundancy (static HTML + CSVs)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from swingbot import config
from swingbot.analysis.pairs import pair_regime_table, pair_table, triple_table
from swingbot.analysis.redundancy import jaccard_matrix, redundancy_groups
from swingbot.analysis.stats import min_dates_for_test
from swingbot.analysis.views import PRIMARY, SUSPECT, universe_views
from swingbot.report.build import (CAVEAT, CDN_URL, SHORT_NOTE, SUSPECT_LABEL, SURVIVORSHIP, _fmt,
                                   stock_mean_text)
from swingbot.signals import SIGNAL_INFO

TOP_PAIRS = 30
COMBO_CAVEAT = (
    "Combo results are discovery split only. Lift is measured against the better single-only edge: the combined "
    "group's edge minus the larger edge of A-only and B-only (rows where the other signal does not fire). "
    "q-values correct for multiple testing within each table only. The pair table has q (edge) and q (lift); "
    "the triple table has q (lift) only; the regime table has q (edge) only. "
    "Each q column is corrected separately. "
    "A synergy here is a hypothesis to be confirmed on the validation split, not a result. "
    "The stock view is suspect: survivorship-biased, so read it as a control, not a finding.")
PAIR_COLS_SHOWN = [
    ("a", "A"), ("b", "B"), ("direction", "Dir"), ("verdict", "Verdict"), ("n_ab", "n (A&B)"),
    ("n_dates_ab", "Dates"), ("mean_ab", "Mean excess A&B (bps)"),
    ("mean_ret_ab", "Mean return A&B, absolute (bps)"), ("edge_ab", "Edge A&B (bps)"),
    ("q_ab", "q edge"), ("edge_a_only", "Edge A only (bps)"), ("edge_b_only", "Edge B only (bps)"),
    ("lift", "Lift (bps)"), ("q_lift", "q lift"), ("jaccard", "Jaccard"),
]
TRIPLE_COLS_SHOWN = [
    ("a", "A"), ("b", "B"), ("c", "C"), ("direction", "Dir"), ("verdict", "Verdict"),
    ("n_abc", "n (A&B&C)"), ("n_dates_abc", "Dates"), ("mean_abc", "Mean excess (bps)"),
    ("mean_ret_abc", "Mean return, absolute (bps)"),
    ("edge_abc", "Edge (bps)"), ("best_pair", "Best pair"), ("edge_best_pair_only", "Best pair only edge (bps)"),
    ("lift", "Lift (bps)"), ("q_lift", "q lift"),
]
REGIME_COLS_SHOWN = [
    ("a", "A"), ("b", "B"), ("direction", "Dir"), ("regime", "Trend regime"), ("n_ab", "n (A&B)"),
    ("n_dates_ab", "Dates"), ("mean_ab", "Mean excess (bps)"),
    ("mean_ret_ab", "Mean return, absolute (bps)"), ("edge_ab", "Edge (bps)"),
    ("t_ab", "t"), ("q_ab", "q edge (BH, this table)"),
]
_TEMPLATES = Path(__file__).parent / "templates"


def select_pairs(pairs: pd.DataFrame, top: int = TOP_PAIRS) -> pd.DataFrame:
    """Every synergy and conflict row, then the ``top`` other non-insufficient rows by edge_ab (desc)."""
    keep = pairs["verdict"].isin(["synergy", "conflict"])
    rest = pairs[~keep & ~pairs["insufficient"].astype(bool)]
    rest = rest.sort_values("edge_ab", ascending=False, na_position="last", kind="stable").head(top)
    order = {"synergy": 0, "conflict": 1}
    out = pd.concat([pairs[keep], rest])
    out = out.assign(_r=out["verdict"].map(order).fillna(2))
    return out.sort_values(["_r", "edge_ab"], ascending=[True, False], na_position="last",
                           kind="stable").drop(columns="_r")


def jaccard_for_report(panel: pd.DataFrame, split: str | None = "discovery") -> pd.DataFrame:
    """``jaccard_matrix`` with a NaN diagonal for signals that never fire (so they do not look like overlap)."""
    m = jaccard_matrix(panel, split)
    sel = np.ones(len(panel), dtype=bool) if split is None else (panel["split"] == split).to_numpy(dtype=bool)
    for s in m.index:
        if not panel[s].to_numpy(dtype=bool)[sel].any():
            m.loc[s, s] = np.nan
    return m


def _rows(t: pd.DataFrame, cols: list[tuple[str, str]]) -> list[dict]:
    return [{"insufficient": bool(r["insufficient"]), "cells": [_fmt(c, r[c]) for c, _ in cols]}
            for r in t.to_dict("records")]


def _jaccard_html(m: pd.DataFrame, title: str) -> str:
    z = [[None if np.isnan(x) else float(x) for x in row] for row in m.to_numpy(dtype=float)]
    fig = go.Figure(go.Heatmap(z=z, x=list(m.columns), y=list(m.index), zmin=0, zmax=1, colorscale="Blues",
                               hoverongaps=False, colorbar={"title": "Jaccard"},
                               hovertemplate="%{y} & %{x}<br>Jaccard=%{z:.2f}<extra></extra>"))
    fig.update_layout(title=title, height=max(450, 18 * len(m) + 200),
                      margin={"l": 170, "r": 20, "t": 50, "b": 150},
                      yaxis={"autorange": "reversed"}, xaxis={"tickangle": -50})
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def build_combo_report(panels: dict[str, pd.DataFrame], out_path: Path, horizon: int = 10,
                       as_of: str | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = as_of or dt.date.today().isoformat()

    def save(name: str, view: str, df: pd.DataFrame) -> None:
        df.to_csv(out_path.parent / f"combo_{name}_{view}_{stamp}.csv", index=False)

    views, stock_mean = [], "not computed (no stock panel)"
    for view, panel in universe_views(panels).items():
        if view == "stock":
            stock_mean = stock_mean_text(panel, horizon)
        pairs = pair_table(panel, horizon=horizon)
        triples = triple_table(panel, pairs, horizon=horizon)
        regimes = pair_regime_table(panel, pairs, horizon=horizon)
        jac = jaccard_for_report(panel)
        for name, df in (("pairs", pairs), ("triples", triples), ("pair_regimes", regimes)):
            save(name, view, df)
        jac.rename_axis("signal").reset_index().to_csv(
            out_path.parent / f"combo_jaccard_{view}_{stamp}.csv", index=False)
        shown = select_pairs(pairs)
        views.append({
            "view": view, "suspect": view in SUSPECT, "primary": view in PRIMARY,
            "pair_counts": pairs["verdict"].value_counts().to_dict(),
            "triple_counts": triples["verdict"].value_counts().to_dict(),
            "n_pairs": len(pairs), "n_pairs_shown": len(shown), "n_triples": len(triples),
            "pair_rows": _rows(shown, PAIR_COLS_SHOWN), "triple_rows": _rows(triples, TRIPLE_COLS_SHOWN),
            "regime_rows": _rows(regimes, REGIME_COLS_SHOWN),
            "heat": _jaccard_html(jac, f"Jaccard overlap of firing rows: {view}"),
            "groups": redundancy_groups(jac),
        })

    env = Environment(loader=FileSystemLoader(_TEMPLATES), autoescape=select_autoescape(["j2", "html"]))
    page = env.get_template("combos.html.j2").render(
        caveat=CAVEAT, survivorship=SURVIVORSHIP.format(stock_mean=stock_mean), combo_caveat=COMBO_CAVEAT,
        short_note=SHORT_NOTE, redundant_jaccard=config.REDUNDANT_JACCARD,
        slice_rows=config.MIN_N_SLICE, slice_dates=max(config.MIN_N_SLICE, min_dates_for_test(horizon)),
        cdn=CDN_URL, horizon=horizon, stamp=stamp, views=views, suspect_label=SUSPECT_LABEL,
        pair_headers=[h for _, h in PAIR_COLS_SHOWN], triple_headers=[h for _, h in TRIPLE_COLS_SHOWN],
        regime_headers=[h for _, h in REGIME_COLS_SHOWN], top_pairs=TOP_PAIRS,
        verdicts=["synergy", "conflict", "redundant", "not significant", "insufficient"])
    out_path.write_text(page, encoding="utf-8")
    return out_path
