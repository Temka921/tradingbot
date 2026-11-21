"""Render the ML attribution report: out-of-sample LightGBM scores, SHAP importance and pair interactions."""
from __future__ import annotations

import datetime as dt
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from swingbot import config
from swingbot.analysis.attribution import (OOS_COLS, feature_matrix, fit_discovery_score_validation, oos_metrics,
                                           shap_pair_interactions, shap_summary, walk_forward)
from swingbot.analysis.pairs import pair_table
from swingbot.analysis.solo import solo_table
from swingbot.analysis.views import PRIMARY, SUSPECT, universe_views
from swingbot.report.build import CDN_URL, SHORT_NOTE, SUSPECT_LABEL, SURVIVORSHIP, _bps, add_verdicts, stock_mean_text

ML_CAVEAT = (
    "SHAP explains the model, not the market; a feature the model leans on is not proof of an edge. "
    "The validation row is the only out-of-sample score outside discovery: the walk-forward folds are all "
    "inside the discovery split, and the validation model is fitted once on all discovery rows. "
    "Hyperparameters are fixed (config.ML_PARAMS) and not tuned. "
    "Interaction magnitudes are the raw off-diagonal SHAP interaction values, which are half of the full "
    "pairwise effect. SHAP is computed on a random row sample. "
    "The stock view is suspect: survivorship-biased, so read it as a control, not a finding.")
OOS_SHOWN = [("fold", "Fold"), ("n", "n"), ("n_dates", "Dates"), ("ic_mean", "Rank IC"), ("ic_t", "IC t"),
             ("ic_p", "IC p"), ("spread_mean", "Top-decile spread (bps)"), ("spread_t", "Spread t"),
             ("spread_p", "Spread p"), ("top_mean", "Top-decile mean excess (bps)"),
             ("top_ret_n", "Top-decile rows"), ("insufficient", "Insufficient")]
SHAP_SHOWN = [("rank", "Rank"), ("feature", "Feature"), ("kind", "Kind"),
              ("mean_abs_shap", "Mean |SHAP| (bps)"), ("mean_shap_when_on", "Mean SHAP when on (bps)"),
              ("n_on", "n on (sample)")]
INTER_SHOWN = [("a", "A"), ("b", "B"), ("mean_abs_interaction", "Mean |interaction| (bps, half effect)"),
               ("mean_interaction_when_both", "Mean interaction when both (bps, half effect)"),
               ("n_both", "n both (sample)"), ("phase4_verdict", "Phase 4 verdict"),
               ("phase4_lift", "Phase 4 lift (bps)")]
_BPS = {"spread_mean", "top_mean", "mean_abs_shap", "mean_shap_when_on", "mean_abs_interaction",
        "mean_interaction_when_both", "phase4_lift"}
_INT = {"n", "n_dates", "top_ret_n", "n_on", "n_both", "rank"}
_P3 = {"ic_p", "spread_p"}
_TEMPLATES = Path(__file__).parent / "templates"


def _cell(col: str, v) -> str:
    if col in ("fold", "feature", "kind", "a", "b", "phase4_verdict"):
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    if col == "insufficient":
        return "yes" if bool(v) else ""
    if pd.isna(v):
        return ""
    if col in _INT:
        return str(int(v))
    if col in _BPS:
        return _bps(v)
    if col == "ic_mean":
        return f"{v:.4f}"
    if col in _P3:
        return f"{v:.3f}"
    return f"{v:.2f}"


def _rows(t: pd.DataFrame, cols: list[tuple[str, str]]) -> list[dict]:
    flag = "insufficient" in t.columns
    return [{"insufficient": bool(r["insufficient"]) if flag else False,
             "cells": [_cell(c, r[c]) for c, _ in cols]} for r in t.to_dict("records")]


def _shap_chart(shap: pd.DataFrame, view: str, top: int = 20) -> str:
    d = shap.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=(d["mean_abs_shap"] * 1e4).tolist(), y=d["feature"].tolist(), orientation="h"))
    fig.update_layout(title=f"Mean |SHAP| (bps), top {len(d)}: {view}", height=max(300, 22 * len(d) + 120),
                      margin={"l": 200, "r": 20, "t": 50, "b": 40}, xaxis={"title": "bps"})
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def _best_solo(panel: pd.DataFrame, horizon: int) -> dict | None:
    t = add_verdicts(solo_table(panel, horizon=horizon))
    c = t[t["verdict"] == "candidate"].sort_values("edge_vs_all_days", ascending=False, kind="stable")
    if c.empty:
        return None
    r = c.iloc[0]
    return {"signal": str(r["signal"]), "edge_bps": _bps(r["edge_vs_all_days"])}


def _skip_text(skipped: list[tuple]) -> list[str]:
    return [f"{'all' if y is None else y}: {reason}" for y, reason in skipped]


def build_ml_report(panels: dict[str, pd.DataFrame], out_path: Path, horizon: int = 10,
                    as_of: str | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = as_of or dt.date.today().isoformat()

    def save(name: str, view: str, df: pd.DataFrame) -> None:
        df.to_csv(out_path.parent / f"ml_{name}_{view}_{stamp}.csv", index=False)

    all_views = universe_views(panels)
    stock_mean = (stock_mean_text(all_views["stock"], horizon) if "stock" in all_views
                  else "not computed (no stock panel)")
    views = []
    for view in list(all_views):
        panel = all_views.pop(view)                      # one view at a time; freed before the next
        wf = walk_forward(panel, horizon=horizon)
        skipped = list(wf.attrs.get("skipped", []))
        oos = oos_metrics(wf, horizon)
        model, val_pred = fit_discovery_score_validation(panel, horizon=horizon)
        skipped_val = list(val_pred.attrs.get("skipped", []))
        val_rows = oos_metrics(val_pred, horizon) if model is not None else oos.iloc[0:0]
        oos_all = pd.concat([oos[oos["fold"].astype(str) != "validation"], val_rows], ignore_index=True)
        save("oos", view, oos_all)
        all_row = oos_all[oos_all["fold"].astype(str) == "all"]
        ml_spread = _bps(all_row["spread_mean"].iloc[0]) if len(all_row) else ""
        best = _best_solo(panel, horizon)

        shap = inter = None
        if model is not None:
            disc = panel[(panel["split"] == "discovery").to_numpy()]
            X = feature_matrix(disc)
            shap = shap_summary(model, X)
            inter = shap_pair_interactions(model, X, top=30)
            del X, disc
            p4 = pair_table(panel, horizon=horizon)[["a", "b", "verdict", "lift"]].rename(
                columns={"verdict": "phase4_verdict", "lift": "phase4_lift"})
            inter = inter.merge(p4, on=["a", "b"], how="left")
            save("shap", view, shap)
            save("interactions", view, inter)
            del p4
        views.append({
            "view": view, "suspect": view in SUSPECT, "primary": view in PRIMARY,
            "oos_rows": _rows(oos_all[oos_all["fold"].astype(str) != "validation"], OOS_SHOWN),
            "val_rows": _rows(val_rows, OOS_SHOWN), "skipped": _skip_text(skipped),
            "skipped_val": _skip_text(skipped_val), "ml_spread": ml_spread, "best": best,
            "shap_chart": _shap_chart(shap, view) if shap is not None and len(shap) else "",
            "shap_rows": _rows(shap, SHAP_SHOWN) if shap is not None else [],
            "inter_rows": _rows(inter, INTER_SHOWN) if inter is not None else [],
            "has_model": model is not None,
        })
        del panel, wf, oos, oos_all, val_pred, model, shap, inter
        gc.collect()

    env = Environment(loader=FileSystemLoader(_TEMPLATES), autoescape=select_autoescape(["j2", "html"]))
    page = env.get_template("ml.html.j2").render(
        survivorship=SURVIVORSHIP.format(stock_mean=stock_mean), ml_caveat=ML_CAVEAT, short_note=SHORT_NOTE,
        cdn=CDN_URL, horizon=horizon, stamp=stamp, views=views, suspect_label=SUSPECT_LABEL,
        oos_headers=[h for _, h in OOS_SHOWN], shap_headers=[h for _, h in SHAP_SHOWN],
        inter_headers=[h for _, h in INTER_SHOWN])
    out_path.write_text(page, encoding="utf-8")
    return out_path
