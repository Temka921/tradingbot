"""Render the finalists report: validation gate, validation backtests, robustness and tail check."""
from __future__ import annotations

import datetime as dt
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from swingbot import config
from swingbot.analysis.solo import clip_bounds, clip_values, date_weighted_mean
from swingbot.analysis.validation import GATE_COLS, validation_gate
from swingbot.analysis.views import MARKET_OF_VIEW, PRIMARY, SUSPECT, universe_views
from swingbot.backtest.portfolio import ALL_RULE, metrics, rule_mask, simulate, yearly_returns
from swingbot.data import store
from swingbot.report.build import CDN_URL, SHORT_NOTE, SUSPECT_LABEL, SURVIVORSHIP, stock_mean_text

MAX_POSITIONS_GRID = (8, 10, 12)
SEEDS = tuple(range(5))
CLIP_LO_HI = (2.5, 97.5)
BUSY_DATE_SHARE = 0.05
SIZING_NOTE = (
    "Backtests use fixed-horizon equal-slot sizing (max_positions equal slots, entry at the next open, exit at "
    "the close of day t+h, one random draw per day when signals exceed free slots) instead of the spec's "
    "ATR-stop / 1%-risk sizing: the fixed-horizon exit is what the statistics actually tested. Costs are "
    "config.COSTS only; capacity, market impact and slippage beyond that are not modelled. The equity curve "
    "is realized P&L: each trade's result is booked on its exit date and open positions are not marked to "
    "market, so max drawdown and Sharpe understate the risk actually carried. Discovery-split backtests are "
    "in-sample and shown only for reference.")
TAIL_NOTE = (
    "Tail check on every validation-split firing with a known return (not only the slot-limited simulated "
    "trades). Returns are first clipped at the gate's pooled 0.1/99.9 percentile bounds over the whole "
    "split, and means are date-weighted (mean of per-date means), exactly as in the gate, so the raw figure "
    "equals the gate's validation mean return. The clipped figure is the same date-weighted mean after also "
    "clipping the firings' returns at their own 2.5th/97.5th percentiles. Verdict: \"mean not positive\" "
    "when the raw mean is not above 0; \"not tail-driven\" when the clipped mean keeps more than half of the "
    "raw mean; otherwise \"tail-driven\". The per-firing (occurrence-weighted) mean is shown separately: "
    "firings cluster on stress days (many assets trigger on the same sell-off), so it can differ sharply "
    "from the date-weighted mean. The backtest's slot limit caps how many of those same-day firings are "
    "actually entered.")
ROBUSTNESS_NOTE = (
    "The spread across seeds comes from the random daily fill (which of the day's signals get the free "
    "slots); the spread across max_positions from capacity. The universe baseline enters any eligible row "
    "of the view with the same direction, slots, seeds and fill, so strategy minus baseline is the rule's "
    "value over just being in this universe.")
SUSPECT_HEADLINE = ("Outperformance vs SPY is within the survivorship drift of this universe; judge it "
                    "against the universe baseline, and treat the whole result as suspect.")
HOLDOUT_LOCKED = "Holdout locked: no finalist from the primary views (ETF, crypto) passed validation."
_TEMPLATES = Path(__file__).parent / "templates"

GATE_SHOWN = [("view", "View"), ("kind", "Kind"), ("rule", "Rule"), ("direction", "Dir"),
              ("disc_edge", "Disc edge (bps)"), ("disc_q", "Disc q"), ("val_n", "Val n"),
              ("val_n_dates", "Val dates"), ("val_mean", "Val mean excess (bps)"),
              ("val_mean_ret", "Val mean return (bps)"), ("val_edge", "Val edge (bps)"),
              ("val_lift", "Val lift (bps)"), ("val_p", "Val p"), ("val_q", "Val q"),
              ("finalist", "Finalist"), ("holdout_eligible", "Holdout eligible"), ("reason", "Reason")]
_BPS = {"disc_edge", "val_mean", "val_mean_ret", "val_edge", "val_lift"}
_P3 = {"disc_q", "val_p", "val_q"}
METRIC_ROWS = [("total_return", "Total return", "pct"), ("cagr", "CAGR", "pct"),
               ("max_drawdown", "Max drawdown", "pct"), ("sharpe_monthly", "Sharpe (monthly)", "num"),
               ("n_trades", "Trades", "int"), ("exposure", "Exposure", "pct")]


def _num(v, kind: str) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "int":
        return str(int(v))
    return f"{v:.2f}"


def _gate_cell(col: str, v) -> str:
    if col in ("view", "kind", "rule", "direction", "reason"):
        return str(v)
    if col in ("finalist", "holdout_eligible"):
        return "yes" if bool(v) else ""
    if pd.isna(v):
        return ""
    if col in ("val_n", "val_n_dates"):
        return str(int(v))
    if col in _BPS:
        return f"{v * 1e4:.1f}"
    if col in _P3:
        return f"{v:.3f}"
    return f"{v:.2f}"


def _gate_rows(g: pd.DataFrame) -> list[dict]:
    return [{"finalist": bool(r["finalist"]),
             "cells": [_gate_cell(c, r[c]) for c, _ in GATE_SHOWN]} for r in g.to_dict("records")]


def _bench(market: str, cache: dict) -> pd.Series | None:
    """Benchmark closes (holdout locked), cached per market; None when the file is missing."""
    if market not in cache:
        try:
            cache[market] = store.load_bars(config.BENCHMARK[market], market)["close"]
        except (FileNotFoundError, KeyError, OSError):
            cache[market] = None
    return cache[market]


def _equity_chart(equity: pd.Series, bench: pd.Series | None, title: str) -> str:
    fig = go.Figure(go.Scatter(x=equity.index, y=equity.values, name="Strategy", mode="lines"))
    if bench is not None:
        b = bench.dropna().loc[equity.index[0]:equity.index[-1]]
        if len(b) > 1:
            fig.add_trace(go.Scatter(x=b.index, y=(b / b.iloc[0]).values, name="Benchmark buy-and-hold",
                                     mode="lines"))
    fig.update_layout(title=title, height=340, margin={"l": 50, "r": 20, "t": 50, "b": 40},
                      yaxis={"title": "Equity (start = 1)"})
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def _metric_rows(m: dict) -> list[list[str]]:
    rows = [[label, _num(m[key], kind)] for key, label, kind in METRIC_ROWS]
    for key, label in (("bench_total_return", "Benchmark total return"), ("bench_cagr", "Benchmark CAGR"),
                       ("bench_max_drawdown", "Benchmark max drawdown")):
        rows.append([label, _num(m[key], "pct")])
    rows.append(["Benchmark Sharpe (monthly)", _num(m["bench_sharpe_monthly"], "num")])
    return rows


def _split_slice(bench: pd.Series | None, equity: pd.Series) -> pd.Series | None:
    return None if bench is None else bench.loc[equity.index[0]:equity.index[-1]]


def tail_verdict(raw: float, clipped: float) -> str:
    """"mean not positive" (raw <= 0 or NaN), "not tail-driven" (clipped > raw / 2), else "tail-driven"."""
    if not raw > 0:
        return "mean not positive"
    return "not tail-driven" if clipped > 0.5 * raw else "tail-driven"


def tail_check(panel: pd.DataFrame, rule: str, direction: str, horizon: int) -> dict:
    """Date-weighted tail check of ``rule`` on the validation split.

    ``ret_{direction}_{horizon}`` is clipped with the gate's pooled ``clip_bounds`` over every
    validation row, then restricted to the rule's firings with a known return. ``raw`` is the
    date-weighted mean (equals the gate's ``val_mean_ret``); ``clipped`` is the date-weighted
    mean after also clipping the firings at their own CLIP_LO_HI percentiles. ``occ_mean`` is the
    per-firing mean and ``busy_share`` the share of firings on the busiest BUSY_DATE_SHARE of
    firing dates (by firing count, at least one date).
    """
    p = panel.loc[(panel["split"] == "validation").to_numpy()]
    r_all = p[f"ret_{direction}_{horizon}"].to_numpy(dtype="float64")
    r_all = clip_values(r_all, clip_bounds(r_all))
    keep = rule_mask(p, rule) & ~np.isnan(r_all)
    r, d = r_all[keep], p["date"].to_numpy()[keep]
    if not len(r):
        return {"n": 0, "n_dates": 0, "raw": np.nan, "clipped": np.nan, "occ_mean": np.nan,
                "busy_share": np.nan, "verdict": tail_verdict(np.nan, np.nan)}
    lo, hi = np.percentile(r, CLIP_LO_HI)
    raw = date_weighted_mean(r, d)
    clipped = date_weighted_mean(np.clip(r, lo, hi), d)
    counts = pd.Series(d).value_counts()
    k = max(1, int(np.ceil(BUSY_DATE_SHARE * len(counts))))
    return {"n": len(r), "n_dates": len(counts), "raw": raw, "clipped": clipped,
            "occ_mean": float(r.mean()), "busy_share": float(counts.nlargest(k).sum() / len(r)),
            "verdict": tail_verdict(raw, clipped)}


def _bps(v) -> str:
    return _num(v * 1e4 if v is not None and not pd.isna(v) else np.nan, "num")


def _baseline_runs(panel: pd.DataFrame, direction: str, horizon: int) -> pd.DataFrame:
    """Same-model universe baseline (rule ``ALL_RULE``) over the max_positions x seeds grid."""
    runs = []
    for mp in MAX_POSITIONS_GRID:
        for sd in SEEDS:
            _, eq = simulate(panel, ALL_RULE, direction, "validation", horizon, max_positions=mp, seed=sd)
            m = metrics(eq)
            runs.append({"max_positions": mp, "seed": sd, "baseline_total_return": m["total_return"],
                         "baseline_sharpe": m["sharpe_monthly"]})
    return pd.DataFrame(runs)


def _finalist_block(panel: pd.DataFrame, g: dict, view: str, horizon: int, bench: pd.Series | None,
                    baseline: pd.DataFrame):
    rule, direction = g["rule"], g["direction"]
    t_val, eq_val = simulate(panel, rule, direction, "validation", horizon)
    m_val = metrics(eq_val, _split_slice(bench, eq_val))
    t_disc, eq_disc = simulate(panel, rule, direction, "discovery", horizon)
    m_disc = metrics(eq_disc, _split_slice(bench, eq_disc)) if len(eq_disc) else None

    yr = yearly_returns(eq_val)
    ntr = (t_val["date"].dt.year.value_counts() if len(t_val) else pd.Series(dtype=int))
    years = [{"year": int(y), "ret": _num(v, "pct"), "n": int(ntr.get(y, 0))} for y, v in yr.items()]

    runs = []
    for mp in MAX_POSITIONS_GRID:
        for sd in SEEDS:
            _, eq = simulate(panel, rule, direction, "validation", horizon, max_positions=mp, seed=sd)
            m = metrics(eq)
            runs.append({"view": view, "rule": rule, "direction": direction, "max_positions": mp, "seed": sd,
                         "total_return": m["total_return"], "sharpe_monthly": m["sharpe_monthly"],
                         "n_trades": m["n_trades"]})
    rob = pd.DataFrame(runs).merge(baseline, on=["max_positions", "seed"], how="left")
    rob["excess_vs_baseline"] = rob["total_return"] - rob["baseline_total_return"]

    def agg(col, kind):
        s = rob[col].dropna()
        return [_num(s.min(), kind), _num(s.median(), kind), _num(s.max(), kind)] if len(s) else ["n/a"] * 3

    def med(col, kind):
        s = rob[col].dropna()
        return _num(s.median(), kind) if len(s) else "n/a"

    tail = tail_check(panel, rule, direction, horizon)
    rob_ret, rob_ex = agg("total_return", "pct"), agg("excess_vs_baseline", "pct")
    block = {
        "rule": rule, "direction": direction, "view": view,
        "suspect": view in SUSPECT, "eligible": bool(g["holdout_eligible"]),
        "headline": (f"median total return {rob_ret[1]} (range {rob_ret[0]} to {rob_ret[2]}); "
                     f"median excess over the universe baseline {rob_ex[1]} "
                     f"(range {rob_ex[0]} to {rob_ex[2]})"),
        "val_metrics": _metric_rows(m_val), "bench_missing": bench is None,
        "val_chart": _equity_chart(eq_val, bench, f"Validation equity: {rule} ({view})"),
        "years": years,
        "disc_metrics": _metric_rows(m_disc) if m_disc else None,
        "disc_chart": _equity_chart(eq_disc, bench, f"Discovery equity (in-sample): {rule} ({view})")
        if m_disc else "",
        "n_runs": len(rob),
        "rob_ret": rob_ret, "rob_sharpe": agg("sharpe_monthly", "num"), "rob_excess": rob_ex,
        "rob_base_ret": agg("baseline_total_return", "pct"), "rob_base_sharpe": agg("baseline_sharpe", "num"),
        "compare": [["Strategy (median of runs)", med("total_return", "pct"), med("sharpe_monthly", "num")],
                    ["Universe baseline, same model (median of runs)", med("baseline_total_return", "pct"),
                     med("baseline_sharpe", "num")],
                    ["Benchmark buy-and-hold (" + config.BENCHMARK[MARKET_OF_VIEW[view]] + ")",
                     _num(m_val["bench_total_return"], "pct"), _num(m_val["bench_sharpe_monthly"], "num")]],
        "rob_rows": [[str(r.max_positions), str(r.seed), _num(r.total_return, "pct"),
                      _num(r.sharpe_monthly, "num"), _num(r.n_trades, "int"), _num(r.baseline_total_return, "pct"),
                      _num(r.excess_vs_baseline, "pct")] for r in rob.itertuples()],
        "tail": {"n": tail["n"], "n_dates": tail["n_dates"], "raw": _bps(tail["raw"]),
                 "clipped": _bps(tail["clipped"]), "verdict": tail["verdict"], "occ_mean": _bps(tail["occ_mean"]),
                 "busy_share": _num(tail["busy_share"], "pct"),
                 "busy_pct": f"{BUSY_DATE_SHARE * 100:.0f}%"},
    }
    return block, rob


def build_finalists_report(panels: dict[str, pd.DataFrame], out_path: Path, horizon: int = 10,
                           as_of: str | None = None,
                           signal_info: dict[str, tuple[str, str]] | None = None
                           ) -> tuple[Path, pd.DataFrame]:
    """Gate every view, backtest each finalist, write the HTML, gate CSV and robustness CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = as_of or dt.date.today().isoformat()

    all_views = universe_views(panels)
    stock_mean = (stock_mean_text(all_views["stock"], horizon) if "stock" in all_views
                  else "not computed (no stock panel)")
    bench_cache: dict = {}
    gates: list[pd.DataFrame] = []
    robs: list[pd.DataFrame] = []
    views = []
    for view in list(all_views):
        panel = all_views.pop(view)                      # one view at a time; freed before the next
        gate = validation_gate(panel, view, horizon, signal_info)
        gates.append(gate)
        bench = _bench(MARKET_OF_VIEW[view], bench_cache)
        blocks = []
        baselines: dict[str, pd.DataFrame] = {}          # per direction, shared by this view's finalists
        for g in gate[gate["finalist"]].to_dict("records"):
            if g["direction"] not in baselines:
                baselines[g["direction"]] = _baseline_runs(panel, g["direction"], horizon)
            block, rob = _finalist_block(panel, g, view, horizon, bench, baselines[g["direction"]])
            blocks.append(block)
            robs.append(rob)
        views.append({"view": view, "suspect": view in SUSPECT, "primary": view in PRIMARY,
                      "n_gated": len(gate), "n_finalists": len(blocks),
                      "gate_rows": _gate_rows(gate), "finalists": blocks})
        del panel
        gc.collect()

    gate_all = pd.concat(gates, ignore_index=True) if gates else pd.DataFrame(columns=GATE_COLS)
    gate_all = gate_all[GATE_COLS] if len(gate_all) else pd.DataFrame(columns=GATE_COLS)
    gate_all.to_csv(out_path.parent / f"finalists_{stamp}.csv", index=False)
    rob_cols = ["view", "rule", "direction", "max_positions", "seed", "total_return", "sharpe_monthly", "n_trades",
                "baseline_total_return", "baseline_sharpe", "excess_vs_baseline"]
    rob_all = pd.concat(robs, ignore_index=True)[rob_cols] if robs else pd.DataFrame(columns=rob_cols)
    rob_all.to_csv(out_path.parent / f"finalists_robustness_{stamp}.csv", index=False)

    elig = gate_all[gate_all["holdout_eligible"].astype(bool)] if len(gate_all) else gate_all
    eligible = [f"{r.rule} ({r.view}, {r.direction})" for r in elig.itertuples()]

    env = Environment(loader=FileSystemLoader(_TEMPLATES), autoescape=select_autoescape(["j2", "html"]))
    page = env.get_template("finalists.html.j2").render(
        survivorship=SURVIVORSHIP.format(stock_mean=stock_mean), sizing_note=SIZING_NOTE, tail_note=TAIL_NOTE,
        robustness_note=ROBUSTNESS_NOTE, suspect_headline=SUSPECT_HEADLINE,
        short_note=SHORT_NOTE, cdn=CDN_URL, horizon=horizon, stamp=stamp, views=views,
        suspect_label=SUSPECT_LABEL, gate_headers=[h for _, h in GATE_SHOWN],
        eligible=eligible, holdout_locked=HOLDOUT_LOCKED)
    out_path.write_text(page, encoding="utf-8")
    return out_path, gate_all
