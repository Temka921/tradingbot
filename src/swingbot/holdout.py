"""One-time holdout run: the only code path that may read 2024+ data.

It runs at most once, and only for validation finalists of a primary view
(``views.PRIMARY``: etf, crypto). ``config.DATA_DIR / "holdout_used.json"`` is
written before any data is loaded, so a crashed run still counts as used and
nobody can "retry" after peeking. Every bar load goes through
``store.load_bars(..., allow_holdout=True)``, which appends to
``data/holdout_log.txt``.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment

from swingbot import config
from swingbot.analysis.pairs import pair_table
from swingbot.analysis.solo import SIGNAL_INFO, solo_table
from swingbot.analysis.views import MARKET_OF_VIEW, PRIMARY, universe_views
from swingbot.backtest.portfolio import metrics, simulate, yearly_returns
from swingbot.data import store
from swingbot.panel import build_panel

MARKER_NAME = "holdout_used.json"
P_CONFIRM = 0.10
LOCKED_MSG = "holdout locked: no eligible finalists"
NAN = float("nan")


def _is_true(v) -> bool:
    """CSV round-trips bools as True/False or as the strings "True"/"False"."""
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v) if not pd.isna(v) else False


def eligible_finalists(finalists_csv: Path) -> pd.DataFrame:
    """Rows of the gate CSV with ``holdout_eligible`` true and a primary view."""
    g = pd.read_csv(finalists_csv)
    keep = g["holdout_eligible"].map(_is_true) & g["view"].astype(str).isin(PRIMARY)
    return g[keep.to_numpy(dtype=bool)].reset_index(drop=True)


def _write_marker(path: Path, finalists_csv: Path, horizon: int, rules: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "finalists_csv": str(finalists_csv), "horizon": horizon, "rules": rules}
    with open(path, "x", encoding="utf-8") as f:  # exclusive create: never overwrite a marker
        json.dump(payload, f, indent=2)


def _used_on(path: Path) -> str:
    try:
        return str(json.loads(path.read_text(encoding="utf-8"))["timestamp"])
    except (OSError, ValueError, KeyError, TypeError):
        return "unknown date"


def verdict(edge: float, p: float, mean_ret: float) -> str:
    """"confirmed" iff edge > 0, p < 0.10 and mean_ret > 0 (NaN fails every condition)."""
    return "confirmed" if (edge > 0 and p < P_CONFIRM and mean_ret > 0) else "not confirmed"


def _edge_stats(view_panel: pd.DataFrame, kind: str, rule: str, horizon: int,
                signal_info, cache: dict) -> dict:
    if kind == "pair":
        if "pair" not in cache:
            cache["pair"] = pair_table(view_panel, horizon, split="holdout",
                                       signal_info=signal_info).set_index(["a", "b"])
        a, b = rule.split("+")
        v = cache["pair"].loc[(a, b)]
        return {"n": int(v["n_ab"]), "n_dates": int(v["n_dates_ab"]), "edge": float(v["edge_ab"]),
                "p": float(v["p_ab"]), "mean_ret": float(v["mean_ret_ab"]),
                "mean_excess": float(v["mean_ab"]), "insufficient": bool(v["insufficient"])}
    if "solo" not in cache:
        cache["solo"] = solo_table(view_panel, horizon, split="holdout",
                                   signal_info=signal_info).set_index("signal")
    v = cache["solo"].loc[rule]
    return {"n": int(v["n_all"]), "n_dates": int(v["n_dates"]), "edge": float(v["edge_vs_all_days"]),
            "p": float(v["p"]), "mean_ret": float(v["mean_ret"]),
            "mean_excess": float(v["mean_all"]), "insufficient": bool(v["insufficient"])}


def _bench(market: str, cache: dict) -> pd.Series | None:
    """Benchmark closes through the logged holdout load (part of the one allowed run)."""
    if market not in cache:
        try:
            cache[market] = store.load_bars(config.BENCHMARK[market], market, allow_holdout=True)["close"]
        except (FileNotFoundError, KeyError, OSError):
            cache[market] = None
    return cache[market]


def _evaluate(views: dict[str, pd.DataFrame], eligible: pd.DataFrame, horizon: int,
              signal_info) -> list[dict]:
    out, edge_cache, bench_cache = [], {}, {}
    for g in eligible.itertuples(index=False):
        view, kind, rule, direction = str(g.view), str(g.kind), str(g.rule), str(g.direction)
        row = {"view": view, "kind": kind, "rule": rule, "direction": direction,
               "n": None, "n_dates": None, "edge": NAN, "p": NAN, "mean_ret": NAN,
               "mean_excess": NAN, "metrics": {}, "yearly": "", "verdict": "not confirmed", "reason": ""}
        try:
            _evaluate_rule(row, views, horizon, signal_info, edge_cache, bench_cache)
        except Exception as e:  # noqa: BLE001 - the holdout is spent: always leave a record
            row.pop("insufficient", None)
            row["verdict"] = "not confirmed"
            row["reason"] = f"error: {type(e).__name__}: {e}"
        out.append(row)
    return out


def _evaluate_rule(row: dict, views: dict[str, pd.DataFrame], horizon: int, signal_info,
                   edge_cache: dict, bench_cache: dict) -> None:
    """Fill ``row`` in place with the holdout stats, backtest and verdict of one rule."""
    view, kind, rule, direction = row["view"], row["kind"], row["rule"], row["direction"]
    vp = views.get(view)
    if vp is None or not (vp["split"] == "holdout").any():
        row["reason"] = f"view {view} not in the holdout panel"
        return
    row.update(_edge_stats(vp, kind, rule, horizon, signal_info, edge_cache.setdefault(view, {})))
    _, equity = simulate(vp, rule, direction, "holdout", horizon)
    if len(equity) > 1:
        row["metrics"] = metrics(equity, _bench(MARKET_OF_VIEW[view], bench_cache))
        yr = yearly_returns(equity)
        row["yearly"] = ", ".join(f"{y}: {r * 100:.1f}%" for y, r in yr.items())
    row["verdict"] = verdict(row["edge"], row["p"], row["mean_ret"])
    if row.pop("insufficient"):
        # CLAUDE.md: insufficient rows are never ranked as winners
        row["verdict"] = "not confirmed"
        row["reason"] = "insufficient occurrences on the holdout"
    elif row["verdict"] == "not confirmed":
        row["reason"] = "needs edge > 0, p < 0.10 and mean return > 0"


def _bps(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v * 1e4:.1f}"


def _pct(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v * 100:.1f}%"


def _num(v, nd=3) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.{nd}f}"


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Holdout report {{ stamp }}</title>
<style>body{font-family:system-ui,sans-serif;margin:1.5rem;color:#222}
table{border-collapse:collapse;font-size:.85rem}th,td{border:1px solid #ccc;padding:.25rem .45rem;text-align:right}
th{background:#f3f3f3}td.l{text-align:left}.note{max-width:60rem;color:#555}</style></head>
<body>
<h1>One-time holdout run</h1>
<p class="note">Run at {{ run_at }} from <code>{{ finalists_csv }}</code>, horizon {{ horizon }} days.
Holdout split (2024-01-01 onward), loaded once with every access logged to <code>data/holdout_log.txt</code>.
This run is recorded in <code>data/holdout_used.json</code> and cannot be repeated. Verdict "confirmed" needs
edge &gt; 0, HAC p &lt; 0.10 and mean return &gt; 0 on the holdout, with enough occurrences. Costs
(config.COSTS) are included. Backtest: {{ max_positions }} equal slots, seed 0, fixed {{ horizon }}-day exit,
versus buy-and-hold of the market benchmark over the same dates. This is a research result, not investment advice.</p>
<table>
<tr><th>View</th><th>Kind</th><th>Rule</th><th>Dir</th><th>n</th><th>Dates</th><th>Edge (bps)</th><th>p</th>
<th>Mean return (bps)</th><th>Mean excess (bps)</th><th>Total return</th><th>CAGR</th><th>Max DD</th>
<th>Sharpe (monthly)</th><th>Trades</th><th>Exposure</th><th>Bench total</th><th>Bench CAGR</th>
<th>Bench max DD</th><th>Bench Sharpe</th><th>Per year</th><th>Verdict</th><th>Reason</th></tr>
{% for r in rows %}<tr><td class="l">{{ r.view }}</td><td class="l">{{ r.kind }}</td><td class="l">{{ r.rule }}</td>
<td class="l">{{ r.direction }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}
<td class="l">{{ r.yearly }}</td><td class="verdict">{{ r.verdict }}</td><td class="l">{{ r.reason }}</td></tr>
{% endfor %}</table>
</body></html>
"""


def _render(rows: list[dict], out_path: Path, stamp: str, run_at: str, finalists_csv: Path,
            horizon: int) -> Path:
    shown = []
    for r in rows:
        m = r["metrics"]
        cells = ["n/a" if r["n"] is None else r["n"], "n/a" if r["n_dates"] is None else r["n_dates"],
                 _bps(r["edge"]), _num(r["p"]), _bps(r["mean_ret"]), _bps(r["mean_excess"]),
                 _pct(m.get("total_return")), _pct(m.get("cagr")), _pct(m.get("max_drawdown")),
                 _num(m.get("sharpe_monthly"), 2),
                 "n/a" if pd.isna(m.get("n_trades", NAN)) else int(m["n_trades"]),
                 _pct(m.get("exposure")), _pct(m.get("bench_total_return")), _pct(m.get("bench_cagr")),
                 _pct(m.get("bench_max_drawdown")), _num(m.get("bench_sharpe_monthly"), 2)]
        shown.append({**r, "cells": cells})
    html = Environment(autoescape=True).from_string(_TEMPLATE).render(
        rows=shown, stamp=stamp, run_at=run_at, finalists_csv=str(finalists_csv), horizon=horizon,
        max_positions=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def run_holdout(finalists_csv: Path, horizon: int = 10, out_dir: Path | None = None,
                signal_info: dict[str, tuple[str, str]] | None = None) -> Path | None:
    """Run the one-time holdout test for the eligible finalists of ``finalists_csv``.

    Refusal order: (1) no eligible finalist -> print ``LOCKED_MSG``, return None, touch
    nothing; (2) marker exists -> ``RuntimeError("holdout already used on <timestamp>")``;
    (3) pre-flight: unknown signals or a CSV direction that differs from a signal's registry
    direction raise ValueError; (4) write the marker; (5) build holdout panels only for the markets the eligible views
    need. An error while evaluating one rule is recorded as "not confirmed" with reason
    "error: ..." and the report is still written. Returns the report path (``out_dir`` defaults to the repo's ``reports/``).
    ``signal_info`` (default: the signal registry) is passed to the edge tables.
    """
    finalists_csv = Path(finalists_csv)
    eligible = eligible_finalists(finalists_csv)
    if eligible.empty:
        print(LOCKED_MSG)
        return None
    marker = config.DATA_DIR / MARKER_NAME
    if marker.exists():
        raise RuntimeError(f"holdout already used on {_used_on(marker)}")
    rules = [{"view": str(r.view), "kind": str(r.kind), "rule": str(r.rule), "direction": str(r.direction)}
             for r in eligible.itertuples(index=False)]
    # Pre-flight (no data needed): an unknown signal would crash after the marker and burn the holdout.
    info = SIGNAL_INFO if signal_info is None else signal_info
    unknown = sorted({s for r in rules for s in r["rule"].split("+") if s not in info})
    if unknown:
        raise ValueError(f"finalists CSV names unknown signals {unknown}; holdout not started")
    # Pre-flight: the CSV direction picks the ret_* column; it must match every signal's registry direction.
    wrong = [f"{r['rule']} ({r['direction']}): {s} is {info[s][1]}"
             for r in rules for s in r["rule"].split("+") if info[s][1] != r["direction"]]
    if wrong:
        raise ValueError(f"finalists CSV direction does not match the signal registry {wrong}; "
                         "holdout not started")
    try:
        _write_marker(marker, finalists_csv, horizon, rules)
    except FileExistsError:  # created between the check and the write
        raise RuntimeError(f"holdout already used on {_used_on(marker)}") from None
    run_at = json.loads(marker.read_text(encoding="utf-8"))["timestamp"]

    markets = sorted({MARKET_OF_VIEW[v] for v in eligible["view"].astype(str)})
    panels = {m: build_panel(m, allow_holdout=True) for m in markets}
    views = universe_views(panels)
    del panels
    rows = _evaluate(views, eligible, horizon, signal_info)

    stamp = dt.date.today().isoformat()
    out_dir = Path(out_dir) if out_dir is not None else config.DATA_DIR.parent / "reports"
    return _render(rows, out_dir / f"holdout_report_{stamp}.html", stamp, run_at, finalists_csv, horizon)
