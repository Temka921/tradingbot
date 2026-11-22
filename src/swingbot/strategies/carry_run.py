"""Gated runner for the funding-rate carry strategies F1 and F2.

Design: docs/superpowers/specs/2026-11-22-funding-carry-design.md (periods, pass criteria,
information columns, holdout handling).

- Dev mode (``run_dev``): spot from the holdout-locked crypto panel, perp klines and funding with
  ``allow_holdout=False``. One continuous run per strategy; dev (2020-2021) and validation
  (2022-2023) are slices of it. Writes ``funding_dev_{stamp}.txt|csv`` and the validation gate
  ``funding_gate.json``.
- Final mode (``run_final``): only strategies that passed validation. The one-time 2024+ crypto
  holdout run, recorded in ``data/holdout_use_crypto_funding.json`` (exclusive create) before
  anything from 2024 on is loaded; a second run is refused without loading.

Periods are slices of the continuous equity curve re-based to 1 at the first valid rebalance
(a Sunday signal close) inside the period: the base is the equity at that Sunday's close, i.e.
the day before the Monday trade, so the slice includes the entry costs and the first day's
return (``run._slice``).

Spot relisting segments are named ``"SYM@date"`` in the panel; carry uses only the live
(plain-name) segment, because a perp listing does not continue a dead spot segment.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from swingbot import config
from swingbot.backtest.portfolio import metrics
from swingbot.data import futures, store
from swingbot.panel import build_panel
from swingbot.strategies import run
from swingbot.strategies.carry import F1_COINS, f1_weights, f2_weights, run_carry

PERIODS: dict[str, tuple[str, str | None]] = {
    "dev": ("2020-01-01", "2021-12-31"),
    "validation": ("2022-01-01", "2023-12-31"),
    "final": (config.HOLDOUT_START, None),
}
STRATEGIES = ("F1", "F2")
BTC = "BTCUSDT"
F2_LOOKBACK = 7
SPOT_COST = 0.001      # per side, spot leg notional (spec)
PERP_COST = 0.0005     # per side, perp leg notional (spec)
MIN_CAGR = 0.04        # CAGR > 4%
MIN_SHARPE = 1.0       # monthly Sharpe, annualised, >= 1.0
MAX_DRAWDOWN = 0.15    # |max drawdown| < 15%
GATE_NAME = "funding_gate.json"
RECORD_NAME = "holdout_use_crypto_funding.json"
NAN = float("nan")
INFO_COLS = ["negative_funding_share", "avg_gross_funding_yield", "turnover_per_year"]
REPORT_COLS = ["strategy", "period", "start", "end", "n_days", "total_return", "cagr", "sharpe",
               "max_drawdown", "pass", "reason", "verdict", *INFO_COLS, "worst_week",
               "btc_total_return", "btc_max_drawdown"]

NOTES = [
    "Pass criteria (each period, all must hold): CAGR > 4%, Sharpe (monthly, annualised, 0% rf) "
    ">= 1.0, |max drawdown| < 15%. Only validation gates the final test.",
    "Costs: spot 0.10% + perp 0.05% per side on each leg's traded notional; cash moves free, "
    "cash earns 0%.",
    "Periods are slices of one continuous run, re-based at the first rebalance (Sunday close) "
    "inside the period.",
    "Information columns (negative_funding_share, avg_gross_funding_yield, turnover_per_year) "
    "come from a stand-alone run of the period's own rebalances, so turnover includes the "
    "initial entry. worst_week = worst Sunday-to-Sunday return of the slice. btc_* = BTC spot "
    "buy-and-hold over the slice.",
    "Not modelled: exchange failure (FTX-type), liquidation of the short in extreme spikes "
    "(1x margin), interest on idle cash.",
]


# ---------------------------------------------------------------- inputs

def available_perps() -> set[str]:
    """Futures symbols with both a perp-kline file and a funding file."""
    raw = config.DATA_DIR / "raw"
    perp = {p.stem for p in (raw / "perp").glob("*.parquet")}
    fund = {p.stem for p in (raw / "funding").glob("*.parquet")}
    return perp & fund


def carry_coins(panel: pd.DataFrame, strategies) -> list[str]:
    """Coins to load: F1's BTC/ETH; for F2 the live (plain-name) spot segments that are ever
    eligible; always only coins with a perp and funding file."""
    avail = available_perps()
    asset = panel["asset"].astype(str)
    coins: set[str] = set()
    if "F1" in strategies:
        coins |= set(F1_COINS)
    if "F2" in strategies:
        plain = ~asset.str.contains("@", regex=False)
        coins |= set(asset[plain & panel["eligible"].eq(True)].unique())
    return sorted(c for c in coins if c in avail)


def load_inputs(panel: pd.DataFrame, strategies, allow_holdout: bool) -> dict[str, pd.DataFrame]:
    """Wide frames (dates x coins) for the carry engine.

    Spot opens/closes/eligible come from ``run.crypto_inputs`` on the panel restricted to the
    needed coins' plain-name segments (BTC is always added as the benchmark). Perp bars and
    funding are loaded per coin with the same ``allow_holdout``.
    """
    coins = carry_coins(panel, strategies)
    want = set(coins) | {BTC}
    sub = panel[panel["asset"].astype(str).isin(want)]
    spot_open, spot_close, eligible = run.crypto_inputs(sub, allow_holdout=allow_holdout)
    po, pc, fd = {}, {}, {}
    for c in coins:
        b = store.load_bars(c, "perp", allow_holdout=allow_holdout)
        po[c], pc[c] = b["open"], b["close"]
        fd[c] = futures.daily_funding(futures.load_funding(c, allow_holdout=allow_holdout))

    def wide(d: dict) -> pd.DataFrame:
        df = pd.DataFrame(d).sort_index() if d else pd.DataFrame()
        df.index = pd.DatetimeIndex(df.index, name="date")
        return df.astype(float)

    out = {"spot_open": spot_open, "spot_close": spot_close, "eligible": eligible,
           "perp_open": wide(po), "perp_close": wide(pc), "funding": wide(fd),
           "coins": coins}
    if not allow_holdout:  # defensive: nothing from the holdout in dev/validation
        cut = pd.Timestamp(config.HOLDOUT_START)
        for k, v in out.items():
            if isinstance(v, pd.DataFrame) and len(v) and v.index.max() >= cut:
                raise RuntimeError(f"holdout data in dev inputs: {k}")
    return out


def gap_summary(f: dict) -> str:
    """Missing closes inside each loaded coin's own span (each forces an exit in run_carry)."""
    parts = []
    for k in ("spot_close", "perp_close"):
        df = f[k][[c for c in f["coins"] if c in f[k].columns]]
        gaps = {}
        for c in df.columns:
            s = df[c]
            a, b = s.first_valid_index(), s.last_valid_index()
            if a is not None:
                n = int(s.loc[a:b].isna().sum())
                if n:
                    gaps[c] = n
        worst = ", ".join(f"{c}={n}" for c, n in sorted(gaps.items(), key=lambda x: -x[1])[:5])
        parts.append(f"{k}: {sum(gaps.values())} missing days in {len(gaps)} coins"
                     + (f" (worst: {worst})" if worst else ""))
    return "Data gaps (each forces an exit until the next rebalance): " + "; ".join(parts)


# ---------------------------------------------------------------- strategy runs

def _f2_valid(funding: pd.DataFrame, eligible: pd.DataFrame) -> pd.DatetimeIndex:
    """Sundays with at least one eligible coin that has 7 full days of funding (the candidate
    pool before the positive-mean test; an all-cash week from negative funding still counts)."""
    f = funding.sort_index()
    if not len(f):
        return pd.DatetimeIndex([])
    f = f.reindex(pd.date_range(f.index[0], f.index[-1], freq="D", name="date"))
    full = f.notna().rolling(F2_LOOKBACK, min_periods=F2_LOOKBACK).sum().eq(F2_LOOKBACK)
    sundays = f.index[f.index.day_of_week == 6]
    el = eligible.reindex(index=sundays, columns=f.columns)
    el = el.astype("boolean").fillna(False).astype(bool)
    ok = (full.loc[sundays] & el).any(axis=1).to_numpy()
    return sundays[ok]


def _f1_valid(f: dict, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Sundays on which BTC and ETH both have spot and perp closes."""
    ok = np.ones(len(dates), dtype=bool)
    for k in ("spot_close", "perp_close"):
        df = f[k].reindex(index=dates, columns=list(F1_COINS))
        ok &= df.notna().all(axis=1).to_numpy()
    return dates[ok]


def strategy_weights(strategy: str, f: dict) -> pd.DataFrame:
    if strategy == "F1":
        return f1_weights(f["spot_close"].index)
    if strategy == "F2":
        return f2_weights(f["funding"], f["eligible"], lookback=F2_LOOKBACK)
    raise ValueError(f"not a carry strategy: {strategy}")


def _carry(f: dict, w: pd.DataFrame, days: pd.DatetimeIndex | None = None) -> pd.Series:
    so, sc = f["spot_open"], f["spot_close"]
    if days is not None:
        so, sc = so.reindex(days), sc.reindex(days)
    return run_carry(so, sc, f["perp_open"], f["perp_close"], f["funding"], w,
                     spot_cost=SPOT_COST, perp_cost=PERP_COST)


def run_strategy(strategy: str, f: dict) -> dict:
    """One continuous run: {"equity", "weights", "valid"} (valid = rebalance dates)."""
    w = strategy_weights(strategy, f)
    eq = _carry(f, w)
    if strategy == "F1":
        valid = _f1_valid(f, pd.DatetimeIndex(w.index))
    else:
        valid = _f2_valid(f["funding"], f["eligible"])
    return {"equity": eq, "weights": w, "valid": valid}


def period_info(f: dict, w: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """run_carry's attrs for a stand-alone run of the rebalances in [start, end)."""
    days = f["spot_close"].index
    days = days[(days >= start) & (days <= end)]
    wp = w[(w.index >= start) & (w.index < end)]
    if not len(wp) or not len(days):
        return {k: NAN for k in INFO_COLS}
    eq = _carry(f, wp, days)
    return {k: float(eq.attrs.get(k, NAN)) for k in INFO_COLS}


# ---------------------------------------------------------------- evaluation

def check_criteria(m: dict) -> list[str]:
    """Failed pass criteria (empty = pass). NaN fails every criterion it touches."""
    fails = []
    if not m["cagr"] > MIN_CAGR:
        fails.append("CAGR <= 4%")
    if not m["sharpe"] >= MIN_SHARPE:
        fails.append("Sharpe < 1.0")
    if not abs(m["max_drawdown"]) < MAX_DRAWDOWN:
        fails.append("max drawdown >= 15%")
    return fails


def _worst_week(curve: pd.Series) -> float:
    """Worst Sunday-to-Sunday return; the first week runs from the slice's base value."""
    wk = curve.resample("W-SUN").last().dropna()
    wk = wk[wk.index > curve.index[0]]
    pts = pd.concat([curve.iloc[:1], wk])
    r = (pts / pts.shift(1) - 1.0).dropna()
    return float(r.min()) if len(r) else NAN


def evaluate(equity: pd.Series, period: tuple, valid_dates, btc_close: pd.Series) -> dict:
    """Pass criteria and slice-level info on one period of a continuous equity curve.

    The slice starts at the first ``valid_dates`` entry inside ``period`` (end ``None`` = open)
    and is re-based to 1 at that date's equity (``run._slice``: flat cash 1.0 before the curve).
    """
    p0 = pd.Timestamp(period[0])
    p1 = pd.Timestamp(period[1]) if period[1] is not None else None
    out = {"start": None, "end": None, "n_days": 0, "total_return": NAN, "cagr": NAN,
           "sharpe": NAN, "max_drawdown": NAN, "pass": False, "reason": "",
           "worst_week": NAN, "btc_total_return": NAN, "btc_max_drawdown": NAN}
    eq_idx = equity.dropna().index
    vd = pd.DatetimeIndex(valid_dates)
    vd = vd[(vd >= p0) & ((vd <= p1) if p1 is not None else True)]
    vd = vd[vd < eq_idx[-1]] if len(eq_idx) else vd[:0]
    if not len(vd):
        out["reason"] = "no valid rebalance in period"
        return out
    start = vd[0]
    s = run._slice(equity, start, p1)
    m = metrics(s)
    out.update(start=start, end=s.index[-1], n_days=int(len(s)),
               total_return=m["total_return"], cagr=m["cagr"], sharpe=m["sharpe_monthly"],
               max_drawdown=m["max_drawdown"], worst_week=_worst_week(s))
    b = run._slice(btc_close, start, out["end"], pre=None)
    if len(b) > 1:
        bm = metrics(b)
        out.update(btc_total_return=bm["total_return"], btc_max_drawdown=bm["max_drawdown"])
    fails = check_criteria(out)
    out["pass"] = not fails
    out["reason"] = "; ".join(fails)
    return out


def evaluate_strategy(strategy: str, f: dict, labels) -> list[dict]:
    """Rows for ``strategy`` over the period ``labels`` from one continuous run."""
    r = run_strategy(strategy, f)
    btc = f["spot_close"][BTC]
    rows = []
    for label in labels:
        row = {"strategy": strategy, "period": label,
               **evaluate(r["equity"], PERIODS[label], r["valid"], btc)}
        if row["start"] is not None:
            row.update(period_info(f, r["weights"], row["start"], row["end"]))
        else:
            row.update({k: NAN for k in INFO_COLS})
        rows.append(row)
    return rows


# ---------------------------------------------------------------- reports

def _table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    first = [c for c in REPORT_COLS if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]]


def _write(df: pd.DataFrame, reports_dir: Path, name: str, header: list[str]) -> str:
    reports_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports_dir / f"{name}.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 50,
                           "display.max_colwidth", 60):
        body = df.to_string(index=False, float_format=lambda x: f"{x:.4f}") if len(df) else "(no rows)"
    text = "\n".join(header + [""] + NOTES + ["", body, ""])
    (reports_dir / f"{name}.txt").write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------- dev mode

def run_dev(reports_dir: Path, stamp: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Dev mode: F1/F2 dev + validation on pre-2024 data only; writes the gate and tables."""
    stamp = run._stamp(stamp)
    cut = pd.Timestamp(config.HOLDOUT_START)
    panel = pd.read_parquet(config.DATA_DIR / "panel" / "crypto.parquet",
                            columns=["date", "asset", "eligible"])
    panel = panel[panel["date"] < cut]
    f = load_inputs(panel, STRATEGIES, allow_holdout=False)
    del panel
    gc.collect()
    rows = []
    for s in STRATEGIES:
        rows += evaluate_strategy(s, f, ("dev", "validation"))
    gate = {s: bool(next(r["pass"] for r in rows if r["strategy"] == s and r["period"] == "validation"))
            for s in STRATEGIES}
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / GATE_NAME).write_text(json.dumps(gate, indent=2), encoding="utf-8")
    df = _table(rows)
    header = [f"Funding-rate carry: dev and validation ({stamp})",
              f"Coins loaded: {len(f['coins'])}",
              "Validation gate: " + ", ".join(f"{k}={'PASS' if v else 'fail'}" for k, v in gate.items()),
              gap_summary(f)]
    print(_write(df, reports_dir, f"funding_dev_{stamp}", header))
    return df, gate


# ---------------------------------------------------------------- final mode

def preflight(strategies) -> list[str]:
    """Files the final needs that are missing (empty list = ready)."""
    raw = config.DATA_DIR / "raw"
    need = [config.DATA_DIR / "universe_meta.parquet", store._path(BTC, "crypto")]
    if "F1" in strategies:
        for c in F1_COINS:
            need += [store._path(c, "crypto"), store._path(c, "perp"), futures._funding_path(c)]
    missing = [str(p) for p in dict.fromkeys(need) if not p.exists()]
    if "F2" in strategies and not available_perps():
        missing.append(f"perp + funding files in {raw / 'perp'} and {raw / 'funding'}")
    return missing


def run_final(reports_dir: Path, stamp: str | None = None) -> pd.DataFrame:
    """Final mode: the one-time 2024+ run for strategies marked passing in ``funding_gate.json``.

    No gate: SystemExit. Nothing passed: prints that, loads nothing, writes no record. Missing
    files: SystemExit before anything is spent. Record present: "refused" rows (recorded results
    shown), nothing loaded. Otherwise the record is created before the first 2024+ load and the
    results, the error or the interruption are saved into it (``run._spend``).
    """
    gate_path = reports_dir / GATE_NAME
    if not gate_path.exists():
        raise SystemExit(f"no validation gate at {gate_path}: run scripts/funding_carry.py first")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    passed = [s for s in STRATEGIES if gate.get(s) is True]
    if not passed:
        print("No strategy passed validation: final test not run, crypto holdout not used.")
        return pd.DataFrame(columns=REPORT_COLS)
    missing = preflight(passed)
    if missing:
        raise SystemExit(f"missing files, crypto holdout NOT used: {missing}")
    stamp = run._stamp(stamp)
    record = config.DATA_DIR / RECORD_NAME
    if record.exists():
        rows = run._refused_rows("crypto", passed, record)
    else:
        payload = {"utc": run._now(), "who": "scripts/funding_carry.py --final",
                   "strategies": list(passed), "period": list(PERIODS["final"])}

        def body():
            panel = build_panel("crypto", allow_holdout=True)
            panel = panel[["date", "asset", "eligible"]].copy()
            gc.collect()
            f = load_inputs(panel, passed, allow_holdout=True)
            del panel
            rows = []
            for s in passed:
                for r in evaluate_strategy(s, f, ("final",)):
                    r.update(market="crypto",
                             verdict="confirmed" if r["pass"] else "not confirmed")
                    rows.append(r)
            return rows, {"end_date": f"{f['spot_close'].index[-1]:%Y-%m-%d}",
                          "data_gaps": gap_summary(f)}

        def error_rows(err):
            return [{"market": "crypto", "strategy": s, "period": "final", "pass": False,
                     "verdict": "not confirmed", "reason": f"error: {err}"} for s in passed]

        rows = run._spend(record, payload, body, error_rows)
    df = _table(rows)
    skipped = [s for s in STRATEGIES if s not in passed]
    header = [f"Funding-rate carry: final test, 2024-01-01 to end ({stamp})",
              "Run (passed validation): " + ", ".join(passed),
              "Not run (failed validation): " + (", ".join(skipped) or "none"),
              "Verdict: confirmed only if all three criteria hold on 2024+."]
    print(_write(df, reports_dir, f"funding_final_{stamp}", header))
    return df
