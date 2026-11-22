"""Gated runner for the original strategies C1-C3, S1, S2.

Design: docs/superpowers/specs/2026-11-21-original-strategies-design.md (pass criteria, periods,
holdout handling, builder-check resolutions 1-9).

- Dev mode (``run_dev``): crypto on the holdout-locked panel, S1/S2 on US data before 2024.
  Writes the dev/validation tables, C3's direction choice (before C3's validation is computed)
  and the validation gate ``originals_gate.json``.
- Final mode (``run_final``): only strategies that passed validation. Crypto: the one-time 2024+
  holdout run, recorded in ``data/holdout_use_crypto_originals.json`` before anything is loaded.
  Stocks: Japan and Europe full history (fresh markets, no lock).

Each strategy runs once over all of its market's data; period statistics are slices of that one
equity curve, re-based to 1 at the first valid-signal date inside the period (resolutions 6, 8).
"""
from __future__ import annotations

import datetime as dt
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from swingbot import config
from swingbot.analysis.solo import clip_bounds, clip_values
from swingbot.backtest.portfolio import ALL_RULE, metrics, simulate
from swingbot.backtest.rebalance import run_weights
from swingbot.data import foreign, store, universe
from swingbot.panel import build_panel, split_relistings
from swingbot.strategies import CASH
from swingbot.strategies import crypto as cs
from swingbot.strategies import stocks as ss

PERIODS: dict[str, dict[str, tuple[str | None, str | None]]] = {
    "crypto": {"discovery": ("2018-02-01", "2021-12-31"),
               "validation": ("2022-01-01", "2023-12-31"),
               "final": (config.HOLDOUT_START, None)},
    "stock": {"discovery": ("2000-01-01", "2017-12-31"),
              "validation": ("2018-01-01", "2023-12-31")},
}
CRYPTO = ("C1", "C2", "C3")
STOCKS = ("S1", "S2")
ALL = CRYPTO + STOCKS
SEEDS = tuple(range(5))
FINAL_MARKETS = ("japan", "europe")
BTC = "BTCUSDT"
C2_FUNDING = 0.10           # per year on current short notional
LOOKBACK = 30               # C1-C3 ranking window (the builders' default)
BTC_SMA = 200
S1_HORIZON = 10
S1_SLOTS = 10
S2_MIN_VALID = 50
RECORD_NAME = "holdout_use_crypto_originals.json"
CHOICE_NAME = "originals_c3_choice.json"
GATE_NAME = "originals_gate.json"
NAN = float("nan")

REPORT_COLS = ["market", "strategy", "period", "start", "end", "n_days", "total_return", "cagr",
               "max_drawdown", "sharpe", "bench_total_return", "bench_max_drawdown", "bench_sharpe",
               "t_stat", "t_p", "pass", "reason", "verdict"]


# ---------------------------------------------------------------- inputs

def crypto_inputs(panel: pd.DataFrame, allow_holdout: bool
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(opens, closes, eligible) for the panel's assets.

    Bars come from ``store.load_bars`` with relistings re-split exactly as ``build_panel`` does,
    so ``"SYM@date"`` segment names match the panel. ``eligible`` is the panel's flag pivoted
    to dates × assets (False where the panel has no row).
    """
    assets = set(panel["asset"].astype(str).unique())
    opens, closes = {}, {}
    for sym in sorted({a.split("@")[0] for a in assets}):
        df = store.load_bars(sym, "crypto", allow_holdout=allow_holdout)
        for name, seg in split_relistings(sym, df):
            if name in assets:
                opens[name], closes[name] = seg["open"], seg["close"]
    c = pd.DataFrame(closes).sort_index()
    c.index.name = "date"
    o = pd.DataFrame(opens).reindex(index=c.index, columns=c.columns)
    p = panel[["date", "asset", "eligible"]].copy()
    p["asset"] = p["asset"].astype(str)
    el = p.pivot(index="date", columns="asset", values="eligible")
    el = el.reindex(index=c.index, columns=c.columns).eq(True)
    return o, c, el


def us_stock_inputs(stock_panel: pd.DataFrame | None = None
                    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], dict[str, str]]:
    """US single stocks (bars before the holdout only), the ^GSPC close and the home map.

    The universe is the panel's non-ETF assets when ``stock_panel`` is given, else the non-ETF
    rows of ``universe/stocks.csv``; symbols without a raw file are skipped.
    """
    if stock_panel is not None:
        etf = stock_panel["is_etf"].astype(bool).to_numpy()
        syms = sorted(stock_panel.loc[~etf, "asset"].astype(str).unique())
    else:
        u = pd.read_csv(universe.UNIVERSE_DIR / "stocks.csv")
        syms = u.loc[~u["is_etf"].astype(str).str.lower().eq("true"), "symbol"].tolist()
    idx = config.HOME_INDEX["us"]
    bars = {}
    for s in syms:
        if s == idx or not store._path(s, "stock").exists():
            continue
        df = store.load_bars(s, "stock", allow_holdout=False)
        if len(df):
            bars[s] = df
    index_close = {idx: store.load_bars(idx, "stock", allow_holdout=False)["close"]}
    return bars, index_close, {s: idx for s in bars}


def _universe_csv(market: str) -> Path:
    return universe.UNIVERSE_DIR / f"{market}.csv"


def foreign_inputs(market: str
                   ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], dict[str, str]]:
    """Japan/Europe bars (full history, glitch-filtered), home index closes and the home map.

    Reads the cached universe CSV only (raises FileNotFoundError instead of downloading).
    Stocks without a raw file or without their home index file are skipped.
    """
    if not _universe_csv(market).exists():
        raise FileNotFoundError(f"universe file missing: {_universe_csv(market)}")
    uni = {"japan": foreign.japan_universe, "europe": foreign.europe_universe}[market]()
    index_close = {}
    for idx in dict.fromkeys(uni["home_index"]):
        if store._path(idx, market).exists():
            index_close[idx] = foreign.glitch_filter(store.load_bars(idx, market))["close"]
    bars, home = {}, {}
    for r in uni.itertuples(index=False):
        if r.home_index not in index_close or not store._path(r.symbol, market).exists():
            continue
        df = foreign.glitch_filter(store.load_bars(r.symbol, market))
        if len(df):
            bars[r.symbol], home[r.symbol] = df, r.home_index
    return bars, index_close, home


def _foreign_ready(market: str) -> bool:
    raw = config.DATA_DIR / "raw" / market
    return _universe_csv(market).exists() and raw.is_dir() and any(raw.glob("*.parquet"))


def _fng_file() -> Path:
    return config.DATA_DIR / "raw" / "sentiment" / "fng.parquet"


def ensure_index(symbol: str, market: str = "stock") -> None:
    """Download ``symbol`` from Yahoo into ``data/raw/{market}`` if it has no file (script use)."""
    if store._path(symbol, market).exists():
        return
    from swingbot.data.yahoo import fetch_yahoo
    df = fetch_yahoo(symbol)
    if df is None or df.empty:
        raise RuntimeError(f"could not download {symbol}")
    store.save_bars(symbol, market, df)


def _frames(bars: dict[str, pd.DataFrame], dtype: str = "float32"
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = pd.DataFrame({s: d["close"] for s, d in bars.items()}).sort_index().astype(dtype)
    closes.index.name = "date"
    opens = pd.DataFrame({s: d["open"] for s, d in bars.items()}).reindex(closes.index).astype(dtype)
    return opens, closes


# ---------------------------------------------------------------- strategy runs

def crypto_valid_dates(strategy: str, closes: pd.DataFrame, eligible: pd.DataFrame,
                       fng: pd.Series | None = None) -> pd.DatetimeIndex:
    """Weekly signal dates on which ``strategy`` has a valid signal.

    C1/C3: BTC's 200-day SMA exists and at least one eligible coin has a 30-day return; C3 also
    needs the lagged (t − 1 day) Fear & Greed value. C2: at least one eligible non-BTC coin.
    """
    dates, _, cand = cs._inputs(closes, eligible, LOOKBACK)
    if strategy == "C2":
        ok = cand.drop(columns=BTC, errors="ignore").any(axis=1).to_numpy()
        return dates[ok]
    b = closes[BTC].dropna()
    sma_ok = b.rolling(BTC_SMA, min_periods=BTC_SMA).mean().notna()
    ok = cand.any(axis=1).to_numpy() & sma_ok.reindex(dates, fill_value=False).to_numpy(bool)
    if strategy == "C3":
        f = fng.copy()
        f.index = pd.DatetimeIndex(f.index).normalize()
        f = f[~f.index.duplicated(keep="last")]
        ok &= f.reindex(dates - pd.Timedelta(days=1)).notna().to_numpy()
    return dates[ok]


def run_crypto(strategy: str, opens: pd.DataFrame, closes: pd.DataFrame, eligible: pd.DataFrame,
               fng: pd.Series | None = None, mode: str | None = None
               ) -> tuple[pd.Series, pd.DatetimeIndex]:
    """One continuous run of C1/C2/C3: (daily equity, valid signal dates)."""
    cost = config.COSTS["crypto"] / 2
    funding = 0.0
    if strategy == "C1":
        w = cs.c1_weights(closes, eligible)
    elif strategy == "C2":
        w, funding = cs.c2_weights(closes, eligible), C2_FUNDING
    elif strategy == "C3":
        w = cs.c3_weights(closes, eligible, fng, mode=mode)
    else:
        raise ValueError(f"not a crypto strategy: {strategy}")
    eq = run_weights(opens, closes, w, cost, cash_cols=(CASH,), short_funding_annual=funding)
    return eq, crypto_valid_dates(strategy, closes, eligible, fng)


def _s1_valid(bars, index_close, home, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Dates from the first day any stock and its home index both have 200 bars (SMA200)."""
    idx_start = {}
    for k, v in index_close.items():
        v = v.dropna()
        idx_start[k] = v.index[BTC_SMA - 1] if len(v) >= BTC_SMA else None
    starts = []
    for s, df in bars.items():
        c = df["close"].dropna()
        i0 = idx_start.get(home[s])
        if len(c) >= BTC_SMA and i0 is not None:
            starts.append(max(c.index[BTC_SMA - 1], i0))
    if not starts:
        return pd.DatetimeIndex([])
    return calendar[calendar >= min(starts)]


def s1_run(bars, index_close, home, cost: float
           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """S1 and its ``__all__`` random baseline: realized-P&L equity per seed (columns SEEDS).

    One continuous run over all dates; trade returns are net of ``cost`` (round trip) and clipped
    at the pooled 0.1/99.9 percentiles of every row of the run (same bounds for the baseline).
    """
    frame = ss.s1_frame(bars, index_close, home, cost, horizon=S1_HORIZON)
    col = f"ret_long_{S1_HORIZON}"
    v = frame[col].to_numpy(dtype=float)
    frame[col] = clip_values(v, clip_bounds(v))
    frame["asset"] = frame["asset"].astype("category")
    frame["split"] = pd.Categorical(["all"] * len(frame))

    def curves(rule):
        return pd.DataFrame({s: simulate(frame, rule, "long", "all", S1_HORIZON, S1_SLOTS, s)[1]
                             for s in SEEDS})

    eq, base = curves("s1"), curves(ALL_RULE)
    valid = _s1_valid(bars, index_close, home, eq.index)
    del frame
    gc.collect()
    return eq, base, valid


def s2_run(bars, index_close, home, cost_per_side: float
           ) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    """S2 and its same-universe equal-weight hold: (equity, benchmark equity, valid month-ends)."""
    opens, closes = _frames(bars)
    reg = ss.home_regime_frame(closes, index_close, home)
    eq = run_weights(opens, closes, ss.s2_weights(closes, reg, min_valid=S2_MIN_VALID), cost_per_side)
    del reg
    ewh = run_weights(opens, closes, ss.equal_weight_hold(closes, min_valid=S2_MIN_VALID), cost_per_side)
    mom = ss._momentum(closes)
    valid = mom.index[(mom.notna().sum(axis=1) >= S2_MIN_VALID).to_numpy()]
    return eq, ewh, valid


# ---------------------------------------------------------------- evaluation

def _slice(curve, start: pd.Timestamp, end: pd.Timestamp | None, pre: float | None = 1.0):
    """``curve`` from ``start`` through ``end`` re-based to 1 at ``start``.

    The base is the last value on or before ``start``. Before the curve exists it is ``pre``
    (1.0: an equity curve is flat cash before its first trade); with ``pre=None`` (a price
    series) the slice starts at the first value after ``start``.
    """
    if isinstance(curve, pd.DataFrame):
        return pd.DataFrame({c: _slice(curve[c], start, end, pre) for c in curve.columns})
    s = curve.dropna()
    after = s.loc[start:] if end is None else s.loc[start:end]
    before = s.loc[:start]
    if len(before):
        base = float(before.iloc[-1])
    elif pre is not None:
        base = pre
    else:
        return after / after.iloc[0] if len(after) else after
    after = after[after.index > start]
    return pd.concat([pd.Series([base], index=pd.DatetimeIndex([start])), after]) / base


def _monthly(curve):
    me = curve.resample("ME").last()
    prev = me.shift(1)
    prev.iloc[0] = curve.iloc[0]
    return (me / prev - 1.0).dropna(how="all")


def _summary(curve) -> tuple[dict, pd.Series]:
    """Metrics (median over seed columns for a DataFrame) and the monthly-return series."""
    keys = {"total_return": "total_return", "cagr": "cagr", "max_drawdown": "max_drawdown",
            "sharpe": "sharpe_monthly"}
    if isinstance(curve, pd.DataFrame):
        ms = [metrics(curve[c]) for c in curve.columns]
        return ({k: float(np.median([m[v] for m in ms])) for k, v in keys.items()},
                _monthly(curve).median(axis=1))
    m = metrics(curve)
    return {k: float(m[v]) for k, v in keys.items()}, _monthly(curve)


def _hac_t(diff: pd.Series) -> tuple[float, float]:
    d = diff.dropna().to_numpy(dtype=float)
    if len(d) < 6 or not np.std(d) > 0:
        return NAN, NAN
    fit = sm.OLS(d, np.ones((len(d), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return float(fit.tvalues[0]), float(fit.pvalues[0])


def _criteria(strategy: str, m: dict, b: dict, final: bool) -> list[str]:
    fails = []
    tr, sh = m["total_return"], m["sharpe"]
    if strategy in ("C1", "C3"):
        if not sh >= b["sharpe"]:
            fails.append("sharpe below BTC buy-and-hold")
        if not abs(m["max_drawdown"]) < abs(b["max_drawdown"]):
            fails.append("drawdown not smaller than BTC's")
        if final and not tr > 0:
            fails.append("total return <= 0")
    elif strategy == "C2":
        if not tr > 0:
            fails.append("total return <= 0")
        if not sh >= 0.5:
            fails.append("sharpe < 0.5")
    elif strategy == "S1":
        if not tr > 0:
            fails.append("median total return <= 0")
        if not tr > b["total_return"]:
            fails.append("median total return not above the random baseline")
    elif strategy == "S2":
        if not tr > 0:
            fails.append("total return <= 0")
        if not tr > b["total_return"]:
            fails.append("total return not above equal-weight hold")
        if not sh >= b["sharpe"]:
            fails.append("sharpe below equal-weight hold")
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return fails


def evaluate(strategy: str, equity, period: tuple, benchmarks: dict,
             valid_dates: pd.DatetimeIndex | None = None, final: bool = False) -> dict:
    """The spec's pass criteria for ``strategy`` on one period slice.

    The slice starts at the first ``valid_dates`` entry inside ``period`` (``None`` bounds are
    open; default: the equity's own dates) and is re-based to 1 there. ``equity`` is a Series, or
    for S1 a DataFrame of seed columns (medians are used). ``benchmarks``:
    ``btc`` (BTC closes, C1/C3), ``baseline`` (seed DataFrame, S1), ``ewh`` (equity, S2) and
    ``home`` ({index: closes}, information only). C2 is absolute. ``final`` adds C1/C3's
    total return > 0 criterion. Also an information-only HAC (lag 3) t-stat of monthly
    returns minus the benchmark's (C2: minus 0).
    """
    p0 = pd.Timestamp(period[0]) if period[0] is not None else None
    p1 = pd.Timestamp(period[1]) if period[1] is not None else None
    out = {"strategy": strategy, "start": None, "end": None, "n_days": 0,
           "total_return": NAN, "cagr": NAN, "max_drawdown": NAN, "sharpe": NAN,
           "bench_total_return": NAN, "bench_cagr": NAN, "bench_max_drawdown": NAN,
           "bench_sharpe": NAN, "t_stat": NAN, "t_p": NAN, "pass": False, "reason": ""}
    if "worst_short_loss" in getattr(equity, "attrs", {}):
        out["worst_short_loss_run"] = equity.attrs["worst_short_loss"]
    eq_idx = equity.dropna(how="all").index
    vd = pd.DatetimeIndex(valid_dates if valid_dates is not None else eq_idx)
    keep = np.ones(len(vd), dtype=bool)
    if p0 is not None:
        keep &= vd >= p0
    if p1 is not None:
        keep &= vd <= p1
    vd = vd[keep]
    vd = vd[vd < eq_idx[-1]] if len(eq_idx) else vd[:0]
    if not len(vd):
        out["reason"] = "no valid signal in period"
        return out
    start = vd[0]
    s = _slice(equity, start, p1)
    end = s.index[-1]
    m, monthly = _summary(s)
    out.update(start=start, end=end, n_days=int(len(s)), **m)

    bench = None
    if strategy in ("C1", "C3"):
        bench = _slice(benchmarks["btc"], start, end, pre=None)
    elif strategy == "S1":
        bench = _slice(benchmarks["baseline"], start, end)
    elif strategy == "S2":
        bench = _slice(benchmarks["ewh"], start, end)
    b = {"total_return": NAN, "cagr": NAN, "max_drawdown": NAN, "sharpe": NAN}
    b_monthly = pd.Series(0.0, index=monthly.index)
    if bench is not None and len(bench) > 1:
        b, b_monthly = _summary(bench)
    out.update({f"bench_{k}": v for k, v in b.items()})
    out["t_stat"], out["t_p"] = _hac_t(monthly - b_monthly.reindex(monthly.index))

    for name, close in benchmarks.get("home", {}).items():
        h = _slice(close, start, end, pre=None)
        hm = _summary(h)[0] if len(h) > 1 else {"total_return": NAN, "max_drawdown": NAN, "sharpe": NAN}
        for k in ("total_return", "max_drawdown", "sharpe"):
            out[f"{name}_{k}"] = hm[k]

    fails = _criteria(strategy, m, b, final)
    out["pass"] = not fails
    out["reason"] = "; ".join(fails)
    return out


def stock_verdict(passes: dict[str, bool]) -> str:
    """"confirmed" if passed in both Japan and Europe, "mixed" in one, else "not confirmed"."""
    n = sum(bool(passes.get(m, False)) for m in FINAL_MARKETS)
    return "confirmed" if n == len(FINAL_MARKETS) else ("mixed" if n else "not confirmed")


# ---------------------------------------------------------------- dev mode

def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _finite_or_none(v):
    return float(v) if v is not None and np.isfinite(v) else None


def choose_c3(opens, closes, eligible, fng, choice_path: Path) -> str:
    """C3's gate direction: the stored choice if ``choice_path`` exists, else the mode with the
    higher Sharpe on crypto discovery only (data truncated at the discovery end), written to
    ``choice_path`` before anything else happens."""
    if choice_path.exists():
        return str(json.loads(choice_path.read_text(encoding="utf-8"))["c3_mode"])
    p0, p1 = PERIODS["crypto"]["discovery"]
    cut = pd.Timestamp(p1)
    o, c, e, f = opens.loc[:cut], closes.loc[:cut], eligible.loc[:cut], fng.loc[:cut]
    sharpes = {}
    for mode in cs.GATES:
        eq, valid = run_crypto("C3", o, c, e, f, mode)
        sharpes[mode] = evaluate("C3", eq, (p0, p1), {"btc": c[BTC]}, valid_dates=valid)["sharpe"]
    pick = max(sharpes, key=lambda k: sharpes[k] if np.isfinite(sharpes[k]) else -np.inf)
    choice_path.parent.mkdir(parents=True, exist_ok=True)
    with open(choice_path, "x", encoding="utf-8") as fh:
        json.dump({"c3_mode": pick, "discovery_sharpes": {k: _finite_or_none(v) for k, v in sharpes.items()},
                   "discovery_period": [p0, p1], "utc": _now()}, fh, indent=2)
    return pick


def crypto_dev(opens, closes, eligible, fng, choice_path: Path) -> tuple[list[dict], dict]:
    """Discovery and validation rows for C1, C2, C3 and the crypto part of the gate."""
    btc = closes[BTC]
    rows, gate = [], {}

    def periods(name, eq, valid, extra=None):
        for label in ("discovery", "validation"):
            r = evaluate(name, eq, PERIODS["crypto"][label], {"btc": btc}, valid_dates=valid)
            r.update(market="crypto", period=label, **(extra or {}))
            rows.append(r)
        gate[name] = bool(rows[-1]["pass"])

    for name in ("C1", "C2"):
        periods(name, *run_crypto(name, opens, closes, eligible))
    mode = choose_c3(opens, closes, eligible, fng, choice_path)  # file exists from here on
    periods("C3", *run_crypto("C3", opens, closes, eligible, fng, mode), extra={"c3_mode": mode})
    return rows, gate


def stock_dev(bars, index_close, home, market: str = "stock") -> tuple[list[dict], dict]:
    """Discovery and validation rows for S1 and S2 on US data and the stock part of the gate."""
    cost = config.COSTS[market]
    rows, gate = [], {}
    eq, base, valid = s1_run(bars, index_close, home, cost)
    for label in ("discovery", "validation"):
        r = evaluate("S1", eq, PERIODS["stock"][label], {"baseline": base, "home": index_close},
                     valid_dates=valid)
        r.update(market="us", period=label)
        rows.append(r)
    gate["S1"] = bool(rows[-1]["pass"])
    del eq, base
    gc.collect()
    eq, ewh, valid = s2_run(bars, index_close, home, cost / 2)
    for label in ("discovery", "validation"):
        r = evaluate("S2", eq, PERIODS["stock"][label], {"ewh": ewh, "home": index_close},
                     valid_dates=valid)
        r.update(market="us", period=label)
        rows.append(r)
    gate["S2"] = bool(rows[-1]["pass"])
    return rows, gate


def _table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    first = [c for c in REPORT_COLS if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]]


def _write(df: pd.DataFrame, reports_dir: Path, name: str, header: list[str]) -> str:
    reports_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports_dir / f"{name}.csv", index=False)
    shown = df[[c for c in REPORT_COLS if c in df.columns]] if len(df) else df
    with pd.option_context("display.width", 250, "display.max_columns", 50,
                           "display.max_colwidth", 80):
        body = shown.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(no rows)"
    text = "\n".join(header + ["", body, ""])
    (reports_dir / f"{name}.txt").write_text(text, encoding="utf-8")
    return text


def _stamp(stamp: str | None) -> str:
    return stamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_dev(reports_dir: Path, stamp: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Dev mode: crypto (holdout locked) and US S1/S2 dev + validation; writes the gate."""
    stamp = _stamp(stamp)
    reports_dir.mkdir(parents=True, exist_ok=True)
    cut = pd.Timestamp(config.HOLDOUT_START)

    panel = pd.read_parquet(config.DATA_DIR / "panel" / "crypto.parquet",
                            columns=["date", "asset", "eligible"])
    panel = panel[panel["date"] < cut]
    opens, closes, eligible = crypto_inputs(panel, allow_holdout=False)
    del panel
    fng = foreign.load_fng()
    fng = fng[fng.index < cut]
    rows, gate = crypto_dev(opens, closes, eligible, fng, reports_dir / CHOICE_NAME)
    del opens, closes, eligible
    gc.collect()

    ensure_index(config.HOME_INDEX["us"], "stock")
    bars, index_close, home = us_stock_inputs()
    srows, sgate = stock_dev(bars, index_close, home)
    rows += srows
    gate.update(sgate)
    del bars
    gc.collect()

    (reports_dir / GATE_NAME).write_text(json.dumps(gate, indent=2), encoding="utf-8")
    choice = json.loads((reports_dir / CHOICE_NAME).read_text(encoding="utf-8"))
    df = _table(rows)
    header = [f"Original strategies: dev and validation ({stamp})",
              f"C3 direction (chosen on discovery only): {choice['c3_mode']} "
              f"(discovery Sharpes {choice.get('discovery_sharpes')})",
              "Validation gate: " + ", ".join(f"{k}={'PASS' if v else 'fail'}" for k, v in gate.items())]
    print(_write(df, reports_dir, f"originals_dev_{stamp}", header))
    return df, gate


# ---------------------------------------------------------------- final mode

def _jsonable(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, pd.Timestamp):
            v = f"{v:%Y-%m-%d}"
        elif isinstance(v, (bool, np.bool_)):
            v = bool(v)
        elif isinstance(v, (int, np.integer)):
            v = int(v)
        elif isinstance(v, (float, np.floating)):
            v = _finite_or_none(float(v))
        out[k] = v
    return out


def crypto_final(strategies: list[str], reports_dir: Path) -> list[dict]:
    """The one-time crypto 2024+ holdout run for ``strategies`` (all passed validation).

    Pre-flight checks the required files, refuses if the record exists, writes the record,
    then builds the panel with ``allow_holdout=True`` (every load logged). Results, or the
    error, are saved into the record.
    """
    record = config.DATA_DIR / RECORD_NAME
    need = [config.DATA_DIR / "universe_meta.parquet", store._path(BTC, "crypto")]
    if "C3" in strategies:
        need += [_fng_file(), reports_dir / CHOICE_NAME]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"missing files, crypto holdout NOT used: {missing}")
    if record.exists():
        raise SystemExit(f"crypto originals holdout already used: {record.read_text(encoding='utf-8')}")
    mode = (str(json.loads((reports_dir / CHOICE_NAME).read_text(encoding="utf-8"))["c3_mode"])
            if "C3" in strategies else None)
    payload = {"utc": _now(), "who": "scripts/originals.py --final",
               "strategies": list(strategies), "c3_mode": mode}
    record.parent.mkdir(parents=True, exist_ok=True)
    with open(record, "x", encoding="utf-8") as fh:  # exclusive: never overwrite a record
        json.dump(payload, fh, indent=2)
    try:
        panel = build_panel("crypto", allow_holdout=True)
        panel = panel[["date", "asset", "eligible"]].copy()
        gc.collect()
        opens, closes, eligible = crypto_inputs(panel, allow_holdout=True)
        del panel
        fng = foreign.load_fng() if "C3" in strategies else None
        btc = closes[BTC]
        rows = []
        for s in strategies:
            eq, valid = run_crypto(s, opens, closes, eligible, fng, mode)
            r = evaluate(s, eq, PERIODS["crypto"]["final"], {"btc": btc}, valid_dates=valid, final=True)
            r.update(market="crypto", period="final",
                     verdict="confirmed" if r["pass"] else "not confirmed")
            if s == "C3":
                r["c3_mode"] = mode
            rows.append(r)
        payload.update(end_date=f"{closes.index[-1]:%Y-%m-%d}", results=[_jsonable(r) for r in rows])
    except Exception as exc:  # noqa: BLE001 - the holdout is spent: always leave a record
        payload["error"] = f"{type(exc).__name__}: {exc}"
        rows = [{"market": "crypto", "strategy": s, "period": "final", "pass": False,
                 "verdict": "not confirmed", "reason": f"error: {payload['error']}"} for s in strategies]
    record.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return rows


def stock_final_market(strategy: str, market: str, inputs) -> dict:
    """Final-test row for S1 or S2 on one fresh market (full history, from the first valid date)."""
    bars, index_close, home = inputs
    cost = config.COSTS[market]
    if strategy == "S1":
        eq, base, valid = s1_run(bars, index_close, home, cost)
        bench = {"baseline": base}
    elif strategy == "S2":
        eq, ewh, valid = s2_run(bars, index_close, home, cost / 2)
        bench = {"ewh": ewh}
    else:
        raise ValueError(f"not a stock strategy: {strategy}")
    r = evaluate(strategy, eq, (None, None), {**bench, "home": index_close},
                 valid_dates=valid, final=True)
    r["n_stocks"] = len(bars)
    return r


def stock_final(strategies: list[str]) -> list[dict]:
    """Japan and Europe final tests for ``strategies`` with per-strategy verdicts.

    Runs only if both markets have data; otherwise nothing foreign is loaded.
    """
    missing = [m for m in FINAL_MARKETS if not _foreign_ready(m)]
    if missing:
        return [{"market": "+".join(FINAL_MARKETS), "strategy": s, "period": "final", "pass": False,
                 "verdict": "not run",
                 "reason": f"data missing for {missing}; no foreign data was loaded"} for s in strategies]
    rows, passes = [], {s: {} for s in strategies}
    for m in FINAL_MARKETS:
        inputs = foreign_inputs(m)
        for s in strategies:
            r = stock_final_market(s, m, inputs)
            r.update(market=m, period="final")
            rows.append(r)
            passes[s][m] = bool(r["pass"])
        del inputs
        gc.collect()
    for r in rows:
        r["verdict"] = stock_verdict(passes[r["strategy"]])
    return rows


def run_final(reports_dir: Path, stamp: str | None = None) -> pd.DataFrame:
    """Final mode: only strategies marked as passing in ``originals_gate.json``."""
    gate_path = reports_dir / GATE_NAME
    if not gate_path.exists():
        raise SystemExit(f"no validation gate at {gate_path}: run scripts/originals.py first")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    passed = [s for s in ALL if gate.get(s) is True]
    stamp = _stamp(stamp)
    rows: list[dict] = []
    crypto_s = [s for s in CRYPTO if s in passed]
    stock_s = [s for s in STOCKS if s in passed]
    if crypto_s:
        rows += crypto_final(crypto_s, reports_dir)
    if stock_s:
        rows += stock_final(stock_s)
    df = _table(rows) if rows else pd.DataFrame(columns=REPORT_COLS)
    skipped = [s for s in ALL if s not in passed]
    header = [f"Original strategies: final tests ({stamp})",
              "Run (passed validation): " + (", ".join(passed) or "none"),
              "Not run (failed validation): " + (", ".join(skipped) or "none"),
              "Five strategies were tested; a single pass could be luck. "
              "Stock passes are weaker evidence (today's index members: survivorship bias)."]
    print(_write(df, reports_dir, f"originals_final_{stamp}", header))
    return df
