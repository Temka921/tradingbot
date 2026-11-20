"""Assemble signals, outcomes, regimes and metadata into one long panel per market."""
import pandas as pd

from swingbot import config
from swingbot.data import store
from swingbot.data.universe import excluded_crypto_base
from swingbot.outcomes import compute_outcomes
from swingbot.regimes import label_regimes
from swingbot.signals import compute_signals, cross_sectional_signals

META_COLS = ["sector", "cap_bucket", "is_etf", "is_major"]
CATEGORICAL = ["trend_regime", "vol_regime", "market_regime", "adx_regime",
               "sector", "cap_bucket", "market", "split"]


def _skip(symbol: str, reason: str) -> None:
    print(f"SKIP {symbol}: {reason}")


def _eligible_assets(market: str) -> pd.DataFrame:
    meta = pd.read_parquet(config.DATA_DIR / "universe_meta.parquet")
    meta = meta[meta["market"] == market].drop_duplicates("symbol", keep="last")
    keep = []
    for r in meta.itertuples():
        if market == "crypto" and excluded_crypto_base(r.symbol.removesuffix("USDT")):
            _skip(r.symbol, "excluded crypto base")
        elif not store._path(r.symbol, market).exists():
            _skip(r.symbol, "no data file")
        else:
            keep.append(r.symbol)
    return meta[meta["symbol"].isin(keep)].reset_index(drop=True)


def build_panel(market: str, allow_holdout: bool = False,
                max_skip_frac: float | None = None) -> pd.DataFrame:
    """One long panel for ``market``.

    No rows are dropped on price level (Yahoo closes are back-adjusted, so a low
    adjusted price at t selects later winners). Raises if more than ``max_skip_frac`` (default
    ``config.MAX_SKIP_FRAC``) of the market's assets with a data file are skipped
    because of an error. Expected skips (excluded crypto base, no rows left after
    dropping the pre-period) do not count.
    """
    max_skip_frac = config.MAX_SKIP_FRAC if max_skip_frac is None else max_skip_frac
    meta = _eligible_assets(market)
    bench_sym = config.BENCHMARK[market]
    if bench_sym not in set(meta["symbol"]):
        raise ValueError(f"benchmark {bench_sym} missing from {market} data")
    failed: list[str] = []
    expected = 0

    # Load every asset exactly once (one holdout-log entry per asset).
    bars: dict[str, pd.DataFrame] = {}
    for sym in meta["symbol"]:
        try:
            bars[sym] = store.load_bars(sym, market, allow_holdout=allow_holdout)
        except Exception as e:  # corrupt/unreadable file
            if sym == bench_sym:
                raise
            _skip(sym, f"{type(e).__name__}: {e}")
            failed.append(sym)
    bench = bars[bench_sym]

    closes = pd.DataFrame({s: d["close"] for s, d in bars.items()})
    xs = cross_sectional_signals(closes)  # once per market, this market's closes only
    del closes

    parts = []
    for r in meta.itertuples():
        if r.symbol not in bars:
            continue
        df = bars.pop(r.symbol)
        try:
            sig = compute_signals(df)
            for name, frame in xs.items():
                sig[name] = frame[r.symbol].reindex(df.index, fill_value=False).astype(bool)
            part = pd.concat([sig, compute_outcomes(df, bench, market), label_regimes(df, bench)], axis=1)
            part.insert(0, "asset", r.symbol)
            for col in META_COLS:
                part[col] = getattr(r, col)
            part["market"] = market
            part["split"] = store.split_of(df.index, market)
            part = part[part["split"] != "pre"]
            if part.empty:
                _skip(r.symbol, "no rows after dropping pre-period")
                expected += 1
                continue
            f64 = part.select_dtypes("float64").columns
            part[f64] = part[f64].astype("float32")
            parts.append(part.reset_index())  # 'date' index -> column
        except Exception as e:
            _skip(r.symbol, f"{type(e).__name__}: {e}")
            failed.append(r.symbol)
    n_assets = len(meta)
    msg = f"{market}: {len(failed)} of {n_assets} assets skipped for errors, {expected} expected skips"
    print(msg)
    if n_assets and len(failed) / n_assets > max_skip_frac:
        raise ValueError(f"{len(failed)} of {n_assets} {market} assets ({len(failed) / n_assets:.1%}) "
                         f"skipped for errors, above {max_skip_frac:.0%}: {failed[:20]}")
    if not parts:
        raise ValueError(f"no assets could be built for market {market!r}")
    out = pd.concat(parts, ignore_index=True)
    del parts
    cols = ["date", "asset"] + [c for c in out.columns if c not in ("date", "asset")]
    out = out[cols]
    for c in CATEGORICAL:
        out[c] = out[c].astype("category")
    out.attrs["build_stats"] = {"assets_attempted": n_assets, "skipped_errors": len(failed),
                                "skipped_expected": expected}
    return out
