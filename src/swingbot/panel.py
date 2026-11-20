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


def build_panel(market: str, allow_holdout: bool = False) -> pd.DataFrame:
    meta = _eligible_assets(market)
    bench_sym = config.BENCHMARK[market]
    if bench_sym not in set(meta["symbol"]):
        raise ValueError(f"benchmark {bench_sym} missing from {market} data")
    bench = store.load_bars(bench_sym, market, allow_holdout=allow_holdout)

    closes = pd.DataFrame({
        s: store.load_bars(s, market, allow_holdout=allow_holdout)["close"] for s in meta["symbol"]
    })
    xs = cross_sectional_signals(closes)  # once per market, this market's closes only
    del closes

    parts = []
    for r in meta.itertuples():
        df = store.load_bars(r.symbol, market, allow_holdout=allow_holdout)
        sig = compute_signals(df)
        for name, frame in xs.items():
            sig[name] = frame[r.symbol].reindex(df.index).fillna(False).astype(bool)
        part = pd.concat([sig, compute_outcomes(df, bench, market), label_regimes(df, bench)], axis=1)
        part.insert(0, "asset", r.symbol)
        for col in META_COLS:
            part[col] = getattr(r, col)
        part["market"] = market
        part["split"] = store.split_of(df.index, market)
        part = part[part["split"] != "pre"]
        parts.append(part.reset_index())  # 'date' index -> column
    out = pd.concat(parts, ignore_index=True)
    cols = ["date", "asset"] + [c for c in out.columns if c not in ("date", "asset")]
    out = out[cols]
    for c in CATEGORICAL:
        out[c] = out[c].astype("category")
    return out



