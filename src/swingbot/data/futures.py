"""Binance USD-M perpetual futures data: funding rates and daily perp klines.

Monthly archive zips (data.binance.vision) cover history; the fapi API fills the months after the
archive. Reuses the archive listing / zip / timestamp helpers of swingbot.data.binance.
"""
from datetime import datetime, timezone

import pandas as pd
import requests

from swingbot import config
from swingbot.data import binance, store
from swingbot.data.universe import crypto_universe

FAPI_URL = "https://fapi.binance.com"
FUNDING_PREFIX = "data/futures/um/monthly/fundingRate/"
KLINES_PREFIX = "data/futures/um/monthly/klines/"
API_LIMIT = 1000          # fundingRate page size (max 1000)
KLINES_API_LIMIT = 1500   # klines page size (max 1500)
EARLIEST = "2019-09-01"   # first USD-M perp listing; API start when no archive exists
FUNDING_COLS = ["funding_time", "rate"]


def _empty_funding() -> pd.DataFrame:
    return pd.DataFrame({"funding_time": pd.Series(dtype="datetime64[ns]"), "rate": pd.Series(dtype=float)})


def list_um_symbols() -> list[str]:
    """USDT-quoted perpetual symbols in the funding archive (dated futures like X_240628 never end in USDT)."""
    return binance.list_archive_symbols("USDT", prefix=FUNDING_PREFIX)


def futures_universe() -> list[str]:
    """USDT perps whose symbol is in universe/crypto.csv (so its base passed the crypto exclusions)."""
    allowed = {a.symbol for a in crypto_universe()}
    return [s for s in list_um_symbols() if s in allowed]


def _funding_frame(times_ms: pd.Series, rates: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"t": pd.to_numeric(times_ms, errors="coerce"), "rate": pd.to_numeric(rates, errors="coerce")})
    df = df.dropna()
    if df.empty:
        return _empty_funding()
    out = pd.DataFrame({"funding_time": pd.to_datetime(binance.epoch_to_ms(df["t"]), unit="ms"),
                        "rate": df["rate"].astype(float)})
    return out


def _tidy_funding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates("funding_time", keep="last").sort_values("funding_time")
    return df.reset_index(drop=True)


def _funding_api(symbol: str, start_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    while True:
        resp = requests.get(f"{FAPI_URL}/fapi/v1/fundingRate",
                            params={"symbol": symbol, "startTime": start_ms, "limit": API_LIMIT}, timeout=30)
        resp.raise_for_status()
        page = resp.json()
        rows.extend(page)
        if len(page) < API_LIMIT:
            break
        start_ms = int(page[-1]["fundingTime"]) + 1
    if not rows:
        return _empty_funding()
    return _funding_frame(pd.Series([r["fundingTime"] for r in rows]), pd.Series([r["fundingRate"] for r in rows]))


def fetch_funding(symbol: str) -> pd.DataFrame:
    """Funding events (`funding_time` tz-naive UTC, `rate`): archive months, then the API after the last one."""
    frames = []
    for raw in binance.archive_frames(f"{FUNDING_PREFIX}{symbol}/"):
        # columns: calc_time, [funding_interval_hours,] last_funding_rate (header row in newer files):
        # time = first column, rate = last column; non-numeric (header) rows are dropped.
        frames.append(_funding_frame(raw.iloc[:, 0], raw.iloc[:, -1]))
    archive = _tidy_funding(pd.concat(frames, ignore_index=True)) if frames else _empty_funding()
    if archive.empty:
        start_ms = int(pd.Timestamp(EARLIEST, tz="UTC").timestamp() * 1000)
    else:
        start_ms = int(archive["funding_time"].iloc[-1].tz_localize("UTC").timestamp() * 1000) + 1
    tail = _funding_api(symbol, start_ms)
    both = pd.concat([archive, tail], ignore_index=True)
    return _tidy_funding(both) if not both.empty else _empty_funding()


def _klines_api(symbol: str, start_ms: int) -> pd.DataFrame:
    rows: list[list] = []
    while True:
        resp = requests.get(f"{FAPI_URL}/fapi/v1/klines",
                            params={"symbol": symbol, "interval": "1d", "startTime": start_ms,
                                    "limit": KLINES_API_LIMIT}, timeout=30)
        resp.raise_for_status()
        page = resp.json()
        rows.extend(page)
        if len(page) < KLINES_API_LIMIT:
            break
        start_ms = int(page[-1][0]) + binance.DAY_MS
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).iloc[:, :6]


def fetch_perp_klines(symbol: str, start: str = "2019-09-01") -> pd.DataFrame:
    """Daily perp bars: archive months, then the API after the last archived day; open day dropped."""
    frames = binance.archive_frames(f"{KLINES_PREFIX}{symbol}/1d/")
    archive = (binance.bars_from_klines(pd.concat(frames, ignore_index=True).iloc[:, :6], start)
               if frames else binance._empty_bars())
    if archive.empty:
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    else:
        start_ms = int(archive.index[-1].tz_localize("UTC").timestamp() * 1000) + binance.DAY_MS
    api = _klines_api(symbol, start_ms)
    if not api.empty:
        api = api[pd.to_numeric(api.iloc[:, 0]).astype("int64") < binance._today_ms()]  # drop the open day
    tail = binance.bars_from_klines(api, start) if not api.empty else binance._empty_bars()
    out = pd.concat([archive, tail])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[out.index < pd.Timestamp(binance._today_ms(), unit="ms")]


def _funding_path(symbol: str):
    return config.DATA_DIR / "raw" / "funding" / f"{symbol}.parquet"


def save_funding(symbol: str, df: pd.DataFrame):
    path = _funding_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def load_funding(symbol: str, allow_holdout: bool = False) -> pd.DataFrame:
    """Funding events. Rows from HOLDOUT_START on need allow_holdout=True, which is logged."""
    df = pd.read_parquet(_funding_path(symbol))
    if allow_holdout:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(config.DATA_DIR / "holdout_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{stamp} holdout access funding/{symbol}\n")
        return df
    return df[df["funding_time"] < pd.Timestamp(config.HOLDOUT_START)].reset_index(drop=True)


def daily_funding(df: pd.DataFrame) -> pd.Series:
    """Funding earned over each UTC day (index `date`, days with no event are absent).

    A payment at exactly 00:00 settles the period that ended at the previous day's 24:00 close,
    so it belongs to the previous day: day d holds the payments after 00:00 on d up to and
    including 00:00 on d+1 (e.g. 08:00 and 16:00 on d plus 00:00 on d+1). This lines up day-d
    returns with positions actually held over those timestamps (a position opened at d's open
    does not collect d's 00:00 payment).
    """
    if df.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="date"), name="funding")
    day = (df["funding_time"] - pd.Timedelta("1ms")).dt.normalize()
    s = df["rate"].groupby(day).sum()
    s.index.name = "date"
    s.name = "funding"
    return s.sort_index()


def download_symbol(symbol: str) -> str:
    """Fetch and save funding + perp klines for one symbol; returns a log line."""
    f = fetch_funding(symbol)
    k = fetch_perp_klines(symbol)
    if f.empty or k.empty:
        return f"SKIP {symbol}: empty funding ({len(f)}) or perp bars ({len(k)})"
    save_funding(symbol, f)
    store.save_bars(symbol, "perp", k)
    return f"OK {symbol}: {len(f)} funding rows, {len(k)} perp bars"
