import io
import time
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests

from swingbot.data.store import COLUMNS

BASE_URL = "https://api.binance.com"
DAY_MS = 86_400_000
ARCHIVE_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE_FILE_URL = "https://data.binance.vision/"
ARCHIVE_SPOT_PREFIX = "data/spot/monthly/klines/"


def _today_ms() -> int:
    """Open time (ms, UTC) of the current, still-open daily candle."""
    now = int(time.time() * 1000)
    return now - now % DAY_MS


def fetch_binance(symbol: str, start: str = "2017-01-01") -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows: list[list] = []
    while True:
        resp = requests.get(
            f"{BASE_URL}/api/v3/klines",
            params={"symbol": symbol, "interval": "1d", "limit": 1000, "startTime": start_ms},
            timeout=30,
        )
        resp.raise_for_status()
        page = resp.json()
        rows.extend(page)
        if len(page) < 1000:
            break
        start_ms = int(page[-1][0]) + DAY_MS
    if not rows:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="date"), dtype=float)
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["open_time"] + COLUMNS
    df = df[df["open_time"].astype("int64") < _today_ms()]  # drop open current-day candle
    idx = pd.to_datetime(df["open_time"].astype("int64"), unit="ms").dt.normalize()
    out = df[COLUMNS].astype(float)
    out.index = pd.DatetimeIndex(idx, name="date")
    return out[~out.index.duplicated(keep="last")].sort_index()


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="date"), dtype=float)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _list_archive(prefix: str, delimiter: str | None, tag: str, child: str) -> list[str]:
    """Paginated S3 listing; returns the text of <child> under every <tag> element."""
    out: list[str] = []
    marker: str | None = None
    while True:
        params = {"prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if marker:
            params["marker"] = marker
        resp = requests.get(ARCHIVE_LIST_URL, params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        truncated, next_marker, last = False, None, None
        for el in root.iter():
            name = _local(el.tag)
            if name == "IsTruncated":
                truncated = (el.text or "").strip().lower() == "true"
            elif name == "NextMarker":
                next_marker = (el.text or "").strip() or None
            elif name == tag:
                for c in el:
                    if _local(c.tag) == child and c.text:
                        out.append(c.text)
                        last = c.text
        if not truncated:
            return out
        marker = next_marker or last
        if marker is None:
            return out


def list_archive_symbols(quote: str = "USDT") -> list[str]:
    """Every spot symbol in the Binance public archive (live and delisted) ending in `quote`."""
    prefixes = _list_archive(ARCHIVE_SPOT_PREFIX, "/", "CommonPrefixes", "Prefix")
    syms = {p[len(ARCHIVE_SPOT_PREFIX):].strip("/") for p in prefixes}
    return sorted(s for s in syms if s.endswith(quote))


def fetch_binance_archive(symbol: str, start: str = "2017-01-01") -> pd.DataFrame:
    """Daily bars from the monthly archive zips (works for delisted pairs)."""
    prefix = f"{ARCHIVE_SPOT_PREFIX}{symbol}/1d/"
    keys = [k for k in _list_archive(prefix, None, "Contents", "Key") if k.endswith(".zip")]
    frames = []
    for key in sorted(keys):
        resp = requests.get(ARCHIVE_FILE_URL + key, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    df = pd.read_csv(f, header=None, usecols=range(6))
                frames.append(df)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty_bars()
    df = pd.concat(frames, ignore_index=True)
    df.columns = ["open_time"] + COLUMNS
    t = df["open_time"].astype("int64")
    t = t.where(t <= 10**14, t // 1000)  # microseconds (2025+) -> ms
    idx = pd.to_datetime(t, unit="ms").dt.normalize()
    out = df[COLUMNS].astype(float)
    out.index = pd.DatetimeIndex(idx, name="date")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[out.index >= pd.Timestamp(start)]


def fetch_crypto(symbol: str) -> pd.DataFrame:
    """Live API; falls back to the archive for delisted/invalid symbols (HTTP 400)."""
    try:
        return fetch_binance(symbol)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 400:
            return fetch_binance_archive(symbol)
        raise
