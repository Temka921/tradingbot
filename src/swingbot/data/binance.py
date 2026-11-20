import time

import pandas as pd
import requests

from swingbot.data.store import COLUMNS

BASE_URL = "https://api.binance.com"
DAY_MS = 86_400_000


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
