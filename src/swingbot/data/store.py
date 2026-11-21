from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from swingbot import config

COLUMNS = ["open", "high", "low", "close", "volume"]


def _path(symbol: str, market: str) -> Path:
    return config.DATA_DIR / "raw" / market / f"{symbol}.parquet"


def save_bars(symbol: str, market: str, df: pd.DataFrame) -> Path:
    path = _path(symbol, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def load_bars(symbol: str, market: str, allow_holdout: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(_path(symbol, market))
    if market not in config.HOLDOUT_MARKETS:
        return df
    if allow_holdout:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(config.DATA_DIR / "holdout_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{stamp} holdout access {market}/{symbol}\n")
        return df
    return df[df.index < pd.Timestamp(config.HOLDOUT_START)]


def split_of(dates: pd.DatetimeIndex, market: str) -> pd.Series:
    out = pd.Series("pre", index=dates, dtype=object)
    for name, (start, end) in config.SPLITS[market].items():
        out[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))] = name
    out[dates >= pd.Timestamp(config.HOLDOUT_START)] = "holdout"
    return out
