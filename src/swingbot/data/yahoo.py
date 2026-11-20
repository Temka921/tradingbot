import pandas as pd
import yfinance as yf

from swingbot.data.store import COLUMNS


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="date"), dtype=float)


def fetch_yahoo(ticker: str, start: str = "2000-01-01") -> pd.DataFrame:
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        return _empty()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    df = raw[COLUMNS].astype(float).dropna(subset=["close"])
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize().rename("date")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df
