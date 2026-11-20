import numpy as np
import pandas as pd
import pytest

from swingbot import config
from swingbot.data import binance, universe, yahoo
from swingbot.data.download import run
from swingbot.data.quality import check_bars
from swingbot.data.universe import AssetMeta

DAY_MS = 86_400_000


def _kline(open_ms):
    return [open_ms, "1.0", "2.0", "0.5", "1.5", "10.0", open_ms + DAY_MS - 1, "0", 0, "0", "0", "0"]


class FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def test_binance_paging(monkeypatch):
    start = int(pd.Timestamp("2021-01-01").timestamp() * 1000)
    first = [_kline(start + i * DAY_MS) for i in range(1000)]
    second = [_kline(start + (1000 + i) * DAY_MS) for i in range(3)]
    pages = [first, second]
    calls = []

    def fake_get(url, params=None, timeout=None, **kw):
        calls.append(params)
        return FakeResp(pages[len(calls) - 1])

    monkeypatch.setattr(binance.requests, "get", fake_get)
    monkeypatch.setattr(binance, "_today_ms", lambda: start + 1002 * DAY_MS)  # last row is current day
    df = binance.fetch_binance("BTCUSDT", start="2021-01-01")
    assert len(df) == 1002
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df.index.tz is None
    assert df["close"].dtype == float
    assert calls[1]["startTime"] == start + 1000 * DAY_MS
    assert calls[0]["interval"] == "1d" and calls[0]["limit"] == 1000


def test_yahoo_normalizes(monkeypatch):
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03"], name="Date")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["AAPL"]])
    raw = pd.DataFrame(np.arange(10, dtype=float).reshape(2, 5) + 1, index=idx, columns=cols)
    monkeypatch.setattr(yahoo.yf, "download", lambda *a, **k: raw)
    df = yahoo.fetch_yahoo("AAPL")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"
    assert len(df) == 2
    assert df["open"].dtype == float


def test_yahoo_empty(monkeypatch):
    monkeypatch.setattr(yahoo.yf, "download", lambda *a, **k: pd.DataFrame())
    assert yahoo.fetch_yahoo("ZZZZ").empty


def test_crypto_universe_filters(monkeypatch, tmp_path):
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    tickers = [
        {"symbol": "USDCUSDT", "quoteVolume": "9e9"},
        {"symbol": "BTCUPUSDT", "quoteVolume": "8e9"},
        {"symbol": "ETHBTC", "quoteVolume": "7e9"},
        {"symbol": "BTCUSDT", "quoteVolume": "6e9"},
    ]
    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: FakeResp(tickers))
    out = universe.crypto_universe(60)
    assert [a.symbol for a in out] == ["BTCUSDT"]
    assert out[0].is_major and out[0].market == "crypto" and out[0].sector == "crypto"
    assert (tmp_path / "crypto.csv").exists()


def _bars(dates, **over):
    n = len(dates)
    df = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 100.0},
                      index=pd.DatetimeIndex(dates, name="date"))
    for k, v in over.items():
        df[k] = v
    return df


def test_check_bars_flags():
    assert check_bars(_bars(pd.bdate_range("2020-01-01", periods=30)), "stock") == []
    dates = list(pd.bdate_range("2020-01-01", periods=10))
    dates += list(pd.bdate_range("2020-02-03", periods=5))  # gap > 5 days
    dates.append(dates[3])  # duplicate
    df = _bars(dates)
    df.iloc[2, df.columns.get_loc("high")] = 5.0  # high < low
    issues = check_bars(df, "stock")
    assert any("duplicate" in s for s in issues)
    assert any("gap" in s for s in issues)
    assert any("high < low" in s for s in issues)


def test_check_bars_other_flags():
    d = pd.bdate_range("2020-01-01", periods=20)
    df = _bars(d, volume=0.0)
    assert any("zero-volume" in s for s in check_bars(df, "stock"))
    df = _bars(d)
    df.iloc[5, df.columns.get_loc("close")] = 30.0
    assert any("split" in s for s in check_bars(df, "stock"))
    df = _bars(d)
    df.iloc[5, df.columns.get_loc("open")] = -1.0
    assert any("non-positive" in s for s in check_bars(df, "stock"))
    # crypto gap of 2 days flagged
    df = _bars(pd.DatetimeIndex(["2020-01-01", "2020-01-03"]))
    assert any("gap" in s for s in check_bars(df, "crypto"))


def test_download_skips_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assets = [AssetMeta(s, "stock", "tech", "large", False, False) for s in ("GOOD", "BOOM", "EMPTY")]
    good = _bars(pd.bdate_range("2020-01-01", periods=30))

    def fetch(a):
        if a.symbol == "BOOM":
            raise RuntimeError("net down")
        if a.symbol == "EMPTY":
            return pd.DataFrame()
        return good

    msgs = run(assets, fetch)
    skips = [m for m in msgs if m.startswith("SKIP")]
    assert len(skips) == 2
    assert (tmp_path / "raw" / "stock" / "GOOD.parquet").exists()
    assert not (tmp_path / "raw" / "stock" / "BOOM.parquet").exists()
    assert (tmp_path / "quality_report.csv").exists()
