import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from swingbot import config
from swingbot.data import binance, download, universe, yahoo
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


class FakeText:
    def __init__(self, text="", content=b""):
        self.text = text
        self.content = content

    def raise_for_status(self):
        pass


class FakeStatus:
    def __init__(self, code):
        self.status_code = code


def _no_archive(monkeypatch, symbols=()):
    monkeypatch.setattr(universe.binance, "list_archive_symbols", lambda quote="USDT": list(symbols))


def test_crypto_universe_filters(monkeypatch, tmp_path):
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    _no_archive(monkeypatch)
    tickers = [
        {"symbol": "USDCUSDT", "quoteVolume": "9e9"},
        {"symbol": "BTCUPUSDT", "quoteVolume": "8e9"},
        {"symbol": "ETHBTC", "quoteVolume": "7e9"},
        {"symbol": "BTCUSDT", "quoteVolume": "6e9"},
        {"symbol": "USD1USDT", "quoteVolume": "5e9"},
        {"symbol": "PYUSDUSDT", "quoteVolume": "4e9"},
        {"symbol": "XUSDUSDT", "quoteVolume": "3e9"},
    ]
    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: FakeResp(tickers))
    out = universe.crypto_universe()
    assert [a.symbol for a in out] == ["BTCUSDT"]
    assert out[0].is_major and out[0].market == "crypto" and out[0].sector == "crypto"
    assert (tmp_path / "crypto.csv").exists()


def test_list_archive_symbols_paginates(monkeypatch):
    pages = [
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>true</IsTruncated>'
        '<NextMarker>data/spot/monthly/klines/ETHBTC/</NextMarker>'
        '<CommonPrefixes><Prefix>data/spot/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>'
        '<CommonPrefixes><Prefix>data/spot/monthly/klines/ETHBTC/</Prefix></CommonPrefixes></ListBucketResult>',
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>'
        '<CommonPrefixes><Prefix>data/spot/monthly/klines/LUNAUSDT/</Prefix></CommonPrefixes></ListBucketResult>']
    calls = []

    def get(url, params=None, timeout=None, **k):
        calls.append(params.get("marker"))
        return FakeText(pages[len(calls) - 1])

    monkeypatch.setattr(binance.requests, "get", get)
    assert binance.list_archive_symbols() == ["BTCUSDT", "LUNAUSDT"]
    assert calls == [None, "data/spot/monthly/klines/ETHBTC/"]


def _zip_bytes(rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", "\n".join(",".join(str(c) for c in r) for r in rows))
    return buf.getvalue()


def _arow(open_time, close="1.5"):
    return [open_time, "1.0", "2.0", "0.5", close, "10.0", open_time, "0", 0, "0", "0", "0"]


def test_fetch_binance_archive_ms_and_us(monkeypatch):
    ms = lambda d: int(pd.Timestamp(d).timestamp() * 1000)  # noqa: E731
    z1 = _zip_bytes([_arow(ms("2021-01-01")), _arow(ms("2021-01-02"))])
    z2 = _zip_bytes([_arow(ms("2025-01-01") * 1000)])  # microseconds
    base = "data/spot/monthly/klines/LUNAUSDT/1d/"
    keys = [base + "LUNAUSDT-1d-2021-01.zip", base + "LUNAUSDT-1d-2021-01.zip.CHECKSUM",
            base + "LUNAUSDT-1d-2025-01.zip"]
    zips = {binance.ARCHIVE_FILE_URL + keys[0]: z1, binance.ARCHIVE_FILE_URL + keys[2]: z2}
    listing = ('<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>'
               + "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys) + "</ListBucketResult>")

    def get(url, params=None, timeout=None, **k):
        if url == binance.ARCHIVE_LIST_URL:
            assert params["prefix"] == base
            return FakeText(listing)
        return FakeText(content=zips[url])  # KeyError if a CHECKSUM file is requested

    monkeypatch.setattr(binance.requests, "get", get)
    df = binance.fetch_binance_archive("LUNAUSDT")
    assert list(df.index) == list(pd.to_datetime(["2021-01-01", "2021-01-02", "2025-01-01"]))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"] and df.index.name == "date"
    assert df["close"].dtype == float


def test_fetch_binance_archive_header_row_and_start(monkeypatch):
    ms = lambda d: int(pd.Timestamp(d).timestamp() * 1000)  # noqa: E731
    header = "open_time,open,high,low,close,volume,close_time,qav,trades,tbb,tbq,ignore\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", header + "\n".join(
            ",".join(str(c) for c in _arow(ms(d))) for d in ["2016-12-31", "2021-03-01", "2021-03-02"]))
    key = "data/spot/monthly/klines/HDRUSDT/1d/HDRUSDT-1d-2021-03.zip"
    listing = ('<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>'
               f"<Contents><Key>{key}</Key></Contents></ListBucketResult>")
    monkeypatch.setattr(binance.requests, "get", lambda url, **k: FakeText(
        listing if url == binance.ARCHIVE_LIST_URL else "", content=buf.getvalue()))
    df = binance.fetch_binance_archive("HDRUSDT")  # 2016-12-31 is before the default start
    assert list(df.index) == list(pd.to_datetime(["2021-03-01", "2021-03-02"]))
    assert df["close"].dtype == float
    df2 = binance.fetch_binance_archive("HDRUSDT", start="2021-03-02")
    assert list(df2.index) == list(pd.to_datetime(["2021-03-02"]))


def test_fetch_binance_archive_empty_and_start(monkeypatch):
    empty = '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated></ListBucketResult>'
    monkeypatch.setattr(binance.requests, "get", lambda *a, **k: FakeText(empty))
    df = binance.fetch_binance_archive("NOPEUSDT")
    assert df.empty and list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"


def test_fetch_crypto_falls_back_on_400(monkeypatch):
    def api(*a, **k):
        raise requests.HTTPError(response=FakeStatus(400))

    monkeypatch.setattr(binance, "fetch_binance", api)
    monkeypatch.setattr(binance, "fetch_binance_archive", lambda s, start="2017-01-01": "ARCHIVE")
    assert binance.fetch_crypto("LUNAUSDT") == "ARCHIVE"

    def api500(*a, **k):
        raise requests.HTTPError(response=FakeStatus(500))

    monkeypatch.setattr(binance, "fetch_binance", api500)
    with pytest.raises(requests.HTTPError):
        binance.fetch_crypto("BTCUSDT")


def test_crypto_universe_includes_delisted_and_excludes_fiat(monkeypatch, tmp_path):
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    monkeypatch.setattr(universe.binance, "list_archive_symbols",
                        lambda quote="USDT": ["BTCUSDT", "LUNAUSDT", "EURUSDT", "USTUSDT", "BTCUPUSDT"])
    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: FakeResp(
        [{"symbol": "ETHUSDT", "quoteVolume": "1"}, {"symbol": "ARBUSDT", "quoteVolume": "1"}]))
    out = [a.symbol for a in universe.crypto_universe()]
    assert out == ["BTCUSDT", "ETHUSDT", "ARBUSDT", "LUNAUSDT"]


def test_default_fetch_crypto_uses_fetch_crypto(monkeypatch):
    monkeypatch.setattr(download, "fetch_crypto", lambda s: "X" + s)
    assert download.default_fetch(AssetMeta("LUNAUSDT", "crypto", "crypto", "alt", False, False)) == "XLUNAUSDT"


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
    monkeypatch.setattr(download, "_sleep", lambda s: None)
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
    assert "SKIP BOOM: RuntimeError: net down" in skips
    assert "SKIP EMPTY: empty frame" in skips
    assert (tmp_path / "raw" / "stock" / "GOOD.parquet").exists()
    assert not (tmp_path / "raw" / "stock" / "BOOM.parquet").exists()
    rep = pd.read_csv(tmp_path / "quality_report.csv", keep_default_na=False)
    assert rep["symbol"].tolist() == ["GOOD"] and rep["market"].tolist() == ["stock"]
    assert rep["issues"].tolist() == [""]
    meta = pd.read_parquet(tmp_path / "universe_meta.parquet")
    assert meta["symbol"].tolist() == ["GOOD"]
    assert meta.iloc[0].to_dict() == {"symbol": "GOOD", "market": "stock", "sector": "tech",
                                      "cap_bucket": "large", "is_etf": False, "is_major": False}


def test_with_benchmark_always_included():
    a = [AssetMeta(s, "stock", "x", "large", False, False) for s in ("AAA", "BBB", "SPY", "CCC")]
    out = universe.with_benchmark(a, "stock", 2)
    assert [x.symbol for x in out] == ["SPY", "AAA", "BBB"]
    c = [AssetMeta("ETHUSDT", "crypto", "crypto", "major", False, True),
         AssetMeta("BTCUSDT", "crypto", "crypto", "major", False, True)]
    assert universe.with_benchmark(c, "crypto", 1)[0].symbol == "BTCUSDT"
    with pytest.raises(ValueError):
        universe.with_benchmark(c[:1], "crypto", 1)


def test_reports_upsert_across_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    good = _bars(pd.bdate_range("2020-01-01", periods=30))
    s = AssetMeta("AAA", "stock", "tech", "large", False, False)
    c = AssetMeta("BTCUSDT", "crypto", "crypto", "major", False, True)
    run([s], lambda a: good)
    run([c], lambda a: good)
    meta = pd.read_parquet(tmp_path / "universe_meta.parquet")
    assert set(meta["symbol"]) == {"AAA", "BTCUSDT"}
    rep = pd.read_csv(tmp_path / "quality_report.csv")
    assert set(rep["symbol"]) == {"AAA", "BTCUSDT"}
    assert dict(zip(rep["symbol"], rep["market"])) == {"AAA": "stock", "BTCUSDT": "crypto"}
    # rerunning an asset replaces its row rather than duplicating it
    run([s], lambda a: good)
    assert len(pd.read_csv(tmp_path / "quality_report.csv")) == 2
    assert len(pd.read_parquet(tmp_path / "universe_meta.parquet")) == 2


def test_save_failure_skips_and_reports_written(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    good = _bars(pd.bdate_range("2020-01-01", periods=30))
    real = download.store.save_bars

    def bad_save(sym, market, df):
        if sym == "BAD":
            raise OSError("disk")
        return real(sym, market, df)

    monkeypatch.setattr(download.store, "save_bars", bad_save)
    assets = [AssetMeta(s, "stock", "t", "large", False, False) for s in ("BAD", "OK1")]
    msgs = run(assets, lambda a: good)
    assert sum(m.startswith("SKIP BAD") for m in msgs) == 1
    assert (tmp_path / "quality_report.csv").exists()
    assert (tmp_path / "raw" / "stock" / "OK1.parquet").exists()


def test_retry_then_success(monkeypatch):
    sleeps = []
    monkeypatch.setattr(download, "_sleep", sleeps.append)
    n = {"c": 0}

    def flaky(a):
        n["c"] += 1
        if n["c"] < 3:
            raise RuntimeError("429")
        return _bars(pd.bdate_range("2020-01-01", periods=5))

    a = AssetMeta("X", "stock", "t", "large", False, False)
    assert not download.fetch_with_retry(flaky, a).empty
    assert n["c"] == 3 and sleeps == [1, 2]


def _tick(sym, qv):
    return {"symbol": sym, "quoteVolume": str(qv)}


def test_crypto_universe_exclusions_and_order(monkeypatch, tmp_path):
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    bad = ["USD1", "RLUSD", "XUSD", "RWUSD", "U", "WBTC", "PAXG", "XAUT", "NVDAB", "TQQQB", "QQQB",
           "MUB", "牛来", "ETHDOWN", "BNBBULL", "GBP", "TRY", "BRL", "PAX", "UST", "VAI", "AEUR"]
    good = ["SOLUSDT", "ETHUSDT", "BNBUSDT", "YBUSDT", "BTCUSDT"]
    tickers = [_tick(b + "USDT", 1e12) for b in bad] + [_tick(s, 1) for s in good]
    tickers += [_tick("ETHBTC", 1e12), _tick("BTCEUR", 1e12), _tick("BTCUSDC", 1e12)]
    _no_archive(monkeypatch)
    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: FakeResp(tickers))
    out = universe.crypto_universe()
    assert [a.symbol for a in out] == ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "YBUSDT"]
    assert [a.is_major for a in out] == [True, True, False, False, False]
    assert [a.cap_bucket for a in out] == ["major", "major", "alt", "alt", "alt"]


def test_crypto_universe_n_and_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    _no_archive(monkeypatch)
    tickers = [_tick(f"C{i}USDT", i) for i in range(1, 11)]
    calls = []

    def fake_get(*a, **k):
        calls.append(1)
        return FakeResp(tickers)

    monkeypatch.setattr(universe.requests, "get", fake_get)
    out = universe.crypto_universe(3)
    assert [a.symbol for a in out] == ["C10USDT", "C1USDT", "C2USDT"]
    again = universe.crypto_universe(3)
    assert again == out and len(calls) == 1  # cache reused, no second request
    assert len(universe.crypto_universe()) == 10
    assert [a.symbol for a in universe.crypto_universe(2)] == ["C10USDT", "C1USDT"]


def test_committed_crypto_cache_is_clean():
    cache = universe._read_cache(Path(__file__).resolve().parents[1] / "universe" / "crypto.csv")
    assert len(cache) >= 400 and [a.symbol for a in cache[:2]] == ["BTCUSDT", "ETHUSDT"]
    listed = {a.symbol[:-4] for a in cache}
    for a in cache:
        assert a.symbol.endswith("USDT")
        assert not universe.excluded_crypto_base(a.symbol[:-4], listed), a.symbol


def test_real_coins_ending_in_b_up_down_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    t = [_tick(s + "USDT", v) for s, v in [("ARB", 9), ("SHIB", 8), ("BNB", 7), ("TRB", 6), ("JUP", 5),
                                           ("SYRUP", 4), ("BTC", 3), ("BTCUP", 2), ("BTCDOWN", 1),
                                           ("NVDAB", 99), ("TQQQB", 98)]]
    _no_archive(monkeypatch)
    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: FakeResp(t))
    got = [a.symbol for a in universe.crypto_universe()]
    assert got == ["BTCUSDT", "ARBUSDT", "BNBUSDT", "JUPUSDT", "SHIBUSDT", "SYRUPUSDT", "TRBUSDT"]


def test_yahoo_ticker():
    assert universe._yahoo_ticker("BRK.B") == "BRK-B"
    assert universe._yahoo_ticker(" BF.B ") == "BF-B"
    assert universe._yahoo_ticker("AAPL") == "AAPL"


def test_stock_universe_wikipedia_parse(monkeypatch, tmp_path):
    (tmp_path / "etfs.csv").write_text("symbol,sector\nSPY,broad\nXLK,tech\nAAPL,tech\n")
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    sp500 = ("<table><tr><th>Symbol</th><th>GICS Sector</th></tr>"
             "<tr><td>AAPL</td><td>Information Technology</td></tr>"
             "<tr><td>BRK.B</td><td>Financials</td></tr></table>")
    sp400 = ("<table><tr><th>Symbol</th><th>GICS Sector</th></tr>"
             "<tr><td>AOS</td><td>Industrials</td></tr></table>")

    class R:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    def fake_get(url, **k):
        return R(sp500 if "500" in url else sp400)

    monkeypatch.setattr(universe.requests, "get", fake_get)
    out = universe.stock_universe()
    by = {a.symbol: a for a in out}
    assert [a.symbol for a in out] == ["AAPL", "BRK-B", "AOS", "SPY", "XLK"]  # AAPL deduped
    assert by["AAPL"].cap_bucket == "large" and by["AAPL"].sector == "Information Technology"
    assert by["BRK-B"].cap_bucket == "large" and by["AOS"].cap_bucket == "mid"
    assert by["SPY"].is_etf and by["SPY"].sector == "broad" and by["SPY"].cap_bucket == "etf"
    assert by["XLK"].is_etf and by["XLK"].sector == "tech"
    assert all(a.market == "stock" for a in out)
    # cache written and reused with no network
    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: pytest.fail("network"))
    assert universe.stock_universe() == out


def test_stock_universe_etfs_only_from_etfs_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    pd.DataFrame([("AAA", "stock", "Tech", "large", False, False),
                  ("SPY", "stock", "broad", "etf", True, False)],   # stale ETF row in the cache
                 columns=["symbol", "market", "sector", "cap_bucket", "is_etf", "is_major"]
                 ).to_csv(tmp_path / "stocks.csv", index=False)
    pd.DataFrame({"symbol": ["SPY", "TLT"], "sector": ["us_equity", "bonds"]}).to_csv(tmp_path / "etfs.csv", index=False)
    out = universe.stock_universe()
    syms = [a.symbol for a in out]
    assert syms == ["AAA", "SPY", "TLT"]
    tlt = out[2]
    assert tlt.is_etf and tlt.cap_bucket == "etf" and tlt.sector == "bonds"


def test_committed_etf_list():
    df = pd.read_csv(universe.UNIVERSE_DIR / "etfs.csv")
    assert 95 <= len(df) <= 110 and df["symbol"].is_unique and "SPY" in set(df["symbol"])
    assert set(df["sector"]) == {"us_equity", "factor", "us_sector", "international", "bonds",
                                 "commodity", "currency", "realestate"}
    stocks = pd.read_csv(universe.UNIVERSE_DIR / "stocks.csv")
    assert not stocks["is_etf"].any() and not set(stocks["symbol"]) & set(df["symbol"])
