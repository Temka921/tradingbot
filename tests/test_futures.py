import io
import zipfile

import pandas as pd
import pytest

from swingbot import config
from swingbot.data import binance, futures, store
from tests.test_data import FakeResp, FakeText

DAY_MS = 86_400_000
FUND = "data/futures/um/monthly/fundingRate/"
KL = "data/futures/um/monthly/klines/"


def ms(d) -> int:
    return int(pd.Timestamp(d).timestamp() * 1000)


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", text)
    return buf.getvalue()


def _listing(keys=(), prefixes=(), truncated=False, marker=None):
    body = f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"
    if marker:
        body += f"<NextMarker>{marker}</NextMarker>"
    body += "".join(f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes)
    body += "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    return f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">{body}</ListBucketResult>'


class Net:
    """Fake requests.get routing archive listing, archive zips and the futures API."""

    def __init__(self, listing_keys=(), zips=None, api=None):
        self.keys, self.zips, self.api, self.api_calls = list(listing_keys), zips or {}, api, []

    def __call__(self, url, params=None, timeout=None, **k):
        if url == binance.ARCHIVE_LIST_URL:
            return FakeText(_listing(keys=self.keys))
        if url.startswith(binance.ARCHIVE_FILE_URL):
            return FakeText(content=self.zips[url[len(binance.ARCHIVE_FILE_URL):]])
        self.api_calls.append((url, dict(params)))
        return FakeResp(self.api(url, params))


def test_list_um_symbols_paginates_and_filters(monkeypatch):
    pages = [
        _listing(prefixes=[FUND + "BTCUSDT/", FUND + "BTCUSD_PERP/"], truncated=True, marker=FUND + "BTCUSD_PERP/"),
        _listing(prefixes=[FUND + "ETHUSDT/", FUND + "BTCUSDT_240628/", FUND + "ETHBUSD/"]),
    ]
    calls = []

    def get(url, params=None, timeout=None, **k):
        calls.append(params.get("marker"))
        assert params["prefix"] == FUND
        return FakeText(pages[len(calls) - 1])

    monkeypatch.setattr(binance.requests, "get", get)
    assert futures.list_um_symbols() == ["BTCUSDT", "ETHUSDT"]
    assert calls == [None, FUND + "BTCUSD_PERP/"]


def test_fetch_funding_archive_header_and_no_header_ms_us(monkeypatch):
    z_hdr = _zip("calc_time,funding_interval_hours,last_funding_rate\n"
                 f"{ms('2023-01-01 00:00')},8,0.0001\n{ms('2023-01-01 08:00')},8,0.0002\n")
    z_nohdr = _zip(f"{ms('2023-02-01 00:00')},8,0.0003\n")
    z_us = _zip("calc_time,funding_interval_hours,last_funding_rate\n"
                f"{ms('2025-01-01 00:00') * 1000},8,0.0004\n")
    base = FUND + "BTCUSDT/"
    keys = [base + "BTCUSDT-fundingRate-2023-01.zip", base + "BTCUSDT-fundingRate-2023-01.zip.CHECKSUM",
            base + "BTCUSDT-fundingRate-2023-02.zip", base + "BTCUSDT-fundingRate-2025-01.zip"]
    zips = {keys[0]: z_hdr, keys[2]: z_nohdr, keys[3]: z_us}
    net = Net(keys, zips, api=lambda u, p: [])
    monkeypatch.setattr(binance.requests, "get", net)
    monkeypatch.setattr(binance, "_today_ms", lambda: ms("2025-01-01"))
    df = futures.fetch_funding("BTCUSDT")
    assert list(df.columns) == ["funding_time", "rate"]
    assert list(df["funding_time"]) == list(pd.to_datetime(
        ["2023-01-01 00:00", "2023-01-01 08:00", "2023-02-01 00:00", "2025-01-01 00:00"]))
    assert list(df["rate"]) == [0.0001, 0.0002, 0.0003, 0.0004]
    assert df["funding_time"].dt.tz is None and df["rate"].dtype == float


def test_fetch_funding_api_tail_paginates(monkeypatch):
    base = FUND + "BTCUSDT/"
    key = base + "BTCUSDT-fundingRate-2023-01.zip"
    z = _zip(f"{ms('2023-01-31 16:00')},8,0.0001\n")
    rows = [{"symbol": "BTCUSDT", "fundingTime": ms("2023-02-01") + i * 8 * 3_600_000,
             "fundingRate": "0.0005", "markPrice": "1"} for i in range(5)]

    def api(url, params):
        assert url.endswith("/fapi/v1/fundingRate") and params["symbol"] == "BTCUSDT"
        return [r for r in rows if r["fundingTime"] >= params["startTime"]][: params["limit"]]

    net = Net([key], {key: z}, api)
    monkeypatch.setattr(binance.requests, "get", net)
    monkeypatch.setattr(futures, "API_LIMIT", 2)
    df = futures.fetch_funding("BTCUSDT")
    assert len(df) == 6 and df["funding_time"].is_monotonic_increasing
    assert df["funding_time"].iloc[0] == pd.Timestamp("2023-01-31 16:00")
    assert len(net.api_calls) == 3  # 2 + 2 + 1 rows; the short page stops the loop
    assert net.api_calls[0][1]["startTime"] > ms("2023-01-31 16:00")


def test_fetch_funding_api_only_when_no_archive(monkeypatch):
    rows = [{"fundingTime": ms("2024-05-01"), "fundingRate": "0.001"},
            {"fundingTime": ms("2024-05-01"), "fundingRate": "0.001"}]  # duplicate deduped
    net = Net([], {}, lambda u, p: rows if p["startTime"] <= ms("2024-05-01") else [])
    monkeypatch.setattr(binance.requests, "get", net)
    df = futures.fetch_funding("NEWUSDT")
    assert len(df) == 1 and df["rate"].iloc[0] == 0.001


def test_fetch_funding_empty(monkeypatch):
    monkeypatch.setattr(binance.requests, "get", Net([], {}, lambda u, p: []))
    df = futures.fetch_funding("NOPEUSDT")
    assert df.empty and list(df.columns) == ["funding_time", "rate"]


def _krow(t, close="1.5"):
    return [t, "1.0", "2.0", "0.5", close, "10.0", t, "0", 0, "0", "0", "0"]


def test_fetch_perp_klines_archive_plus_api_drops_open_day(monkeypatch):
    base = KL + "BTCUSDT/1d/"
    key = base + "BTCUSDT-1d-2024-12.zip"
    z = _zip("\n".join(",".join(map(str, _krow(ms(d)))) for d in ["2024-12-30", "2024-12-31"]))
    today = ms("2025-01-04")

    def api(url, params):
        assert url.endswith("/fapi/v1/klines") and params["interval"] == "1d"
        return [_krow(t) for t in (ms("2025-01-01"), ms("2025-01-02"), ms("2025-01-03"), today)
                if t >= params["startTime"]]

    net = Net([key], {key: z}, api)
    monkeypatch.setattr(binance.requests, "get", net)
    monkeypatch.setattr(binance, "_today_ms", lambda: today)
    df = futures.fetch_perp_klines("BTCUSDT")
    assert list(df.index) == list(pd.to_datetime(
        ["2024-12-30", "2024-12-31", "2025-01-01", "2025-01-02", "2025-01-03"]))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"] and df.index.name == "date"
    assert net.api_calls[0][1]["startTime"] == ms("2025-01-01")


def test_daily_funding_sums_per_utc_day():
    df = pd.DataFrame({"funding_time": pd.to_datetime(
        ["2023-01-01 00:00", "2023-01-01 08:00", "2023-01-01 16:00", "2023-01-03 00:00"]),
        "rate": [0.0001, 0.0002, -0.0001, 0.0005]})
    s = futures.daily_funding(df)
    assert s.index.name == "date"
    # 00:00 payments settle the previous day's period
    assert s.loc["2022-12-31"] == pytest.approx(0.0001)
    assert s.loc["2023-01-01"] == pytest.approx(0.0001)   # 08:00 + 16:00
    assert s.loc["2023-01-02"] == pytest.approx(0.0005)   # the 2023-01-03 00:00 payment
    assert pd.Timestamp("2023-01-03") not in s.index
    assert futures.daily_funding(df.iloc[0:0]).empty


@pytest.fixture
def datadir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


def _fund():
    return pd.DataFrame({"funding_time": pd.to_datetime(["2023-12-31 16:00", "2024-01-01 00:00"]),
                         "rate": [0.0001, 0.0002]})


def test_funding_roundtrip_and_holdout_cut_and_log(datadir):
    futures.save_funding("BTCUSDT", _fund())
    assert (datadir / "raw" / "funding" / "BTCUSDT.parquet").exists()
    cut = futures.load_funding("BTCUSDT")
    assert list(cut["rate"]) == [0.0001]
    assert not (datadir / "holdout_log.txt").exists()
    full = futures.load_funding("BTCUSDT", allow_holdout=True)
    assert len(full) == 2
    assert "holdout access funding/BTCUSDT" in (datadir / "holdout_log.txt").read_text()


def test_perp_bars_holdout_cut_and_log(datadir):
    idx = pd.DatetimeIndex(pd.to_datetime(["2023-12-31", "2024-01-01"]), name="date")
    df = pd.DataFrame({c: [1.0, 2.0] for c in store.COLUMNS}, index=idx)
    store.save_bars("BTCUSDT", "perp", df)
    assert (datadir / "raw" / "perp" / "BTCUSDT.parquet").exists()
    assert len(store.load_bars("BTCUSDT", "perp")) == 1
    assert len(store.load_bars("BTCUSDT", "perp", allow_holdout=True)) == 2
    assert "holdout access perp/BTCUSDT" in (datadir / "holdout_log.txt").read_text()


def test_futures_universe_uses_crypto_bases(monkeypatch):
    monkeypatch.setattr(futures, "list_um_symbols", lambda: ["BTCUSDT", "ETHUSDT", "USDCUSDT", "XYZUSDT"])
    monkeypatch.setattr(futures, "crypto_universe", lambda: [
        type("A", (), {"symbol": s})() for s in ["BTCUSDT", "ETHUSDT", "XYZUSDT"]])
    assert futures.futures_universe() == ["BTCUSDT", "ETHUSDT", "XYZUSDT"]


def test_api_400_means_delisted_and_keeps_archive(monkeypatch):
    class R:
        status_code = 400
        def raise_for_status(self):
            raise AssertionError("must not raise for a delisted symbol")
        def json(self):
            return []
    monkeypatch.setattr(futures.requests, "get", lambda *a, **k: R())
    assert futures._funding_api("DEADUSDT", 0).empty
    assert futures._klines_api("DEADUSDT", 0).empty
