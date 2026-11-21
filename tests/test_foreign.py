import pandas as pd
import pytest

from swingbot import config
from swingbot.data import foreign, store, universe


class R:
    def __init__(self, text="", payload=None):
        self.text = text
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def _bars(start, end, **over):
    idx = pd.date_range(start, end, freq="B", name="date")
    df = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 100.0}, index=idx)
    for k, v in over.items():
        df[k] = v
    return df


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path / "universe")
    return tmp_path


def test_config_values():
    assert config.COSTS["japan"] == 0.0020 and config.COSTS["europe"] == 0.0020
    assert config.HOLDOUT_MARKETS == {"stock", "crypto"}
    assert config.HOME_INDEX == {"japan": "^N225", ".DE": "^GDAXI", ".PA": "^FCHI", ".L": "^FTSE", "us": "^GSPC"}


def test_holdout_scope(dirs):
    store.save_bars("7203.T", "japan", _bars("2023-12-29", "2025-01-03"))
    out = store.load_bars("7203.T", "japan")
    assert out.index.max() == pd.Timestamp("2025-01-03")
    out = store.load_bars("7203.T", "japan", allow_holdout=True)
    assert out.index.max() == pd.Timestamp("2025-01-03")
    assert not (config.DATA_DIR / "holdout_log.txt").exists()
    store.save_bars("AAPL", "stock", _bars("2023-12-29", "2025-01-03"))
    assert store.load_bars("AAPL", "stock").index.max() == pd.Timestamp("2023-12-29")
    store.load_bars("AAPL", "stock", allow_holdout=True)
    lines = (config.DATA_DIR / "holdout_log.txt").read_text().splitlines()
    assert len(lines) == 1 and "stock/AAPL" in lines[0]


def test_japan_universe_parse(dirs, monkeypatch):
    html = ("<table><tr><th>Foo</th></tr><tr><td>x</td></tr></table>"
            "<table><tr><th>Code</th><th>Company</th></tr>"
            "<tr><td>7203</td><td>Toyota</td></tr>"
            "<tr><td>986</td><td>Short Code Co</td></tr></table>")
    monkeypatch.setattr(foreign.requests, "get", lambda url, **k: R(html))
    out = foreign.japan_universe()
    assert list(out.columns) == ["symbol", "market", "home_index"]
    assert out["symbol"].tolist() == ["7203.T", "0986.T"]
    assert set(out["market"]) == {"japan"} and set(out["home_index"]) == {"^N225"}
    assert (universe.UNIVERSE_DIR / "japan.csv").exists()
    monkeypatch.setattr(foreign.requests, "get", lambda *a, **k: pytest.fail("network"))
    pd.testing.assert_frame_equal(foreign.japan_universe(), out)


def test_japan_table_missing_raises(dirs, monkeypatch):
    monkeypatch.setattr(foreign.requests, "get", lambda url, **k: R("<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"))
    with pytest.raises(ValueError, match="wikipedia"):
        foreign.japan_universe()


def test_europe_universe_parse(dirs, monkeypatch):
    dax = ("<table><tr><th>Company</th><th>Ticker</th></tr>"
           "<tr><td>Siemens</td><td>SIE.DE</td></tr><tr><td>SAP SE</td><td>SAP</td></tr>"
           "<tr><td>Shared Co</td><td>SHD</td></tr></table>")
    cac = ("<table><tr><th>Company</th><th>Ticker</th></tr>"
           "<tr><td>LVMH</td><td>MC</td></tr>"
           "<tr><td>Shared Co.</td><td>SHC</td></tr></table>")
    ftse = ("<table><tr><th>Company</th><th>EPIC</th></tr>"
            "<tr><td>BT Group</td><td>BT.A</td></tr><tr><td>Shell</td><td>SHEL</td></tr></table>")

    def get(url, **k):
        return R(dax if "DAX" in url else cac if "CAC" in url else ftse)

    monkeypatch.setattr(foreign.requests, "get", get)
    out = foreign.europe_universe()
    assert out["symbol"].tolist() == ["SIE.DE", "SAP.DE", "SHD.DE", "MC.PA", "BT-A.L", "SHEL.L"]
    assert out["home_index"].tolist() == ["^GDAXI"] * 3 + ["^FCHI", "^FTSE", "^FTSE"]
    assert set(out["market"]) == {"europe"}
    assert (universe.UNIVERSE_DIR / "europe.csv").exists()


def test_glitch_filter():
    close = [100.0] * 10
    close[3] = 300.0            # +200% spike that reverts
    close[6:] = [170.0] * 4     # real +70% move that holds
    df = _bars("2020-01-01", "2020-01-14")
    df["close"] = close
    df["open"] = close
    df["high"] = [c * 1.01 for c in close]
    df["low"] = [c * 0.99 for c in close]
    orig = df.copy()
    out = foreign.glitch_filter(df)
    pd.testing.assert_frame_equal(df, orig)  # input untouched
    assert out["close"].iloc[3] == 100.0 and out["open"].iloc[3] == 100.0
    assert out["high"].iloc[3] == pytest.approx(101.0) and out["low"].iloc[3] == pytest.approx(99.0)
    assert out["close"].iloc[6] == 170.0 and out["close"].iloc[9] == 170.0
    assert out.notna().all().all()


def test_fetch_fng(monkeypatch):
    payload = {"data": [{"value": "40", "timestamp": "1700092800"},
                        {"value": "25", "timestamp": "1700006400"}]}
    seen = {}

    def get(url, **k):
        seen["url"] = url
        return R(payload=payload)

    monkeypatch.setattr(foreign.requests, "get", get)
    s = foreign.fetch_fng()
    assert "alternative.me/fng/?limit=0" in seen["url"]
    assert list(s.index) == [pd.Timestamp("2023-11-15"), pd.Timestamp("2023-11-16")]
    assert s.tolist() == [25.0, 40.0] and s.dtype == float and s.index.tz is None


def test_save_load_fng(dirs):
    s = pd.Series([25.0, 40.0], index=pd.DatetimeIndex(["2023-11-15", "2023-11-16"], name="date"), name="fng")
    foreign.save_fng(s)
    assert (config.DATA_DIR / "raw" / "sentiment" / "fng.parquet").exists()
    pd.testing.assert_series_equal(foreign.load_fng(), s, check_freq=False)


def test_download_market_drops_today(dirs, monkeypatch):
    monkeypatch.setattr(foreign.universe, "UNIVERSE_DIR", universe.UNIVERSE_DIR)
    universe.UNIVERSE_DIR.mkdir(parents=True)
    pd.DataFrame({"symbol": ["7203.T", "BAD.T"], "market": "japan", "home_index": "^N225"}
                 ).to_csv(universe.UNIVERSE_DIR / "japan.csv", index=False)
    monkeypatch.setattr(foreign, "_today", lambda: pd.Timestamp("2024-03-06"))
    monkeypatch.setattr(foreign.download, "_sleep", lambda s: None)
    bars = _bars("2024-03-01", "2024-03-08")

    def fake_yahoo(sym, start="2000-01-01"):
        if sym == "BAD.T":
            return bars.iloc[0:0]
        return bars

    monkeypatch.setattr(foreign, "fetch_yahoo", fake_yahoo)
    msgs = foreign.download_market("japan")
    assert any(m.startswith("OK 7203.T") for m in msgs)
    assert any(m.startswith("SKIP BAD.T") for m in msgs)
    assert any(m.startswith("OK ^N225") for m in msgs)
    for sym in ("7203.T", "^N225"):
        saved = store.load_bars(sym, "japan")
        assert saved.index.max() == pd.Timestamp("2024-03-05")
