import pandas as pd
import pytest

from swingbot import config
from swingbot.data import store


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


def _bars(start, end):
    idx = pd.date_range(start, end, freq="B", name="date")
    n = len(idx)
    return pd.DataFrame(
        {
            "open": [1.0 + i for i in range(n)],
            "high": [2.0 + i for i in range(n)],
            "low": [0.5 + i for i in range(n)],
            "close": [1.5 + i for i in range(n)],
            "volume": [100.0 + i for i in range(n)],
        },
        index=idx,
    )


def test_roundtrip():
    df = _bars("2023-01-01", "2023-01-10")
    path = store.save_bars("AAPL", "stock", df)
    assert path == config.DATA_DIR / "raw" / "stock" / "AAPL.parquet"
    out = store.load_bars("AAPL", "stock", allow_holdout=True)
    pd.testing.assert_frame_equal(out, df, check_freq=False)


def test_holdout_cut():
    store.save_bars("AAPL", "stock", _bars("2023-12-29", "2024-01-03"))
    out = store.load_bars("AAPL", "stock")
    assert out.index.max() == pd.Timestamp("2023-12-29")


def test_holdout_logged():
    store.save_bars("AAPL", "stock", _bars("2023-12-29", "2024-01-03"))
    store.load_bars("AAPL", "stock", allow_holdout=True)
    lines = (config.DATA_DIR / "holdout_log.txt").read_text().splitlines()
    assert len(lines) == 1 and "AAPL" in lines[0]


def test_split_of():
    s = store.split_of(
        pd.DatetimeIndex(["2017-12-29", "2018-01-02", "2024-01-02", "1999-12-31"]),
        "stock",
    )
    assert list(s) == ["discovery", "validation", "holdout", "pre"]
    c = store.split_of(pd.DatetimeIndex(["2021-12-31"]), "crypto")
    assert list(c) == ["discovery"]
