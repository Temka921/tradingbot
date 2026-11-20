import numpy as np
import pandas as pd
import pytest

from swingbot import config, panel
from swingbot.data import store
from swingbot.regimes import label_regimes
from swingbot.signals import REGISTRY


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


def bars_from_close(close, start="2015-01-01", freq="B"):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range(start, periods=len(close), freq=freq, name="date")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": 1000.0}, index=idx)


def rand_bars(n, seed=0, start="2015-01-01", freq="B"):
    rng = np.random.default_rng(seed)
    return bars_from_close(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), start, freq)


def test_trend_regime_labels():
    n = 400
    up = bars_from_close(100 * 1.003 ** np.arange(n))
    down = bars_from_close(100 * 0.997 ** np.arange(n))
    flat = bars_from_close(np.full(n, 100.0))
    assert label_regimes(up, up)["trend_regime"].iloc[-1] == "up"
    assert label_regimes(down, down)["trend_regime"].iloc[-1] == "down"
    assert label_regimes(flat, flat)["trend_regime"].iloc[-1] == "sideways"
    assert (label_regimes(up, up)["trend_regime"].iloc[:199] == "unknown").all()


def test_market_regime_ffill_only():
    asset = rand_bars(900, freq="D", start="2018-01-01")  # trades weekends
    bench = bars_from_close(100 * 1.003 ** np.arange(400), start="2018-03-01")  # bull
    out = label_regimes(asset, bench)["market_regime"]
    assert (out[: "2018-02-28"] == "unknown").all()
    # a Saturday takes Friday's value, never Monday's
    sat = out.index[(out.index.day_of_week == 5) & (out.index > bench.index[250])][0]
    fri = sat - pd.Timedelta(days=1)
    assert out[sat] == out[fri] == "bull"
    # before bench SMA200 exists -> unknown even though bench has bars
    assert out[bench.index[100]] == "unknown"


def test_market_regime_no_future_bench():
    asset = rand_bars(400, freq="D", start="2018-01-01")
    close = np.concatenate([100 * 1.003 ** np.arange(300), 100 * 1.003 ** 299 * 0.97 ** np.arange(1, 101)])
    bench = bars_from_close(close, start="2018-01-01")
    full = label_regimes(asset, bench)["market_regime"]
    cut = bench.index[250]
    part = label_regimes(asset, bench[bench.index <= cut])["market_regime"]
    assert (full[: cut] == part[: cut]).all()


def test_short_history_unknown():
    df = rand_bars(100)
    out = label_regimes(df, df)
    for c in ("trend_regime", "vol_regime", "market_regime"):
        assert (out[c] == "unknown").all()
    assert out["adx_regime"].iloc[-1] in {"trending", "ranging", "neutral"}
    assert (out["adx_regime"] != "unknown").sum() > 0


def test_regimes_causal_truncation():
    df = rand_bars(900, seed=3)
    bench = rand_bars(900, seed=4)
    full = label_regimes(df, bench)
    cut = 700
    part = label_regimes(df.iloc[: cut + 1], bench.iloc[: cut + 1])
    pd.testing.assert_frame_equal(full.iloc[: cut + 1], part, check_dtype=False)


def _meta(rows):
    pd.DataFrame(rows, columns=["symbol", "market", "sector", "cap_bucket", "is_etf", "is_major"]) \
        .to_parquet(config.DATA_DIR / "universe_meta.parquet")


def test_build_panel_shape():
    spy = rand_bars(1500, seed=1, start="2016-01-01")  # crosses 2024 holdout
    a = rand_bars(1500, seed=2, start="2016-01-01")
    b = rand_bars(900, seed=3, start="2016-06-01")
    for sym, d in (("SPY", spy), ("AAA", a), ("BBB", b)):
        store.save_bars(sym, "stock", d)
    _meta([("SPY", "stock", "broad", "etf", True, False),
           ("AAA", "stock", "Tech", "large", False, False),
           ("BBB", "stock", "Health", "mid", False, False)])
    out = panel.build_panel("stock")
    exp = 0
    for d in (spy, a, b):
        d = d[d.index < pd.Timestamp(config.HOLDOUT_START)]
        exp += int((store.split_of(d.index, "stock") != "pre").sum())
    assert len(out) == exp
    assert out["date"].max() < pd.Timestamp(config.HOLDOUT_START)
    assert set(out["split"]) <= {"discovery", "validation"}
    assert set(out["asset"]) == {"SPY", "AAA", "BBB"}
    required = {"date", "asset", "sector", "cap_bucket", "is_etf", "is_major", "market", "split",
                "trend_regime", "vol_regime", "market_regime", "adx_regime",
                "ret_long_10", "excess_long_10", "atr_exit_long", "mom_12_1_top_decile"}
    required |= set(REGISTRY)
    assert required <= set(out.columns)
    assert out["mom_12_1_top_decile"].dtype == bool
    assert all(out[s].dtype == bool for s in REGISTRY)
    assert (out["market"] == "stock").all()
    assert out.loc[out.asset == "BBB", "sector"].iloc[0] == "Health"


def test_build_panel_skips_stale_crypto(capsys):
    for sym in ("BTCUSDT", "ETHUSDT", "USD1USDT", "SOLUSDT"):
        store.save_bars(sym, "crypto", rand_bars(500, seed=len(sym), start="2020-01-01", freq="D"))
    _meta([("BTCUSDT", "crypto", "crypto", "major", False, True),
           ("ETHUSDT", "crypto", "crypto", "major", False, True),
           ("USD1USDT", "crypto", "crypto", "alt", False, False),
           ("SOLUSDT", "crypto", "crypto", "alt", False, False),
           ("MISSINGUSDT", "crypto", "crypto", "alt", False, False)])
    out = panel.build_panel("crypto")
    assert set(out["asset"]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    log = capsys.readouterr().out
    assert "SKIP USD1USDT" in log and "SKIP MISSINGUSDT" in log


def test_build_panel_holdout_flag_passthrough():
    store.save_bars("SPY", "stock", rand_bars(1500, seed=1, start="2019-01-01"))
    _meta([("SPY", "stock", "broad", "etf", True, False)])
    out = panel.build_panel("stock", allow_holdout=True)
    assert out["date"].max() >= pd.Timestamp(config.HOLDOUT_START)
    assert (config.DATA_DIR / "holdout_log.txt").exists()


def _stock_setup(extra=()):
    store.save_bars("SPY", "stock", rand_bars(1500, seed=1, start="2016-01-01"))
    store.save_bars("AAA", "stock", rand_bars(1500, seed=2, start="2016-01-01"))
    rows = [("SPY", "stock", "broad", "etf", True, False),
            ("AAA", "stock", "Tech", "large", False, False)]
    _meta(rows + list(extra))


def test_corrupt_asset_skipped(capsys):
    _stock_setup([("BAD", "stock", "Tech", "large", False, False)])
    (config.DATA_DIR / "raw" / "stock" / "BAD.parquet").write_bytes(b"not a parquet")
    out = panel.build_panel("stock")
    assert set(out["asset"]) == {"SPY", "AAA"}
    assert "SKIP BAD:" in capsys.readouterr().out


def test_failing_signal_asset_skipped(capsys, monkeypatch):
    _stock_setup()
    real = panel.compute_signals

    def boom(df):
        if len(df) == 777:
            raise RuntimeError("boom")
        return real(df)

    store.save_bars("ZZZ", "stock", rand_bars(777, seed=5, start="2016-01-01"))
    _meta([("SPY", "stock", "broad", "etf", True, False),
           ("AAA", "stock", "Tech", "large", False, False),
           ("ZZZ", "stock", "Tech", "large", False, False)])
    monkeypatch.setattr(panel, "compute_signals", boom)
    out = panel.build_panel("stock")
    assert set(out["asset"]) == {"SPY", "AAA"}
    assert "SKIP ZZZ: RuntimeError: boom" in capsys.readouterr().out


def test_all_pre_asset_ok(capsys):
    _stock_setup([("OLD", "stock", "Tech", "large", False, False)])
    store.save_bars("OLD", "stock", rand_bars(200, seed=7, start="1990-01-01"))
    out = panel.build_panel("stock")
    assert set(out["asset"]) == {"SPY", "AAA"}
    assert "SKIP OLD" in capsys.readouterr().out


def test_empty_meta_raises():
    _meta([])
    with pytest.raises(ValueError, match="stock"):
        panel.build_panel("stock")


def test_holdout_log_one_entry_per_asset():
    _stock_setup()
    panel.build_panel("stock", allow_holdout=True)
    lines = (config.DATA_DIR / "holdout_log.txt").read_text().splitlines()
    assert len(lines) == 2
    assert {l.split()[-1] for l in lines} == {"stock/SPY", "stock/AAA"}


def test_outcomes_float32():
    _stock_setup()
    out = panel.build_panel("stock")
    assert out["ret_long_10"].dtype == "float32"
