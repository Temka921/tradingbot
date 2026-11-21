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
    out = panel.build_panel("stock", max_skip_frac=1.0)
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
    out = panel.build_panel("stock", max_skip_frac=1.0)
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


@pytest.mark.parametrize("n", [0, 1, 10, 251, 252, 600])
def test_label_regimes_any_length(n):
    df = rand_bars(n, seed=9) if n else rand_bars(1).iloc[:0]
    out = label_regimes(df, rand_bars(700, seed=10))
    assert len(out) == n
    if n < 252:
        assert (out["vol_regime"] == "unknown").all()
    if n == 600:
        assert (out["vol_regime"].iloc[-1] != "unknown")


def test_build_panel_short_history_asset_included(capsys):
    _stock_setup([("NEW", "stock", "Tech", "large", False, False)])
    store.save_bars("NEW", "stock", rand_bars(120, seed=8, start="2020-01-01"))
    out = panel.build_panel("stock")
    new = out[out["asset"] == "NEW"]
    assert len(new) == 120
    assert (new["vol_regime"] == "unknown").all()
    assert "SKIP NEW" not in capsys.readouterr().out


def test_build_panel_keeps_low_priced_stock_rows(capsys):
    """No price-level filter: adjusted closes are back-adjusted, so 'cheap at t' selects later winners."""
    _stock_setup([("PEN", "stock", "Tech", "small", False, False)])
    close = np.concatenate([np.full(400, 0.05), np.full(1100, 5.0)]) * np.exp(
        np.random.default_rng(3).normal(0, 0.001, 1500))
    pen = bars_from_close(close, start="2016-01-01")
    store.save_bars("PEN", "stock", pen)
    out = panel.build_panel("stock")
    exp = pen[pen.index < pd.Timestamp(config.HOLDOUT_START)]
    exp = exp[store.split_of(exp.index, "stock") != "pre"]
    got = out[out["asset"] == "PEN"]
    assert len(got) == len(exp)
    assert got["date"].min() == exp.index.min()
    assert "dropped" not in capsys.readouterr().out


def test_crypto_rows_under_one_dollar_kept():
    for sym in ("BTCUSDT", "CHEAPUSDT"):
        store.save_bars(sym, "crypto", rand_bars(500, seed=len(sym), start="2020-01-01", freq="D"))
    cheap = rand_bars(500, seed=9, start="2020-01-01", freq="D")
    cheap[["open", "high", "low", "close"]] *= 0.001
    store.save_bars("CHEAPUSDT", "crypto", cheap)
    _meta([("BTCUSDT", "crypto", "crypto", "major", False, True),
           ("CHEAPUSDT", "crypto", "crypto", "alt", False, False)])
    out = panel.build_panel("crypto")
    assert (out["asset"] == "CHEAPUSDT").sum() == 500


def test_skip_guard_raises_on_too_many_failures(capsys):
    _stock_setup([("BAD", "stock", "Tech", "large", False, False)])
    (config.DATA_DIR / "raw" / "stock" / "BAD.parquet").write_bytes(b"not a parquet")
    with pytest.raises(ValueError, match=r"1 of 3 stock assets .*skipped"):
        panel.build_panel("stock")
    assert "SKIP BAD:" in capsys.readouterr().out
    out = panel.build_panel("stock", max_skip_frac=0.5)
    assert set(out["asset"]) == {"SPY", "AAA"}
    assert "1 of 3 assets skipped for errors" in capsys.readouterr().out


def test_skip_guard_ignores_expected_skips(capsys):
    # pre-period-only and post-holdout-only assets are expected skips: no raise at the default 2%
    _stock_setup([("OLD", "stock", "Tech", "large", False, False),
                  ("LATE", "stock", "Tech", "large", False, False)])
    store.save_bars("OLD", "stock", rand_bars(200, seed=7, start="1990-01-01"))
    store.save_bars("LATE", "stock", rand_bars(100, seed=8, start="2025-01-01"))
    out = panel.build_panel("stock")
    assert set(out["asset"]) == {"SPY", "AAA"}
    assert "0 of 4 assets skipped for errors" in capsys.readouterr().out


def test_build_panel_empty_after_holdout_cut_skipped(capsys):
    _stock_setup([("LATE", "stock", "Tech", "large", False, False)])
    store.save_bars("LATE", "stock", rand_bars(100, seed=8, start="2025-01-01"))
    out = panel.build_panel("stock")
    assert set(out["asset"]) == {"SPY", "AAA"}
    assert "SKIP LATE" in capsys.readouterr().out


def test_panel_marks_delisted(capsys):
    btc = rand_bars(900, seed=1, start="2020-01-01", freq="D")       # ends mid-2022, the market's last date
    dead = rand_bars(400, seed=2, start="2020-01-01", freq="D")      # ends early 2021
    for s, d in (("BTCUSDT", btc), ("LUNAUSDT", dead)):
        store.save_bars(s, "crypto", d)
    _meta([("BTCUSDT", "crypto", "crypto", "major", False, True),
           ("LUNAUSDT", "crypto", "crypto", "alt", False, False)])
    out = panel.build_panel("crypto")
    assert out.groupby("asset", observed=True)["delisted"].first().to_dict() == {"BTCUSDT": False, "LUNAUSDT": True}
    assert out["delisted"].dtype == bool
    last = out[out.asset == "LUNAUSDT"].iloc[-3]
    assert not np.isnan(last["gross_10"])               # exits at the last close instead of NaN


def test_split_relistings():
    d1 = pd.date_range("2021-01-01", periods=30, freq="D")
    d2 = pd.date_range(d1[-1] + pd.Timedelta(days=18), periods=20, freq="D")
    df = bars_from_close([1.0] * 50)
    df.index = pd.DatetimeIndex(list(d1) + list(d2), name="date")
    segs = panel.split_relistings("LUNAUSDT", df)
    assert [s for s, _ in segs] == ["LUNAUSDT@2021-01-01", "LUNAUSDT"]
    assert [len(x) for _, x in segs] == [30, 20]
    assert panel.split_relistings("BTCUSDT", df.iloc[:30])[0][0] == "BTCUSDT"


def test_panel_relisted_symbol_becomes_two_assets(capsys):
    # BTCUSDT continuous; LUNAUSDT with an 18-day gap -> two assets, the first delisted
    btc = rand_bars(900, seed=1, start="2020-01-01", freq="D")
    l1 = rand_bars(200, seed=2, start="2020-01-01", freq="D")
    l2 = rand_bars(681, seed=3, start=str(l1.index[-1] + pd.Timedelta(days=19))[:10], freq="D")
    assert (l2.index[0] - l1.index[-1]).days == 19       # 18 missing days
    luna = pd.concat([l1, l2])
    store.save_bars("BTCUSDT", "crypto", btc)
    store.save_bars("LUNAUSDT", "crypto", luna)
    _meta([("BTCUSDT", "crypto", "crypto", "major", False, True),
           ("LUNAUSDT", "crypto", "crypto", "alt", False, False)])
    out = panel.build_panel("crypto")
    assert set(out["asset"]) == {"BTCUSDT", "LUNAUSDT@2020-01-01", "LUNAUSDT"}
    assert (out["asset"] == "LUNAUSDT@2020-01-01").sum() == 200
    assert (out["asset"] == "LUNAUSDT").sum() == 681
    d = out.groupby("asset", observed=True)["delisted"].first().to_dict()
    assert d["LUNAUSDT@2020-01-01"] is True and d["LUNAUSDT"] is False
    assert (out[out.asset == "LUNAUSDT@2020-01-01"]["is_major"] == False).all()  # same meta row
    first = out[out.asset == "LUNAUSDT@2020-01-01"]
    assert not np.isnan(first.iloc[-3]["gross_10"])      # dead segment exits at its last close
