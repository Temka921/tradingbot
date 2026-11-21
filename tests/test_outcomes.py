import numpy as np
import pandas as pd
import pytest

from swingbot.config import COSTS, HORIZONS
from swingbot.outcomes import compute_outcomes


def make_bars(n=60, seed=0, start="2020-01-01", freq="B"):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1, n)
    idx = pd.date_range(start, periods=n, freq=freq, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.full(n, 1000.0)}, index=idx)


def flat_bars(n=60, price=100.0):
    """Flat series: TR = 2 every bar so ATR14 = 2 from bar 13 on."""
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame(
        {"open": price, "high": price + 1, "low": price - 1, "close": price,
         "volume": 1000.0}, index=idx)


def test_entry_is_next_open():
    df = make_bars(30)
    out = compute_outcomes(df, df, "stock")
    t = 3
    expect = df["close"].iloc[t + 5] / df["open"].iloc[t + 1] - 1
    assert out["gross_5"].iloc[t] == pytest.approx(expect)
    assert out["gross_5"].iloc[t] != pytest.approx(df["close"].iloc[t + 5] / df["close"].iloc[t] - 1)


def test_costs_applied():
    df = make_bars(40)
    out = compute_outcomes(df, df, "crypto")
    ok = out["gross_5"].notna()
    assert np.allclose(out["ret_long_5"][ok], out["gross_5"][ok] - 0.003)
    assert np.allclose(out["ret_short_5"][ok], -out["gross_5"][ok] - 0.003)
    out_s = compute_outcomes(df, df, "stock")
    assert np.allclose(out_s["ret_long_5"][ok], out_s["gross_5"][ok] - COSTS["stock"])


def test_excess_vs_bench():
    df = make_bars(60)
    for market in ("stock", "crypto"):
        out = compute_outcomes(df, df, market)
        for h in HORIZONS:
            ok = out[f"gross_{h}"].notna()
            assert ok.any()
            assert np.allclose(out[f"excess_long_{h}"][ok], -COSTS[market])
            assert np.allclose(out[f"excess_short_{h}"][ok], -COSTS[market])


def test_mfe_mae():
    df = make_bars(40)
    out = compute_outcomes(df, df, "stock")
    t, h = 4, 5
    win = df.iloc[t + 1 : t + h + 1]
    o = df["open"].iloc[t + 1]
    assert out["mfe_5"].iloc[t] == pytest.approx(win["high"].max() / o - 1)
    assert out["mae_5"].iloc[t] == pytest.approx(win["low"].min() / o - 1)


def _atr_case(modify):
    df = flat_bars(60)
    t = 30
    modify(df, t)
    return compute_outcomes(df, df, "stock"), t


def test_atr_exit_long_target_and_short_stop():
    def mod(df, t):
        df.iloc[t + 3, df.columns.get_loc("high")] = 107.0  # target 106
    out, t = _atr_case(mod)
    assert out["atr_exit_long"].iloc[t] == pytest.approx(106 / 100 - 1 - 0.001)
    # same bar pierces the short stop (104): short exits at the stop
    assert out["atr_exit_short"].iloc[t] == pytest.approx(-(104 / 100 - 1) - 0.001)


def test_atr_exit_stop_first_same_bar():
    def mod(df, t):
        df.iloc[t + 2, df.columns.get_loc("high")] = 110.0  # target hit
        df.iloc[t + 2, df.columns.get_loc("low")] = 90.0    # stop hit (96)
    out, t = _atr_case(mod)
    assert out["atr_exit_long"].iloc[t] == pytest.approx(96 / 100 - 1 - 0.001)
    # short: stop 104 and target 94 both hit -> stop
    assert out["atr_exit_short"].iloc[t] == pytest.approx(-(104 / 100 - 1) - 0.001)


def test_atr_exit_gap_through_stop_exits_at_open():
    def mod(df, t):
        i = t + 4
        df.iloc[i, df.columns.get_loc("open")] = 93.0   # gaps below stop 96
        df.iloc[i, df.columns.get_loc("low")] = 92.0
        df.iloc[i, df.columns.get_loc("high")] = 94.0
        df.iloc[i, df.columns.get_loc("close")] = 93.0
    out, t = _atr_case(mod)
    assert out["atr_exit_long"].iloc[t] == pytest.approx(93 / 100 - 1 - 0.001)


def test_atr_exit_time_exit():
    df = flat_bars(60)
    df.iloc[50, df.columns.get_loc("close")] = 100.5  # close[t+20], t=30
    out = compute_outcomes(df, df, "stock")
    assert out["atr_exit_long"].iloc[30] == pytest.approx(100.5 / 100 - 1 - 0.001)
    assert out["atr_exit_short"].iloc[30] == pytest.approx(-(100.5 / 100 - 1) - 0.001)


def test_tail_nan():
    df = make_bars(60)
    out = compute_outcomes(df, df, "stock")
    for h in HORIZONS:
        for m in ("gross", "ret_long", "ret_short", "bench_gross", "excess_long",
                  "excess_short", "mfe", "mae"):
            col = out[f"{m}_{h}"]
            assert col.iloc[-h:].isna().all(), (m, h)
            assert col.iloc[: -h].notna().all(), (m, h)
    assert out["atr_exit_long"].iloc[-20:].isna().all()
    assert out["atr_exit_long"].iloc[:13].isna().all()
    assert out["atr_exit_long"].iloc[13:-20].notna().all()


def test_short_history_no_crash():
    df = make_bars(30)
    out = compute_outcomes(df, df, "stock")
    assert out.index.equals(df.index)
    # 30 bars: ATR warm-up (13) + 20 forward bars leaves no valid ATR exit row
    assert out["atr_exit_long"].isna().all() and out["atr_exit_short"].isna().all()
    assert out["gross_20"].iloc[-20:].isna().all()
    # exactly 20 bars: no row has t+20, so every *_20 column is NaN
    out20 = compute_outcomes(make_bars(20), make_bars(20), "stock")
    for m in ("gross", "ret_long", "ret_short", "bench_gross", "excess_long",
              "excess_short", "mfe", "mae"):
        assert out20[f"{m}_20"].isna().all()
    tiny = compute_outcomes(make_bars(3), make_bars(3), "stock")
    assert tiny["gross_5"].isna().all()


def test_calendar_mismatch():
    asset = make_bars(50, seed=1, start="2020-01-01", freq="D")   # incl. weekends
    bench = make_bars(40, seed=2, start="2020-01-01", freq="B")   # weekdays, ends earlier
    out = compute_outcomes(asset, bench, "crypto")
    bd = bench.index.to_numpy()
    checked = 0
    for t in range(len(asset) - 5):
        d_in, d_out = asset.index[t + 1], asset.index[t + 5]
        i, j = np.searchsorted(bd, d_in.to_datetime64()), np.searchsorted(bd, d_out.to_datetime64())
        got = out["bench_gross_5"].iloc[t]
        if i >= len(bd) or j >= len(bd):
            assert np.isnan(got)
            continue
        expect = bench["close"].iloc[j] / bench["open"].iloc[i] - 1
        assert got == pytest.approx(expect)
        assert bench.index[i] >= d_in  # never a bar before entry date
        checked += 1
    assert checked > 10
    assert out["bench_gross_5"].iloc[-5:].isna().all()


def bars_from_closes(closes):
    """open = close = given, high/low +-1%."""
    c = np.asarray(closes, dtype=float)
    idx = pd.date_range("2020-01-01", periods=len(c), freq="D", name="date")
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": 1000.0}, index=idx)


def test_delisted_exit_at_last_close():
    df = bars_from_closes([10, 10, 10, 10, 5])          # last bar L = 4; open == close
    bench = bars_from_closes([100] * 5)
    out = compute_outcomes(df, bench, "crypto", delisted=True)
    # t=1: entry open[2]=10, horizon 5 would end at t=6 > L -> exits at close[4]=5
    assert out.loc[df.index[1], "gross_5"] == pytest.approx(-0.5)
    assert out.loc[df.index[1], "ret_long_5"] == pytest.approx(-0.5 - 0.003)
    assert out.loc[df.index[1], "excess_long_5"] == pytest.approx(-0.5 - 0.003)
    assert np.isnan(out.loc[df.index[4], "gross_5"])    # no t+1 bar: no entry
    assert out["atr_exit_long"].notna().sum() >= 0      # does not crash on a short series
    assert np.isnan(out.loc[df.index[1], "mfe_5"]) and np.isnan(out.loc[df.index[1], "mae_5"])


def test_delisted_complete_window_unchanged_and_atr_time_exit():
    df = make_bars(60)
    a = compute_outcomes(df, df, "crypto")
    b = compute_outcomes(df, df, "crypto", delisted=True)
    full = a["gross_5"].notna()
    assert np.allclose(a["gross_5"][full], b["gross_5"][full])
    assert b["gross_5"].iloc[-2] == pytest.approx(df["close"].iloc[-1] / df["open"].iloc[-1] - 1)
    ok = a["atr_exit_long"].notna()
    assert np.allclose(a["atr_exit_long"][ok], b["atr_exit_long"][ok])
    assert b["atr_exit_long"].iloc[-2:-1].notna().all()  # window past L time-exits at close[L]


def test_not_delisted_tail_still_nan():
    df = bars_from_closes([10, 10, 10, 10, 5])
    out = compute_outcomes(df, bars_from_closes([100] * 5), "crypto")
    assert out["gross_5"].isna().all()
