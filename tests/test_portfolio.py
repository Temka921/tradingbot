import time

import numpy as np
import pandas as pd
import pytest

from swingbot.backtest.portfolio import metrics, rule_mask, simulate, yearly_returns


def mk(dates, assets, fire, ret=0.0, split="validation"):
    p = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"]).to_frame(index=False)
    p["A"] = fire
    p["B"] = True
    p["split"] = split
    p["ret_long_10"] = ret
    p["ret_short_10"] = -0.01
    return p


def test_rule_mask_single_and_pair():
    p = pd.DataFrame({"A": [True, True, False], "B": [True, False, True]})
    assert rule_mask(p, "A").tolist() == [True, True, False]
    assert rule_mask(p, "A+B").tolist() == [True, False, False]


def test_single_trade_booking():
    dates = pd.bdate_range("2020-01-01", periods=30)
    p = mk(dates, ["X"], False, 0.10)
    p.loc[p["date"] == dates[3], "A"] = True
    trades, eq = simulate(p, "A", "long", "validation", horizon=10, max_positions=10)
    assert len(trades) == 1
    assert trades.iloc[0]["exit_date"] == dates[13]
    assert eq.index.equals(pd.DatetimeIndex(dates, name="date")) or list(eq.index) == list(dates)
    assert eq.iloc[0] == 1.0
    assert eq[dates[13]] == pytest.approx(1.01)
    assert (eq[: dates[13]].iloc[:-1] == 1.0).all()
    assert np.allclose(eq[dates[13]:].to_numpy(), 1.01)
    assert eq.attrs["n_trades"] == 1
    assert eq.attrs["busy_slots"][dates[4]] == 1 and eq.attrs["busy_slots"][dates[3]] == 0
    assert eq.attrs["busy_slots"][dates[14]] == 0


def test_too_few_bars_left_skipped():
    dates = pd.bdate_range("2020-01-01", periods=12)
    p = mk(dates, ["X"], True, 0.10)
    trades, eq = simulate(p, "A", "long", "validation")
    assert len(trades) == 2  # signals on dates[0], dates[1] only (h=10 rows remain)


def test_capacity_never_exceeded_and_seeded():
    dates = pd.bdate_range("2020-01-01", periods=60)
    p = mk(dates, [f"S{i}" for i in range(50)], True, 0.01)
    t0, e0 = simulate(p, "A", "long", "validation", max_positions=5, seed=0)
    t0b, _ = simulate(p, "A", "long", "validation", max_positions=5, seed=0)
    t1, _ = simulate(p, "A", "long", "validation", max_positions=5, seed=1)
    pd.testing.assert_frame_equal(t0, t0b)
    assert not t0[["date", "asset"]].equals(t1[["date", "asset"]])
    assert e0.attrs["busy_slots"].max() <= 5
    pos = {d: i for i, d in enumerate(dates)}
    occ_incl = np.zeros(len(dates), int)
    for _, r in t0.iterrows():
        occ_incl[pos[r["date"]] + 1 : pos[r["exit_date"]] + 1] += 1
    assert occ_incl.max() <= 5
    assert (t0["exit_date"] > t0["date"]).all()


def test_nan_ret_skipped():
    dates = pd.bdate_range("2020-01-01", periods=30)
    p = mk(dates, ["X"], False, 0.05)
    p.loc[p["date"] == dates[2], "A"] = True
    p.loc[p["date"] == dates[3], "A"] = True
    p.loc[p["date"] == dates[2], "ret_long_10"] = np.nan
    trades, eq = simulate(p, "A", "long", "validation")
    assert len(trades) == 1 and trades.iloc[0]["date"] == dates[3]
    assert not eq.isna().any()


def test_split_filter_and_short():
    dates = pd.bdate_range("2020-01-01", periods=30)
    p = mk(dates, ["X"], True, 0.05)
    p.loc[p["date"] < dates[10], "split"] = "discovery"
    trades, eq = simulate(p, "A", "short", "validation", max_positions=1)
    assert eq.index.min() == dates[10]
    assert (trades["ret"] == -0.01).all()


def test_metrics_hand_example():
    idx = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2021-01-31"])
    eq = pd.Series([1.0, 1.2, 0.9, 1.5], index=idx)
    eq.attrs["busy_slots"] = pd.Series([0, 5, 10, 5], index=idx)
    eq.attrs["max_positions"] = 10
    eq.attrs["n_trades"] = 7
    bench = pd.Series([100.0, 100.0, 100.0, 200.0], index=idx)
    m = metrics(eq, bench)
    assert m["total_return"] == pytest.approx(0.5)
    years = (idx[-1] - idx[0]).days / 365.25
    assert m["cagr"] == pytest.approx(1.5 ** (1 / years) - 1)
    assert m["max_drawdown"] == pytest.approx(-0.25)
    assert m["n_trades"] == 7
    assert m["exposure"] == pytest.approx(np.mean([0, 0.5, 1.0, 0.5]))
    assert m["bench_total_return"] == pytest.approx(1.0)
    assert m["bench_max_drawdown"] == pytest.approx(0.0)
    mo = eq.resample("ME").last().dropna().pct_change().dropna()
    assert m["sharpe_monthly"] == pytest.approx(mo.mean() / mo.std() * np.sqrt(12))
    m2 = metrics(eq)
    assert np.isnan(m2["bench_cagr"]) and np.isnan(m2["bench_total_return"])
    assert np.isnan(metrics(pd.Series(eq.values, index=idx))["exposure"])


def test_yearly_returns_zero_trade_year():
    idx = pd.bdate_range("2019-06-01", "2021-12-31")
    eq = pd.Series(1.0, index=idx)
    eq[eq.index >= "2019-09-01"] = 1.1
    eq[eq.index >= "2021-03-01"] = 1.21
    y = yearly_returns(eq)
    assert y[2019] == pytest.approx(0.1)
    assert y[2020] == 0.0
    assert y[2021] == pytest.approx(0.1)


def test_performance_large():
    n_dates, n_assets = 1500, 200
    dates = pd.bdate_range("2000-01-03", periods=n_dates)
    rng = np.random.default_rng(0)
    p = mk(dates, [f"S{i}" for i in range(n_assets)], rng.random(n_dates * n_assets) < 0.5, 0.0)
    p["ret_long_10"] = rng.normal(0.001, 0.03, len(p))
    t = time.time()
    simulate(p, "A", "long", "validation")
    assert time.time() - t < 30
