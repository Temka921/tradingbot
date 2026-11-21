import numpy as np
import pandas as pd
import pytest

from swingbot.backtest.trend import (month_end_closes, run_portfolio, target_weights,
                                     trend_signals)


def _closes(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2010-01-01", periods=n)
    up = 100 * np.cumprod(1 + 0.001 + rng.normal(0, 0.002, n))
    down = 100 * np.cumprod(1 - 0.001 + rng.normal(0, 0.002, n))
    shy = 100 * np.cumprod(1 + 0.0001 + 0 * rng.normal(0, 0.0005, n))
    return pd.DataFrame({"UP": up, "DOWN": down, "SHY": shy}, index=d)


def test_month_end_is_last_trading_day():
    c = _closes()
    me = month_end_closes(c)
    assert me.index[0] == pd.Timestamp("2010-01-29")
    assert (me.index.to_period("M").value_counts() == 1).all()


def test_signals_no_lookahead():
    c = _closes(600)
    full = trend_signals(c[["UP", "DOWN"]])
    cut = trend_signals(c[["UP", "DOWN"]].loc[:"2011-06-30"])
    pd.testing.assert_frame_equal(full.loc[:"2011-06-30"], cut)


def test_uptrend_held_downtrend_in_cash():
    c = _closes(600)
    sig = trend_signals(c[["UP", "DOWN"]]).dropna()
    w = target_weights(sig)
    assert (w["UP"] == 0.5).all() and (w["DOWN"] == 0).all() and (w["SHY"] == 0.5).all()


def test_trade_at_next_open_and_costs():
    d = pd.bdate_range("2020-01-01", periods=5)
    opens = pd.DataFrame({"A": [10, 10, 11, 12, 13.0]}, index=d)
    closes = pd.DataFrame({"A": [10, 10.5, 11.5, 12.5, 13.5]}, index=d)
    w = pd.DataFrame({"A": [1.0]}, index=[d[0]])  # signal at close of day 0 -> buy at open of day 1
    eq = run_portfolio(opens, closes, w, cost_per_side=0.001)
    assert eq.index[0] == d[1]
    assert eq.iloc[0] == pytest.approx((1 - 0.001) * 10.5 / 10)
    assert eq.iloc[-1] == pytest.approx((1 - 0.001) * 13.5 / 10)


def test_rebalance_charges_turnover_only():
    d = pd.bdate_range("2020-01-01", periods=4)
    px = pd.DataFrame({"A": [10.0] * 4, "SHY": [10.0] * 4}, index=d)
    w = pd.DataFrame({"A": [1.0, 0.0], "SHY": [0.0, 1.0]}, index=[d[0], d[1]])
    eq = run_portfolio(px, px, w, cost_per_side=0.001)
    # day1: buy A (cost 0.1%); day2: sell A, buy SHY -> turnover 2 x value
    v1 = 1 - 0.001
    assert eq.iloc[-1] == pytest.approx(v1 - 2 * v1 * 0.001)
