"""C1-C3, S1, S2 builders and the rebalance engine (no network)."""
import numpy as np
import pandas as pd
import pytest

from swingbot import indicators as ind
from swingbot.backtest.rebalance import run_weights
from swingbot.strategies import crypto, own_series_change, stocks

COST = 0.0015


# ---------------------------------------------------------------- fixtures

def _coins(n=500, k=8, seed=0, btc_drift=0.002):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2018-01-01", periods=n, freq="D")
    cols = {"BTCUSDT": 100 * np.cumprod(1 + btc_drift + rng.normal(0, 0.02, n))}
    for j in range(k):
        cols[f"C{j}USDT"] = 10 * np.cumprod(1 + rng.normal(0.0005 * (j - 3), 0.04, n))
    closes = pd.DataFrame(cols, index=d)
    closes.index.name = "date"
    # a late listing and a delisting
    closes.loc[: d[60], "C6USDT"] = np.nan
    closes.loc[d[350]:, "C7USDT"] = np.nan
    eligible = pd.DataFrame(rng.random(closes.shape) < 0.8, index=d, columns=closes.columns)
    eligible &= closes.notna()
    return closes, eligible


def _fng(index, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(0, 100, len(index)).astype(float), index=index)


def _stocks(n=600, k=60, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2005-01-03", periods=n)
    drift = rng.normal(0.0003, 0.0005, k)
    px = 50 * np.cumprod(1 + drift + rng.normal(0, 0.015, (n, k)), axis=0)
    closes = pd.DataFrame(px, index=d, columns=[f"S{j:02d}" for j in range(k)])
    closes.iloc[:300, -5:] = np.nan  # late listings
    return closes


# ---------------------------------------------------------------- crypto builders

def test_weekly_dates_are_sundays():
    w = crypto.weekly_dates(pd.date_range("2018-01-01", periods=30, freq="D"))
    assert len(w) == 4 and (w.day_of_week == 6).all()


@pytest.mark.parametrize("builder", ["c1", "c2", "c3c", "c3g"])
def test_crypto_builders_no_lookahead(builder):
    closes, elig = _coins()
    fng = _fng(closes.index)

    def build(c, e, f):
        if builder == "c1":
            return crypto.c1_weights(c, e)
        if builder == "c2":
            return crypto.c2_weights(c, e)
        return crypto.c3_weights(c, e, f, "contrarian" if builder == "c3c" else "confirming")

    full = build(closes, elig, fng)
    cut = "2018-12-31"
    part = build(closes.loc[:cut], elig.loc[:cut], fng.loc[:cut])
    pd.testing.assert_frame_equal(full.loc[:cut, part.columns], part)
    assert not part.empty


def test_c1_weights_shape_and_sums():
    closes, elig = _coins(btc_drift=0.004)
    w = crypto.c1_weights(closes, elig)
    assert (w.index.day_of_week == 6).all()
    assert "CASH" in w.columns
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    invested = w.drop(columns="CASH")
    assert ((invested > 0).sum(axis=1) <= 5).all()
    assert (invested.max(axis=1) >= 0.2 - 1e-12).any()
    # only eligible coins with a 30-bar history are held
    held = invested > 0
    assert not (held & ~elig.reindex(w.index)).to_numpy().any()


def test_c1_picks_top_five_by_30d_return():
    d = pd.date_range("2019-01-01", periods=260, freq="D")
    cols = {"BTCUSDT": np.linspace(100, 300, len(d))}
    for j in range(7):  # coin j grows at rate j -> higher j, higher 30-day return
        cols[f"C{j}"] = 10 * np.cumprod(np.full(len(d), 1 + 0.001 * j))
    closes = pd.DataFrame(cols, index=d)
    elig = pd.DataFrame(True, index=d, columns=closes.columns)
    elig["C6"] = False  # best coin not eligible
    w = crypto.c1_weights(closes, elig)
    last = w.iloc[-1]
    held = set(last[last > 0].index) - {"CASH"}
    # BTC 30d return ~ 200/ (300 - 23) -> about 8.3%; C5 ~ 16%, C4 ~ 12.7%, C3 ~ 9.4%, C2 ~ 6.2%
    assert held == {"C5", "C4", "C3", "BTCUSDT", "C2"}
    assert last["CASH"] == 0 and np.isclose(last["C5"], 0.2)


def test_c1_fewer_candidates_equal_weight():
    d = pd.date_range("2019-01-01", periods=260, freq="D")
    closes = pd.DataFrame({"BTCUSDT": np.linspace(100, 300, len(d)), "A": np.linspace(1, 2, len(d))},
                          index=d)
    elig = pd.DataFrame(True, index=d, columns=closes.columns)
    w = crypto.c1_weights(closes, elig)
    last = w.iloc[-1]
    assert np.isclose(last["A"], 0.5) and np.isclose(last["BTCUSDT"], 0.5) and last["CASH"] == 0


def test_c1_btc_regime_cash():
    closes, elig = _coins(btc_drift=-0.004)
    btc = closes["BTCUSDT"]
    assert (btc.iloc[200:] < btc.rolling(200).mean().iloc[200:]).mean() > 0.95
    w = crypto.c1_weights(closes, elig)
    regime = btc > btc.rolling(200).mean()
    off = ~regime.reindex(w.index)
    assert off.sum() > 10
    assert (w.loc[off, "CASH"] == 1.0).all()
    assert (w.loc[off].drop(columns="CASH") == 0).all().all()
    # warm-up (no 200-day SMA yet) is cash too
    assert (w.loc[: btc.index[198], "CASH"] == 1.0).all()


def test_c2_weights_legs():
    closes, elig = _coins()
    w = crypto.c2_weights(closes, elig)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert (w["BTCUSDT"] == 0.5).all()
    shorts = w.drop(columns=["BTCUSDT", "CASH"])
    assert (shorts <= 0).all().all()
    full = (shorts < 0).sum(axis=1) == 5
    assert full.any()
    np.testing.assert_allclose(shorts[full].sum(axis=1), -0.5)
    np.testing.assert_allclose(w.loc[full, "CASH"], 1.0)
    # shorted coins are the bottom 5 eligible by 30d return
    d = full[full].index[-1]
    r30 = closes.loc[:d].apply(lambda s: s.dropna() / s.dropna().shift(30) - 1).loc[d]
    cand = r30[elig.loc[d] & r30.notna()].drop("BTCUSDT", errors="ignore")
    assert set(shorts.loc[d][shorts.loc[d] < 0].index) == set(cand.nsmallest(5).index)


def test_c2_fewer_candidates():
    d = pd.date_range("2019-01-01", periods=60, freq="D")
    closes = pd.DataFrame({"BTCUSDT": np.linspace(100, 200, 60), "A": np.linspace(1, 2, 60),
                           "B": np.linspace(2, 1, 60)}, index=d)
    elig = pd.DataFrame(True, index=d, columns=closes.columns)
    w = crypto.c2_weights(closes, elig)
    last = w.iloc[-1]
    assert np.isclose(last["A"], -0.25) and np.isclose(last["B"], -0.25)
    assert np.isclose(last["CASH"], 1.0) and np.isclose(last["BTCUSDT"], 0.5)
    first = w.iloc[0]  # no coin has 30 prior bars yet -> no shorts, cash keeps the sum at 1
    assert np.isclose(first["CASH"], 0.5) and first.drop(["BTCUSDT", "CASH"]).eq(0).all()


def test_c3_gate_matches_c1():
    closes, elig = _coins(btc_drift=0.004)
    fng = _fng(closes.index)
    c1 = crypto.c1_weights(closes, elig)
    for mode, gate_fn in [("contrarian", lambda v: v <= 25), ("confirming", lambda v: v >= 55)]:
        c3 = crypto.c3_weights(closes, elig, fng, mode)
        lagged = fng.reindex(c3.index - pd.Timedelta(days=1)).to_numpy()
        gate = gate_fn(lagged)
        assert gate.any() and (~gate).any()
        pd.testing.assert_frame_equal(c3[gate], c1[gate])
        assert (c3.loc[~gate, "CASH"] == 1.0).all()
    with pytest.raises(ValueError):
        crypto.c3_weights(closes, elig, fng, "sideways")


def test_c3_fng_lag():
    closes, elig = _coins(btc_drift=0.004)
    c1 = crypto.c1_weights(closes, elig)
    invested = c1.index[c1["CASH"] < 1]
    t = invested[-1]
    fng = pd.Series(50.0, index=closes.index)  # contrarian gate closed everywhere
    base = crypto.c3_weights(closes, elig, fng, "contrarian")
    assert base.loc[t, "CASH"] == 1.0
    same_day = fng.copy()
    same_day.loc[t] = 10.0  # planted on day t: must not affect the decision at t
    pd.testing.assert_frame_equal(crypto.c3_weights(closes, elig, same_day, "contrarian"), base)
    prev_day = fng.copy()
    prev_day.loc[t - pd.Timedelta(days=1)] = 10.0  # value dated t-1 opens the gate at t
    opened = crypto.c3_weights(closes, elig, prev_day, "contrarian")
    pd.testing.assert_series_equal(opened.loc[t], c1.loc[t])
    # a missing FGI value closes the gate
    missing = prev_day.drop(t - pd.Timedelta(days=1))
    assert crypto.c3_weights(closes, elig, missing, "contrarian").loc[t, "CASH"] == 1.0


# ---------------------------------------------------------------- rebalance engine

def test_c2_short_loses_when_price_rises_and_funding():
    d = pd.date_range("2020-01-01", periods=10, freq="D")
    close_x = np.array([100, 100, 101, 103, 104, 106, 108, 110, 110, 110.0])
    opens = pd.DataFrame({"X": np.r_[100.0, close_x[:-1]]}, index=d)  # open = previous close
    closes = pd.DataFrame({"X": close_x}, index=d)
    w = pd.DataFrame({"X": [-0.1, 0.0], "CASH": [1.1, 1.0]}, index=[d[0], d[7]])
    fund = 0.10
    eq = run_weights(opens, closes, w, cost_per_side=COST, short_funding_annual=fund)
    # day1 open: short 0.1 at 100 (cost on 0.1); funding each close days 1..7 on current notional
    v1 = 1 - 0.1 * COST
    notional = 0.1 * v1 * close_x[1:8] / 100
    funding = (fund / 365 * notional).sum()
    value_before_cover = v1 - 0.1 * v1 * (110 / 100 - 1) - funding
    cover_cost = 0.1 * v1 * 1.1 * COST
    expected = value_before_cover - cover_cost
    assert eq.index[0] == d[1]
    assert eq.iloc[-1] == pytest.approx(expected, rel=1e-12)
    # ~ -0.1 x 10% - funding - costs
    assert eq.iloc[-1] - 1 == pytest.approx(-0.01 - funding - 0.1 * COST - cover_cost, abs=1e-5)
    assert eq.attrs["worst_short_loss"] == pytest.approx(0.10)
    assert set(eq.attrs) == {"worst_short_loss"}


def test_cash_turnover_free():
    d = pd.date_range("2020-01-01", periods=4, freq="D")
    px = pd.DataFrame({"A": [10.0] * 4}, index=d)
    w = pd.DataFrame({"A": [0.5, 0.0], "CASH": [0.5, 1.0]}, index=[d[0], d[1]])
    eq = run_weights(px, px, w, cost_per_side=COST)
    v1 = 1 - 0.5 * COST  # buying A costs; the cash half is free
    assert eq.iloc[0] == pytest.approx(v1)
    assert eq.iloc[-1] == pytest.approx(v1 - 0.5 * v1 * COST)  # only the sold A leg costs
    assert np.isnan(eq.attrs["worst_short_loss"])


def test_cash_is_constant_price_even_if_present():
    d = pd.date_range("2020-01-01", periods=4, freq="D")
    px = pd.DataFrame({"CASH": [1.0, 2.0, 3.0, 4.0]}, index=d)
    w = pd.DataFrame({"CASH": [1.0]}, index=[d[0]])
    eq = run_weights(px, px, w, cost_per_side=COST)
    assert (eq == 1.0).all()


def test_delisted_coin_ffilled_no_nan():
    d = pd.date_range("2020-01-01", periods=9, freq="D")
    cx = [10, 10, 11, 12, np.nan, np.nan, np.nan, np.nan, np.nan]
    ox = [10, 10, 10.5, 11.5, np.nan, np.nan, np.nan, np.nan, np.nan]
    closes = pd.DataFrame({"X": cx, "Y": [5.0] * 9}, index=d)
    opens = pd.DataFrame({"X": ox, "Y": [5.0] * 9}, index=d)
    w = pd.DataFrame({"X": [1.0, 0.0], "Y": [0.0, 1.0], "CASH": [0.0, 0.0]}, index=[d[0], d[5]])
    eq = run_weights(opens, closes, w, cost_per_side=COST)
    assert eq.notna().all() and len(eq) == 8
    v1 = 1 - COST
    held = v1 * 12 / 10  # carried at its last close
    assert eq.loc[d[3]:d[5]].to_numpy() == pytest.approx([held] * 3)
    # sold at the next rebalance (open of d6) at the last close, bought Y
    assert eq.iloc[-1] == pytest.approx(held - 2 * held * COST)


def test_trend_run_portfolio_still_delegates():
    from swingbot.backtest.trend import run_portfolio
    d = pd.bdate_range("2020-01-01", periods=4)
    px = pd.DataFrame({"A": [10.0] * 4, "SHY": [10.0] * 4}, index=d)
    w = pd.DataFrame({"A": [1.0, 0.0], "SHY": [0.0, 1.0]}, index=[d[0], d[1]])
    pd.testing.assert_series_equal(run_portfolio(px, px, w, cost_per_side=0.001),
                                   run_weights(px, px, w, cost_per_side=0.001, cash_cols=()),
                                   check_names=False)


# ---------------------------------------------------------------- stocks

def _s1_inputs(n=320, seed=0, index_up=True):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2010-01-04", periods=n)
    close = 50 * np.cumprod(1 + 0.004 + rng.normal(0, 0.003, n))
    close[260:266] *= np.cumprod([0.96] * 6)  # a sharp dip well above the 200-day SMA
    close[266:] *= 0.96 ** 6
    openp = np.r_[close[0], close[:-1]] * 1.001
    bars = {"AAA": pd.DataFrame({"open": openp, "high": close * 1.01, "low": close * 0.99,
                                 "close": close, "volume": 1e6}, index=d)}
    di = pd.bdate_range("2009-01-01", periods=n + 260)
    trend = np.linspace(100, 300, len(di)) if index_up else np.linspace(300, 100, len(di))
    index_close = {"^GSPC": pd.Series(trend, index=di)}
    return bars, index_close, {"AAA": "^GSPC"}


def test_s1_signal_and_returns_net_cost():
    bars, idx, home = _s1_inputs()
    f = stocks.s1_frame(bars, idx, home, cost=0.001)
    assert list(f.columns) == ["date", "asset", "s1", "ret_long_10", "split"]
    b = bars["AAA"]
    r = ind.rsi(b["close"], 14)
    assert (r.iloc[262:266] < 30).any()
    fired = f.set_index("date")["s1"]
    expect = (r < 30) & (b["close"] > b["close"].rolling(200).mean())
    pd.testing.assert_series_equal(fired, expect.rename("s1"), check_names=False, check_freq=False)
    assert fired.iloc[262:266].any() and not fired.iloc[:200].any()
    t = fired[fired].index[0]
    i = b.index.get_loc(t)
    want = b["close"].iloc[i + 10] / b["open"].iloc[i + 1] - 1 - 0.001
    assert f.set_index("date").loc[t, "ret_long_10"] == pytest.approx(want)
    assert f["ret_long_10"].iloc[-10:].isna().all()
    assert f["split"].isna().all()
    # home index in a downtrend switches the signal off
    bars2, idx2, home2 = _s1_inputs(index_up=False)
    assert not stocks.s1_frame(bars2, idx2, home2, cost=0.001)["s1"].any()


def test_s1_no_lookahead():
    rng = np.random.default_rng(3)
    d = pd.bdate_range("2010-01-04", periods=500)
    bars, home = {}, {}
    for j, ix in enumerate(["^GSPC", "^GSPC", "^N225"]):
        c = 20 * np.cumprod(1 + 0.0005 + rng.normal(0, 0.02, len(d)))
        bars[f"S{j}"] = pd.DataFrame({"open": c * 1.001, "high": c, "low": c, "close": c,
                                      "volume": 1.0}, index=d)
        home[f"S{j}"] = ix
    idx = {k: pd.Series(100 * np.cumprod(1 + 0.0003 + rng.normal(0, 0.01, len(d))), index=d)
           for k in ["^GSPC", "^N225"]}
    cut = d[380]
    full = stocks.s1_frame(bars, idx, home, cost=0.001)
    part = stocks.s1_frame({k: v.loc[:cut] for k, v in bars.items()},
                           {k: v.loc[:cut] for k, v in idx.items()}, home, cost=0.001)
    a = full[full["date"] <= cut].reset_index(drop=True)
    pd.testing.assert_series_equal(a["s1"], part["s1"])
    assert a["s1"].any()


def test_s2_no_lookahead_and_sums():
    closes = _stocks(800)
    regime = pd.DataFrame(True, index=closes.index, columns=closes.columns)
    regime.iloc[500:540, :20] = False
    full = stocks.s2_weights(closes, regime)
    cut = closes.index[650]
    part = stocks.s2_weights(closes.loc[:cut], regime.loc[:cut])
    # month-end rows only (the truncated tail month is a partial month and not comparable)
    common = part.index[part.index < cut.to_period("M").start_time]
    pd.testing.assert_frame_equal(full.loc[common], part.loc[common])
    np.testing.assert_allclose(full.sum(axis=1), 1.0)
    held = (full.drop(columns="CASH") > 0).sum(axis=1)
    assert (held[full["CASH"] == 0] >= 5).all() and (held <= 6).all()
    assert ((full["CASH"] > 0) & (full["CASH"] < 1)).any()  # some slots switched to cash


def test_s2_monthly_union_calendar():
    closes = _stocks(400)
    regime = pd.DataFrame(True, index=closes.index, columns=closes.columns)
    w = stocks.s2_weights(closes, regime)
    ends = closes.index.to_series().groupby(closes.index.to_period("M")).max()
    assert list(w.index) == list(ends)


def test_s2_min_valid_cash():
    closes = _stocks(600, k=40)
    regime = pd.DataFrame(True, index=closes.index, columns=closes.columns)
    w = stocks.s2_weights(closes, regime)  # 40 < 50 valid stocks -> always cash
    assert (w["CASH"] == 1.0).all()
    w10 = stocks.s2_weights(closes, regime, min_valid=10)
    assert (w10["CASH"] < 1).any()
    # before 252 prior bars nobody is valid
    assert (w10.loc[: closes.index[251], "CASH"] == 1.0).all()


def test_s2_momentum_ranking_and_count():
    closes = _ramp()
    d = closes.index
    regime = pd.DataFrame(True, index=d, columns=closes.columns)
    w = stocks.s2_weights(closes, regime)
    last = w.iloc[-1]
    held = last[last > 0].drop("CASH", errors="ignore")
    assert set(held.index) == {f"S{j:02d}" for j in range(54, 60)}  # top 10% of 60 = 6
    np.testing.assert_allclose(held, 1 / 6)


def _ramp(n=320, k=60, start="2005-01-03"):
    d = pd.bdate_range(start, periods=n)
    rates = np.linspace(-0.001, 0.002, k)
    return pd.DataFrame(10 * np.cumprod(1 + np.tile(rates, (len(d), 1)), axis=0), index=d,
                        columns=[f"S{j:02d}" for j in range(k)])


def test_s2_home_regime_per_stock():
    closes = _ramp()
    d = closes.index
    regime = pd.DataFrame(True, index=d, columns=closes.columns)
    regime[["S59", "S58"]] = False  # their home index is below its SMA
    last = stocks.s2_weights(closes, regime).iloc[-1]
    assert last["S59"] == 0 and last["S58"] == 0
    held = last[last > 0].drop("CASH")
    # top 6 are S54..S59; the two switched-off slots go to cash, nobody is promoted
    assert set(held.index) == {f"S{j:02d}" for j in range(54, 58)}
    np.testing.assert_allclose(held, 1 / 6)
    assert last["CASH"] == pytest.approx(2 / 6)
    # every home index off -> all cash
    off = pd.DataFrame(False, index=d, columns=closes.columns)
    assert stocks.s2_weights(closes, off)["CASH"].eq(1.0).all()


def test_s2_two_home_indices_crash_switch():
    closes = _ramp()
    d = closes.index
    home = {c: ("^GDAXI" if int(c[1:]) % 2 else "^FTSE") for c in closes.columns}
    up = pd.Series(np.linspace(100, 200, len(d) + 300), index=pd.bdate_range("2003-11-24", periods=len(d) + 300))
    down = pd.Series(np.linspace(200, 100, len(up)), index=up.index)
    regime = stocks.home_regime_frame(closes, {"^FTSE": up, "^GDAXI": down}, home)
    last = stocks.s2_weights(closes, regime).iloc[-1]
    top6 = [f"S{j:02d}" for j in range(54, 60)]
    even = [c for c in top6 if home[c] == "^FTSE"]
    assert set(last[last > 0].drop("CASH").index) == set(even)  # odd (DAX-off) slots not refilled
    np.testing.assert_allclose(last[even], 1 / 6)
    assert last["CASH"] == pytest.approx(3 / 6)


def test_s2_month_end_holiday_keeps_group_ranked():
    closes = _ramp(330)
    ends = closes.index.to_series().groupby(closes.index.to_period("M")).max()
    t = ends.iloc[-2]  # a complete month end with 252+ bars
    group = [f"S{j:02d}" for j in range(30, 60)]  # includes the top momentum stocks
    holiday = closes.copy()
    holiday.loc[t, group] = np.nan  # local holiday on the union month end
    regime = pd.DataFrame(True, index=closes.index, columns=closes.columns)
    w_hol = stocks.s2_weights(holiday, regime).loc[t]
    assert w_hol["CASH"] == 0
    assert set(w_hol[w_hol > 0].index) == {f"S{j:02d}" for j in range(54, 60)}
    ew = stocks.equal_weight_hold(holiday, min_valid=50).loc[t]
    assert (ew.drop("CASH") > 0).sum() == 60


def test_own_series_change_staleness_limit():
    closes = _ramp(330, k=3)
    closes.iloc[-7:, 0] = np.nan  # S00 stops trading 7 bars before the end
    v = own_series_change(closes, 21, 252)
    assert v["S00"].iloc[-7:-2].notna().all()  # 1..5 bars stale: last own value carried
    assert v["S00"].iloc[-7] == v["S00"].iloc[-8]
    assert v["S00"].iloc[-2:].isna().all()  # more than 5 bars stale: missing
    assert v["S01"].iloc[-1] == pytest.approx(closes["S01"].iloc[-22] / closes["S01"].iloc[-253] - 1)


def test_s2_stale_stock_drops_out():
    closes = _ramp(330)
    ends = closes.index.to_series().groupby(closes.index.to_period("M")).max()
    t = ends.iloc[-2]
    i = closes.index.get_loc(t)
    closes.iloc[i - 6:i + 1, closes.columns.get_loc("S59")] = np.nan  # no bar for 7 bars
    regime = pd.DataFrame(True, index=closes.index, columns=closes.columns)
    row = stocks.s2_weights(closes, regime).loc[t]
    assert row["S59"] == 0
    assert set(row[row > 0].index) == {f"S{j:02d}" for j in range(54, 59)}  # 59 valid -> k = 5


def test_run_weights_rows_must_sum_to_one():
    d = pd.date_range("2020-01-01", periods=4, freq="D")
    px = pd.DataFrame({"A": [10.0] * 4}, index=d)
    with pytest.raises(ValueError):
        run_weights(px, px, pd.DataFrame({"A": [0.5], "CASH": [0.4]}, index=[d[0]]), COST)


def test_home_regime_frame_ffills_index():
    d = pd.bdate_range("2010-01-04", periods=300)
    closes = pd.DataFrame({"A": 1.0, "B": 1.0}, index=d)
    up = pd.Series(np.linspace(100, 200, 300), index=d)
    down = pd.Series(np.linspace(200, 100, 300), index=d).iloc[::2]  # sparser calendar
    reg = stocks.home_regime_frame(closes, {"^UP": up, "^DN": down}, {"A": "^UP", "B": "^DN"})
    assert reg["A"].iloc[250:].all() and not reg["A"].iloc[:199].any()
    assert not reg["B"].any()
    assert reg.dtypes.eq(bool).all()


def test_equal_weight_hold():
    closes = _stocks(600)
    w = stocks.equal_weight_hold(closes, min_valid=50)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    inv = w["CASH"] < 1
    assert inv.any()
    row = w[inv].iloc[-1].drop("CASH")
    held = row[row > 0]
    np.testing.assert_allclose(held, 1 / len(held))
    assert len(held) >= 50
    regime = pd.DataFrame(True, index=closes.index, columns=closes.columns)
    s2 = stocks.s2_weights(closes, regime)
    assert list(w.index) == list(s2.index)
    pd.testing.assert_series_equal(w["CASH"] == 1, s2["CASH"] == 1)
