import numpy as np
import pandas as pd
import pytest

from swingbot import indicators as ind


def make_bars(n=60, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1, n)
    vol = rng.uniform(1000, 2000, n)
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def test_sma_ema_values():
    s = pd.Series(np.arange(1.0, 11.0))
    assert ind.sma(s, 3).iloc[-1] == 9
    assert ind.sma(s, 3).iloc[:2].isna().all()
    exp = [s.iloc[0]]
    a = 2 / (3 + 1)
    for x in s.iloc[1:]:
        exp.append(a * x + (1 - a) * exp[-1])
    got = ind.ema(s, 3)
    assert np.allclose(got.to_numpy()[2:], exp[2:])


def test_rsi_all_up_is_100():
    s = pd.Series(np.arange(1.0, 40.0))
    r = ind.rsi(s, 14)
    assert r.iloc[-1] == 100
    assert r.iloc[:14].isna().all()


def test_rsi_flat_is_50():
    r = ind.rsi(pd.Series([5.0] * 30), 14)
    assert r.iloc[-1] == 50


def test_rsi_known_value():
    # Classic Wilder / StockCharts example series
    closes = [44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245,
              45.8433, 46.0826, 45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0028,
              46.0328, 46.4116, 46.2222, 45.6439, 46.2122, 46.2521, 45.7137, 46.4515,
              45.7835, 45.3548, 44.0288, 44.1783, 44.2220, 44.5659, 43.4205, 42.6628,
              43.1314]
    s = pd.Series(closes)
    r = ind.rsi(s, 14)
    # Published RSI(14) values for this series (StockCharts "RSI" ChartSchool
    # spreadsheet, Wilder smoothing seeded with the SMA of the first 14 changes).
    published = [70.53, 66.32, 66.55, 69.41, 66.36, 57.97]
    assert r.iloc[:14].isna().all()
    assert [round(v, 2) for v in r.iloc[14:20]] == published
    assert round(r.iloc[-1], 2) == 37.77


def test_atr_uses_prior_close():
    df = pd.DataFrame(
        {"open": [10, 10, 15], "high": [11, 11, 16], "low": [9, 9, 14.5],
         "close": [10, 10, 15.5], "volume": [1, 1, 1]},
        index=pd.date_range("2020-01-01", periods=3, name="date"),
    ).astype(float)
    tr = ind.true_range(df)
    assert tr.iloc[2] == 16 - 10  # gap up: high - prev close
    assert tr.iloc[1] == 2
    a = ind.atr(df, 2)
    assert np.isnan(a.iloc[0])
    assert a.iloc[1] == pytest.approx(2.0)  # SMA seed of TR [2, 2]
    assert a.iloc[2] == pytest.approx(0.5 * 6 + 0.5 * a.iloc[1])


def test_bollinger_ddof0():
    s = pd.Series([1.0, 2, 3, 4, 5, 6])
    mid, up, lo = ind.bollinger(s, 3, 2.0)
    assert mid.iloc[-1] == 5
    sd = np.std([4, 5, 6])  # ddof=0
    assert up.iloc[-1] == pytest.approx(5 + 2 * sd)
    assert lo.iloc[-1] == pytest.approx(5 - 2 * sd)


def test_keltner_shapes():
    df = make_bars()
    mid, up, lo = ind.keltner(df)
    assert (up[up.notna()] >= mid[up.notna()]).all() and (lo[lo.notna()] <= mid[lo.notna()]).all()


def test_macd_shapes():
    df = make_bars()
    line, sig = ind.macd(df["close"])
    assert len(line) == len(sig) == len(df)
    assert line.index.equals(df.index)
    assert line.iloc[-1] == pytest.approx(
        ind.ema(df["close"], 12).iloc[-1] - ind.ema(df["close"], 26).iloc[-1]
    )


def test_adx_range_0_100():
    df = make_bars(200)
    a, p, m = ind.adx(df)
    for x in (a, p, m):
        v = x.dropna()
        assert len(v) > 100 and v.between(0, 100).all()
    assert a.iloc[:14].isna().all()


def test_stoch_range_0_100():
    df = make_bars()
    k, d = ind.stochastic(df)
    assert k.dropna().between(0, 100).all() and d.dropna().between(0, 100).all()
    assert d.iloc[-1] == pytest.approx(k.iloc[-3:].mean())


def test_obv_ad_zscore():
    df = pd.DataFrame(
        {"open": 1.0, "high": [2, 3, 3, 2.0], "low": [1, 1, 1, 1.0],
         "close": [1.5, 2.0, 1.5, 1.5], "volume": [10, 20, 30, 40.0]},
        index=pd.date_range("2020-01-01", periods=4, name="date"),
    )
    assert ind.obv(df).tolist() == [0, 20, -10, -10]
    ad = ind.ad_line(df)
    assert ad.iloc[0] == 0  # clv = 0
    assert ad.iloc[1] == pytest.approx(20 * ((2 - 1) - (3 - 2)) / 2)
    z = ind.zscore(pd.Series([1.0, 2, 3, 4]), 3)
    assert z.iloc[-1] == pytest.approx((4 - 3) / np.std([2, 3, 4]))


def test_donchian_excludes_today():
    h = pd.Series([1.0, 2, 3, 5, 4])
    dh = ind.donchian_high(h, 3)
    assert dh.iloc[3] == 3  # today's 5 not included
    assert dh.iloc[4] == 5
    assert np.isnan(dh.iloc[2])
    low = pd.Series([5.0, 4, 3, 1, 2])
    assert ind.donchian_low(low, 3).iloc[3] == 3


def test_swing_low_known():
    low = pd.Series([5.0, 4, 3, 4, 5, 6, 7])
    sl = ind.swing_low(low, 2)
    assert sl.iloc[:4].isna().all()
    assert sl.iloc[4] == 3 and sl.iloc[6] == 3  # confirmed at t=4, forward-filled
    high = pd.Series([1.0, 2, 3, 2, 1, 0, 1])
    assert ind.swing_high(high, 2).iloc[4] == 3
    # ties are not pivots
    assert ind.swing_low(pd.Series([5.0, 3, 3, 4, 5]), 2).isna().all()


@pytest.mark.parametrize("fn", ["swing_low", "swing_high"])
def test_swing_no_future(fn):
    df = make_bars(80, seed=3)
    s = df["low"] if fn == "swing_low" else df["high"]
    full = getattr(ind, fn)(s)
    assert full.notna().sum() > 5
    for t in range(len(s)):
        part = getattr(ind, fn)(s.iloc[: t + 1])
        a, b = part.iloc[-1], full.iloc[t]
        assert (np.isnan(a) and np.isnan(b)) or a == b


@pytest.mark.parametrize("fn", ["supertrend_dir", "psar_dir"])
def test_path_dependent_no_future(fn):
    df = make_bars(120, seed=5)
    full = getattr(ind, fn)(df)
    assert set(full.dropna().unique()) <= {-1.0, 1.0} and full.notna().sum() > 80
    assert full.dtype == float
    first = full.first_valid_index()
    pos = full.index.get_loc(first)
    assert full.iloc[:pos].isna().all()  # NaN (never 0) before initialisation
    d = full.diff()
    assert np.isnan(d.iloc[pos])  # no false flip at initialisation
    for t in range(0, len(df), 7):
        a, b = getattr(ind, fn)(df.iloc[: t + 1]).iloc[-1], full.iloc[t]
        assert (np.isnan(a) and np.isnan(b)) or a == b


def test_psar_trend_direction():
    n = 40
    idx = pd.date_range("2020-01-01", periods=n, name="date")
    c = np.concatenate([np.linspace(100, 140, 20), np.linspace(140, 100, 20)])
    df = pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1.0}, index=idx)
    d = ind.psar_dir(df)
    assert d.iloc[10] == 1 and d.iloc[-1] == -1
    st = ind.supertrend_dir(df, n=5, mult=1.0)
    assert st.iloc[15] == 1 and st.iloc[-1] == -1


def _all_funcs(df):
    c, h, l = df["close"], df["high"], df["low"]
    return [
        ind.sma(c, 3), ind.ema(c, 3), ind.rsi(c), ind.atr(df), *ind.bollinger(c),
        *ind.keltner(df), *ind.macd(c), *ind.adx(df), *ind.stochastic(df),
        ind.obv(df), ind.ad_line(df), ind.zscore(c), ind.donchian_high(h, 3),
        ind.donchian_low(l, 3), ind.supertrend_dir(df), ind.psar_dir(df),
        ind.swing_low(l), ind.swing_high(h), ind.true_range(df),
    ]


@pytest.mark.parametrize("n", [0, 1, 3, 10, 40])
def test_nan_and_short_input(n):
    df = make_bars(max(n, 1)).iloc[:n].copy()
    if n >= 5:
        df.iloc[4, df.columns.get_loc("close")] = np.nan
    for out in _all_funcs(df):
        assert len(out) == n
        assert out.index.equals(df.index)
