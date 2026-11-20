import numpy as np
import pandas as pd
import pytest

from swingbot import indicators as ind
from swingbot.signals import (
    REGISTRY,
    SIGNAL_INFO,
    compute_signals,
    cross_sectional_signals,
)
from swingbot.signals._swings import confirmed_pivots

EXPECTED = {
    # trend
    "ma20_50_cross_up": ("trend", "long"),
    "ma20_50_cross_down": ("trend", "short"),
    "ma50_200_cross_up": ("trend", "long"),
    "ma50_200_cross_down": ("trend", "short"),
    "close_above_ma200_cross": ("trend", "long"),
    "macd_cross_up": ("trend", "long"),
    "macd_cross_down": ("trend", "short"),
    "adx_trend_up": ("trend", "long"),
    "supertrend_flip_up": ("trend", "long"),
    "supertrend_flip_down": ("trend", "short"),
    "psar_flip_up": ("trend", "long"),
    # pullback
    "pullback_ma20_uptrend": ("pullback", "long"),
    "rsi14_pullback_uptrend": ("pullback", "long"),
    "rsi2_oversold": ("pullback", "long"),
    "fib_retrace_50_618": ("pullback", "long"),
    # breakout
    "donchian20_breakout": ("breakout", "long"),
    "donchian20_breakdown": ("breakout", "short"),
    "donchian55_breakout": ("breakout", "long"),
    "squeeze_release_up": ("breakout", "long"),
    "volume_breakout": ("breakout", "long"),
    "high_52w": ("breakout", "long"),
    # meanrev
    "bb_lower_close": ("meanrev", "long"),
    "bb_upper_close": ("meanrev", "short"),
    "rsi14_oversold": ("meanrev", "long"),
    "rsi14_overbought": ("meanrev", "short"),
    "zscore_below_2": ("meanrev", "long"),
    "stoch_oversold_cross": ("meanrev", "long"),
    "gap_down_fill_setup": ("meanrev", "long"),
    # momentum
    "roc20_accel": ("momentum", "long"),
    # divergence
    "rsi_bull_divergence": ("divergence", "long"),
    "macd_bull_divergence": ("divergence", "long"),
    # volume
    "obv_leads_price": ("volume", "long"),
    "close_above_avwap_swinglow": ("volume", "long"),
    "ad_rising": ("volume", "long"),
    # sr
    "swing_low_bounce": ("sr", "long"),
    "pivot_support_hold": ("sr", "long"),
    # candles
    "bullish_engulfing": ("candles", "long"),
    "hammer": ("candles", "long"),
    "inside_bar_breakout": ("candles", "long"),
    "morning_star": ("candles", "long"),
}
FAMILIES = {"trend", "pullback", "breakout", "meanrev", "momentum", "divergence", "volume", "sr", "candles"}


# ---------------------------------------------------------------- helpers
def random_walk(n=600, seed=0):
    # fat-tailed wicks and volume so rare patterns (volume spikes, hammers) occur
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + rng.exponential(0.006, n))
    low = np.minimum(open_, close) * (1 - rng.exponential(0.008, n))
    vol = 2000 * np.exp(rng.normal(0, 0.5, n))
    idx = pd.bdate_range("2000-01-03", periods=n, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def bars(close, open_=None, high=None, low=None, volume=None, pad=0.5):
    close = np.asarray(close, float)
    n = len(close)
    open_ = close.copy() if open_ is None else np.asarray(open_, float)
    high = np.maximum(open_, close) + pad if high is None else np.asarray(high, float)
    low = np.minimum(open_, close) - pad if low is None else np.asarray(low, float)
    volume = np.full(n, 1000.0) if volume is None else np.asarray(volume, float)
    idx = pd.bdate_range("2000-01-03", periods=n, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def fires(df, name):
    s = compute_signals(df)[name]
    return list(np.flatnonzero(s.to_numpy()))


# ---------------------------------------------------------------- registry
def test_registry_complete():
    assert set(REGISTRY) == set(EXPECTED)
    for name, sig in REGISTRY.items():
        assert sig.name == name
        assert sig.family in FAMILIES
        assert sig.direction in ("long", "short")
        assert (sig.family, sig.direction) == EXPECTED[name], name


def test_signal_info_covers_registry_plus_cross_sectional():
    assert set(SIGNAL_INFO) == set(EXPECTED) | {"mom_12_1_top_decile"}
    assert SIGNAL_INFO["mom_12_1_top_decile"] == ("momentum", "long")
    for name, sig in REGISTRY.items():
        assert SIGNAL_INFO[name] == (sig.family, sig.direction)


def test_compute_signals_shape_and_dtype():
    df = random_walk()
    out = compute_signals(df)
    assert list(out.columns) == list(REGISTRY)
    assert out.index.equals(df.index)
    assert (out.dtypes == bool).all()


# ---------------------------------------------------------------- lookahead
_RW = random_walk()
_TS = sorted(np.random.default_rng(0).choice(np.arange(300, 600), size=20, replace=False))
_CACHE: dict = {}


def _full():
    if "full" not in _CACHE:
        _CACHE["full"] = compute_signals(_RW)
        _CACHE["trunc"] = {t: compute_signals(_RW.iloc[: t + 1]) for t in _TS}
    return _CACHE["full"], _CACHE["trunc"]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_no_lookahead(name):
    full, trunc = _full()
    for t in _TS:
        part = trunc[t]
        assert part[name].iloc[-1] == full[name].iloc[t], (name, t)
        # stronger: the whole truncated history must match the full run's prefix
        assert part[name].equals(full[name].iloc[: t + 1]), (name, t)


def test_random_walk_exercises_most_signals():
    full, _ = _full()
    fired = {c for c in full.columns if full[c].any()}
    # a lookahead test on signals that never fire proves little; the two rare
    # ones (adx_trend_up, morning_star) have positive hand fixtures below
    assert set(EXPECTED) - fired <= {"adx_trend_up", "morning_star"}, sorted(set(EXPECTED) - fired)


def test_short_history_all_false():
    flat = bars(np.full(50, 100.0), pad=0.0)
    out = compute_signals(flat)
    assert set(out.columns) == set(EXPECTED)
    assert not out.to_numpy().any()

    # random 50-bar frame: no exception, bool, and long-lookback signals all False
    rw = compute_signals(random_walk(50))
    assert (rw.dtypes == bool).all()
    long_lookback = [
        "ma20_50_cross_up", "ma20_50_cross_down", "ma50_200_cross_up", "ma50_200_cross_down",
        "close_above_ma200_cross", "pullback_ma20_uptrend", "rsi14_pullback_uptrend",
        "rsi2_oversold", "fib_retrace_50_618", "donchian55_breakout", "high_52w", "ad_rising",
    ]
    assert not rw[long_lookback].to_numpy().any()


def test_nan_bars_do_not_raise():
    df = random_walk(300)
    df.iloc[100:103] = np.nan
    out = compute_signals(df)
    assert (out.dtypes == bool).all()
    assert not out.iloc[100:103].to_numpy().any()


def test_confirmed_pivots_match_indicator_swings():
    df = random_walk(300)
    for low in (True, False):
        col = "low" if low else "high"
        conf = confirmed_pivots(df[col], k=2, low=low)
        vals = df[col].shift(2).where(conf).ffill()
        ref = ind.swing_low(df["low"]) if low else ind.swing_high(df["high"])
        pd.testing.assert_series_equal(vals, ref, check_names=False)


# ---------------------------------------------------------------- hand fixtures
def test_ma20_50_cross_up_fixture():
    close = np.concatenate([200.0 - np.arange(100), np.full(20, 300.0)])
    assert fires(bars(close), "ma20_50_cross_up") == [102]


def test_rsi2_oversold_fixture():
    up = 100.0 + np.arange(250)
    c = list(up) + [up[-1] - 5, up[-1] - 10] + list(up[-1] - 10 + np.arange(1, 11))
    assert fires(bars(c), "rsi2_oversold") == [251]


def test_donchian20_breakout_fixture():
    c = np.full(40, 100.0)
    c[30] = 105.0
    c[31:] = 100.5
    df = bars(c, high=np.full(40, 101.0), low=np.full(40, 99.0))
    df.loc[df.index[30], "high"] = 105.5
    assert fires(df, "donchian20_breakout") == [30]


def test_bb_lower_close_fixture():
    c = np.where(np.arange(40) % 2 == 0, 100.0, 101.0)
    c[30] = 95.0
    assert fires(bars(c), "bb_lower_close") == [30]


def test_roc20_accel_fixture():
    c = np.concatenate([np.full(40, 100.0), 101.0 + np.arange(20), np.full(20, 120.0)])
    assert fires(bars(c), "roc20_accel") == [40]


def test_rsi_bull_divergence_fixture():
    seg = [
        100 + 0.2 * np.arange(30),  # 0..29 gentle rise
    ]
    c = list(seg[0])
    for _ in range(10):  # 30..39 sharp drop -> pivot A at 39
        c.append(c[-1] - 3)
    for _ in range(10):  # 40..49 bounce
        c.append(c[-1] + 2)
    for _ in range(25):  # 50..74 slow drift to lower low -> pivot B at 74
        c.append(c[-1] - 0.85)
    for _ in range(10):  # 75..84 bounce
        c.append(c[-1] + 2)
    df = bars(c)
    r = ind.rsi(df["close"])
    assert df["low"].iloc[74] < df["low"].iloc[39] and r.iloc[74] > r.iloc[39]
    # pivot B (bar 74) is confirmed two bars later
    assert fires(df, "rsi_bull_divergence") == [76]


def test_obv_leads_price_fixture():
    c = list(100 + 0.5 * np.arange(31))  # 0..30 rising, peak at 30
    v = [1000.0] * 31
    for i in range(10):  # 31..40 alternate down (small vol) / up (big vol)
        if i % 2 == 0:
            c.append(c[-1] - 2)
            v.append(100.0)
        else:
            c.append(c[-1] + 1)
            v.append(1000.0)
    assert fires(bars(c, volume=v), "obv_leads_price") == [32, 34, 36, 38, 40]


def test_swing_low_bounce_fixture():
    o, c = [], []
    prev = 111.0
    for _ in range(21):  # 0..20 falling to close 90 -> pivot low 89.8 at bar 20
        o.append(prev)
        prev -= 1
        c.append(prev)
    for _ in range(10):  # 21..30 rising, gap-up opens
        o.append(prev + 0.3)
        prev += 1
        c.append(prev)
    for _ in range(9):  # 31..39 falling to 91
        o.append(prev)
        prev -= 1
        c.append(prev)
    o.append(90.4)  # 40: bullish bar testing the swing low
    c.append(90.9)
    o.append(93.0)  # 41: gap up
    c.append(94.0)
    prev = 94.0
    for _ in range(8):  # 42..49 rising
        o.append(prev + 0.3)
        prev += 1
        c.append(prev)
    o, c = np.array(o), np.array(c)
    low = np.minimum(o, c) - 0.2
    low[40] = 90.0
    df = bars(c, open_=o, low=low)
    assert fires(df, "swing_low_bounce") == [40]


def test_bullish_engulfing_fixture():
    c = np.full(15, 100.0)
    o = c.copy()
    o[10], c[10] = 100.0, 99.0
    o[11], c[11] = 98.8, 100.5
    o[12:], c[12:] = 100.5, 100.5
    assert fires(bars(c, open_=o), "bullish_engulfing") == [11]


def test_morning_star_fixture():
    c = np.full(30, 100.0)
    o = c.copy()
    o[20], c[20] = 100.0, 99.0  # long bearish (ATR ~1)
    o[21], c[21] = 98.8, 98.9  # small star below the prior close
    o[22], c[22] = 99.0, 99.8  # bullish, above midpoint 99.5
    o[23:], c[23:] = 99.8, 99.8
    assert fires(bars(c, open_=o), "morning_star") == [22]


def test_hammer_fixture():
    c = np.full(30, 100.0)
    c[20:23] = [99.0, 98.0, 97.0]
    o = np.concatenate([[100.0], c[:-1]])  # each bar opens at the prior close
    o[:20] = 100.0
    h = np.maximum(o, c) + 0.5
    lo = np.minimum(o, c) - 0.5
    o[23], c[23], h[23], lo[23] = 96.8, 97.0, 97.1, 96.0
    o[24:], c[24:] = 97.0, 97.0
    h[24:], lo[24:] = 97.5, 96.5
    assert fires(bars(c, open_=o, high=h, low=lo), "hammer") == [23]


def test_adx_trend_up_fixture():
    # long decline (-DI dominant, ADX high) then a sharp sustained rally
    c = np.concatenate([200.0 - 1.0 * np.arange(60), 140.0 + 3.0 * np.arange(1, 21)])
    df = bars(c)
    a, pdi, mdi = ind.adx(df)
    got = fires(df, "adx_trend_up")
    assert len(got) == 1
    t = got[0]
    assert pdi.iloc[t] > mdi.iloc[t] and pdi.iloc[t - 1] <= mdi.iloc[t - 1] and a.iloc[t] > 25


def test_avwap_no_fire_on_confirmation_then_same_anchor_cross():
    # swing low (bar 10, heavy volume, high close) confirmed at bar 12 while price
    # stays below the new-anchor AVWAP -> no fire; real cross at bar 15 -> fire
    o = [111.0 - i for i in range(10)]
    c = [110.0 - i for i in range(10)]
    h = [x + 0.5 for x in o]
    lo = [x - 0.5 for x in c]
    v = [1000.0] * 10
    o.append(99.0); c.append(100.0); h.append(101.0); lo.append(90.0); v.append(20000.0)
    for x in (95.0, 95.5, 96.0, 96.5):
        o.append(x - 0.2); c.append(x); h.append(x + 0.3); lo.append(x - 1.0); v.append(1000.0)
    o.append(96.5); c.append(99.0); h.append(99.2); lo.append(96.3); v.append(1000.0)
    for x in (99.5, 100.0, 100.5, 101.0):
        o.append(x - 0.2); c.append(x); h.append(x + 0.3); lo.append(x - 0.4); v.append(1000.0)
    df = bars(c, open_=o, high=h, low=lo, volume=v)
    assert list(np.flatnonzero(confirmed_pivots(df["low"]).to_numpy())) == [12]
    assert fires(df, "close_above_avwap_swinglow") == [15]


def test_avwap_anchor_reset_alone_is_not_a_cross():
    # old anchor (bar 5) VWAP is far above price after a high-volume rally; a new
    # swing low (bar 23) is confirmed at bar 25 with close already above the
    # same-anchor VWAP on bar 24 -> mixing anchors would fire at 25; must not
    c = np.array([100, 98, 96, 94, 92, 90, 93, 97, 102, 108, 115, 122, 128, 130,
                  128, 125, 121, 117, 112, 107, 103, 100, 98, 97,
                  99, 101, 103, 105, 107, 109], float)
    v = np.full(len(c), 1000.0)
    v[8:14] = 8000.0
    df = bars(c, volume=v, pad=0.3)
    assert list(np.flatnonzero(confirmed_pivots(df["low"]).to_numpy())) == [7, 25]
    assert fires(df, "close_above_avwap_swinglow") == []


# ---------------------------------------------------------------- cross-sectional
def test_cross_sectional_top_decile():
    n_assets, n_bars = 20, 420
    idx = pd.bdate_range("2010-01-01", periods=n_bars, name="date")
    t = np.arange(n_bars)
    closes = pd.DataFrame(
        {f"A{i:02d}": 100 * np.exp(0.0002 * (i + 1) * t) for i in range(n_assets)}, index=idx
    )
    # A00 explodes only within the most recent month: excluded by the 12-1 skip
    closes.loc[idx[-15:], "A00"] *= 10.0
    # A05 has short history: never eligible
    closes.loc[idx[:200], "A05"] = np.nan

    out = cross_sectional_signals(closes)["mom_12_1_top_decile"]
    assert out.shape == closes.shape and (out.dtypes == bool).all()

    # first rebalance where the prior month's last bar has >= 252 bars of history
    month = idx.to_period("M")
    firsts = [i for i in range(1, n_bars) if month[i] != month[i - 1]]
    first_ok = next(i for i in firsts if i - 1 >= 252)
    assert not out.iloc[:first_ok].to_numpy().any()

    held = out.iloc[first_ok:]
    assert held[["A18", "A19"]].all().all()
    assert not held.drop(columns=["A18", "A19"]).to_numpy().any()
    assert out.sum(axis=1).iloc[first_ok:].eq(2).all()


def test_cross_sectional_rebalance_uses_prior_close_only():
    idx = pd.bdate_range("2010-01-01", periods=420, name="date")
    t = np.arange(420)
    closes = pd.DataFrame(
        {f"A{i:02d}": 100 * np.exp(0.0002 * (i + 1) * t) for i in range(20)}, index=idx
    )
    full = cross_sectional_signals(closes)["mom_12_1_top_decile"]
    for cut in (300, 330, 361, 400):
        part = cross_sectional_signals(closes.iloc[: cut + 1])["mom_12_1_top_decile"]
        assert part.equals(full.iloc[: cut + 1])
