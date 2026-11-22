"""Funding-rate carry: F1/F2 weight builders and the carry engine (no network)."""
import numpy as np
import pandas as pd
import pytest

from swingbot.strategies import CASH
from swingbot.strategies.carry import f1_weights, f2_weights, run_carry

SPOT_COST, PERP_COST = 0.001, 0.0005
ENTRY = 1 - 0.5 * SPOT_COST - 0.5 * PERP_COST  # equity after entering a full carry from cash


# ---------------------------------------------------------------- fixtures

def _frames(coins=("XUSDT",), start="2021-01-03", n=4, px=100.0, fund=0.0001):
    """Flat prices on both legs (open = close = px) and constant daily funding."""
    d = pd.date_range(start, periods=n, freq="D", name="date")
    flat = pd.DataFrame(px, index=d, columns=list(coins))
    funding = pd.DataFrame(fund, index=d, columns=list(coins))
    return flat.copy(), flat.copy(), flat.copy(), flat.copy(), funding


def _w(date, **weights):
    return pd.DataFrame([weights], index=pd.DatetimeIndex([pd.Timestamp(date)]), dtype=float)


def _run(so, sc, po, pc, f, w):
    return run_carry(so, sc, po, pc, f, w, spot_cost=SPOT_COST, perp_cost=PERP_COST)


# ---------------------------------------------------------------- run_carry

def test_hand_computed_one_coin():
    # Sunday signal, trade Monday open; both legs +10% on the last day; 0.01%/day for 3 days.
    so, sc, po, pc, f = _frames()
    sc.iloc[-1] = 110.0
    pc.iloc[-1] = 110.0
    eq = _run(so, sc, po, pc, f, _w("2021-01-03", XUSDT=1.0, CASH=0.0))
    assert list(eq.index) == list(sc.index[1:])  # starts at the first trade day
    assert eq.iloc[-1] == pytest.approx(ENTRY * (1 + 0.5 * 3 * 0.0001), abs=1e-12)
    assert eq.attrs["negative_funding_share"] == 0.0
    assert eq.attrs["avg_gross_funding_yield"] == pytest.approx(0.5 * 0.0001 * 365, rel=1e-3)
    assert eq.attrs["turnover_per_year"] == pytest.approx(1.0 / (3 / 365))
    assert all(np.isscalar(v) for v in eq.attrs.values())


def test_legs_drift_and_funding_uses_current_notional():
    # Price +10% on the entry day (open to close), then flat at 110: the hedge nets to 0 but the
    # short notional grows to 0.55, so funding on days 2-3 is earned on the drifted notional.
    so, sc, po, pc, f = _frames()
    for fr in (sc, pc):
        fr.iloc[1:] = 110.0
    for fr in (so, po):
        fr.iloc[2:] = 110.0
    eq = _run(so, sc, po, pc, f, _w("2021-01-03", XUSDT=1.0, CASH=0.0))
    assert eq.iloc[0] == pytest.approx(ENTRY * (1 + 0.5 * 0.0001), abs=1e-12)
    assert eq.iloc[-1] == pytest.approx(ENTRY * (1 + 0.0001 * (0.5 + 0.55 + 0.55)), abs=1e-12)


def test_spot_and_perp_moves_hedge_only_when_equal():
    so, sc, po, pc, f = _frames(fund=0.0)
    sc.iloc[-1] = 110.0  # spot +10%, perp flat: the long leg gains 0.5 * 10%
    eq = _run(so, sc, po, pc, f, _w("2021-01-03", XUSDT=1.0, CASH=0.0))
    assert eq.iloc[-1] == pytest.approx(ENTRY * (1 + 0.5 * 0.10), abs=1e-12)


def test_negative_funding_is_paid():
    so, sc, po, pc, f = _frames(fund=-0.0002)
    eq = _run(so, sc, po, pc, f, _w("2021-01-03", XUSDT=1.0, CASH=0.0))
    assert eq.iloc[-1] == pytest.approx(ENTRY * (1 - 0.5 * 3 * 0.0002), abs=1e-12)
    assert eq.attrs["negative_funding_share"] == 1.0
    assert eq.attrs["avg_gross_funding_yield"] < 0


def test_missing_funding_counts_as_zero():
    so, sc, po, pc, f = _frames()
    f.iloc[2] = np.nan
    eq = _run(so, sc, po, pc, f, _w("2021-01-03", XUSDT=1.0, CASH=0.0))
    assert eq.iloc[-1] == pytest.approx(ENTRY * (1 + 0.5 * 2 * 0.0001), abs=1e-12)


def test_cash_moves_are_free():
    so, sc, po, pc, f = _frames(n=12)
    w = pd.concat([_w("2021-01-03", XUSDT=0.0, CASH=1.0), _w("2021-01-10", XUSDT=0.0, CASH=1.0)])
    eq = _run(so, sc, po, pc, f, w)
    assert (eq == 1.0).all()
    assert eq.attrs["turnover_per_year"] == 0.0
    assert np.isnan(eq.attrs["negative_funding_share"])  # no position-days


def test_costs_only_on_traded_leg_notional():
    so, sc, po, pc, f = _frames(n=12, fund=0.0)
    w = pd.concat([_w("2021-01-03", XUSDT=0.5, CASH=0.5),
                   _w("2021-01-10", XUSDT=0.5, CASH=0.5)])  # unchanged weight: no trade
    eq = _run(so, sc, po, pc, f, w)
    half = 1 - 0.25 * SPOT_COST - 0.25 * PERP_COST
    assert eq.iloc[-1] == pytest.approx(half, abs=1e-12)
    # moving from 0.5 to 1.0 trades only the extra 0.25 on each leg
    w2 = pd.concat([_w("2021-01-03", XUSDT=0.5, CASH=0.5),
                    _w("2021-01-10", XUSDT=1.0, CASH=0.0)])
    eq2 = _run(so, sc, po, pc, f, w2)
    full = half * (1 - 0.25 * SPOT_COST - 0.25 * PERP_COST)
    assert eq2.iloc[-1] == pytest.approx(full, rel=1e-6)


def test_weekly_rebalance_trades_drifted_legs_back_to_target():
    # Spot rallies 20% in week 1 with the perp flat; the second Sunday row resets both legs.
    so, sc, po, pc, f = _frames(n=10, fund=0.0)
    sc.loc["2021-01-05":] = 120.0
    so.loc["2021-01-06":] = 120.0
    w = pd.concat([_w("2021-01-03", XUSDT=1.0, CASH=0.0), _w("2021-01-10", XUSDT=1.0, CASH=0.0)])
    eq = _run(so, sc, po, pc, f, w)
    v = ENTRY * (1 + 0.5 * 0.2)
    assert eq.loc["2021-01-10"] == pytest.approx(v, abs=1e-12)
    # Monday 01-11 open: spot leg 0.6*ENTRY and perp notional 0.5*ENTRY both go to v/2.
    # Targets use the pre-cost value; the legs are then set on the post-cost value.
    cost = abs(0.5 * v - 0.6 * ENTRY) * SPOT_COST + abs(0.5 * v - 0.5 * ENTRY) * PERP_COST
    assert eq.loc["2021-01-11"] == pytest.approx(v - cost, abs=1e-12)


def test_delisting_closes_position_without_nan():
    so, sc, po, pc, f = _frames(coins=("AUSDT", "BUSDT"), n=6, fund=0.0)
    f["BUSDT"] = 0.01
    for fr in (so, sc):
        fr.loc["2021-01-05":, "BUSDT"] = np.nan  # spot delisted from Tuesday
    eq = _run(so, sc, po, pc, f, _w("2021-01-03", AUSDT=0.5, BUSDT=0.5, CASH=0.0))
    assert eq.notna().all()
    mon = ENTRY * (1 + 0.25 * 0.01)
    assert eq.loc["2021-01-04"] == pytest.approx(mon, abs=1e-12)
    exit_cost = 0.25 * ENTRY * (SPOT_COST + PERP_COST)
    after = mon - exit_cost
    assert eq.loc["2021-01-05":].to_numpy() == pytest.approx(np.full(4, after), abs=1e-12)


def test_perp_data_end_closes_position():
    so, sc, po, pc, f = _frames(n=6)
    for fr in (po, pc):
        fr.loc["2021-01-06":] = np.nan
    sc.loc["2021-01-06":] = 50.0  # spot crash after the perp ends must not matter
    so.loc["2021-01-07":] = 50.0
    eq = _run(so, sc, po, pc, f, _w("2021-01-03", XUSDT=1.0, CASH=0.0))
    assert eq.notna().all()
    before = ENTRY * (1 + 0.5 * 2 * 0.0001)
    after = before - 0.5 * ENTRY * (SPOT_COST + PERP_COST)
    assert eq.loc["2021-01-05"] == pytest.approx(before, abs=1e-12)
    assert eq.loc["2021-01-06":].to_numpy() == pytest.approx(np.full(3, after), abs=1e-12)


def test_no_perp_price_at_entry_stays_in_cash():
    so, sc, po, pc, f = _frames(n=4)
    po.iloc[:] = np.nan
    pc.iloc[:] = np.nan
    eq = _run(so, sc, po, pc, f, _w("2021-01-03", XUSDT=1.0, CASH=0.0))
    assert (eq == 1.0).all()


def test_missing_close_on_entry_day_exits_at_open():
    so, sc, po, pc, f = _frames(n=4)
    pc.iloc[1:] = np.nan  # perp opens on Monday, then no closes
    eq = _run(so, sc, po, pc, f, _w("2021-01-03", XUSDT=1.0, CASH=0.0))
    assert eq.notna().all()
    out = ENTRY - 0.5 * ENTRY * (SPOT_COST + PERP_COST)
    assert eq.to_numpy() == pytest.approx(np.full(3, out), abs=1e-12)


def test_negative_weights_rejected():
    so, sc, po, pc, f = _frames()
    with pytest.raises(ValueError):
        _run(so, sc, po, pc, f, _w("2021-01-03", XUSDT=-0.5, CASH=1.5))


def test_weight_rows_must_sum_to_one():
    so, sc, po, pc, f = _frames()
    with pytest.raises(ValueError):
        _run(so, sc, po, pc, f, _w("2021-01-03", XUSDT=0.5, CASH=0.4))


# ---------------------------------------------------------------- F1 / F2

def test_f1_weights_weekly_half_half():
    d = pd.date_range("2021-01-01", periods=30, freq="D", name="date")
    w = f1_weights(d)
    assert (w.index.day_of_week == 6).all() and len(w) == 4
    assert (w["BTCUSDT"] == 0.5).all() and (w["ETHUSDT"] == 0.5).all() and (w[CASH] == 0).all()


def _funding(n=60, k=8, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2021-01-01", periods=n, freq="D", name="date")
    cols = [f"C{j}USDT" for j in range(k)]
    f = pd.DataFrame(rng.normal(0.0001, 0.0003, (n, k)), index=d, columns=cols)
    f.loc[: d[25], "C6USDT"] = np.nan  # late perp listing
    elig = pd.DataFrame(rng.random((n, k)) < 0.8, index=d, columns=cols)
    return f, elig


def test_f2_rule_by_hand():
    f, elig = _funding()
    w = f2_weights(f, elig)
    assert (w.index.day_of_week == 6).all()
    assert list(w.columns) == [*f.columns, CASH]
    assert np.allclose(w.sum(axis=1), 1.0)
    for t in w.index:
        win = f.loc[t - pd.Timedelta(days=6): t]
        mean = win.mean().where(win.notna().sum() >= 7)
        ok = elig.loc[t] & (mean > 0)
        picks = mean[ok].sort_values(ascending=False, kind="mergesort").index[:5]
        row = w.loc[t]
        assert set(row[row > 0].drop(CASH, errors="ignore").index) == set(picks)
        assert np.allclose(row[picks], 0.2)
        assert row[CASH] == pytest.approx(1 - 0.2 * len(picks))


def test_f2_needs_seven_days_of_funding():
    f, elig = _funding()
    elig[:] = True
    f["C6USDT"] = np.nan
    f.loc["2021-01-26":, "C6USDT"] = 1.0  # huge positive, 6 days by Sunday 2021-01-31
    w = f2_weights(f, elig)
    assert w.loc["2021-01-31", "C6USDT"] == 0.0
    assert w.loc["2021-02-07", "C6USDT"] == pytest.approx(0.2)


def test_f2_no_lookahead_truncation():
    f, elig = _funding()
    full = f2_weights(f, elig)
    for t in full.index:
        part = f2_weights(f.loc[:t], elig.loc[:t])
        pd.testing.assert_series_equal(part.loc[t], full.loc[t])


def test_f2_cash_when_no_positive_funding():
    f, elig = _funding()
    f = -f.abs() - 1e-6
    elig[:] = True
    w = f2_weights(f, elig)
    assert (w[CASH] == 1.0).all()
    assert (w.drop(columns=CASH) == 0.0).all().all()


def test_f2_ineligible_or_missing_eligibility_excluded():
    f, elig = _funding()
    f[:] = 0.001
    f["C0USDT"] = 0.01
    elig[:] = True
    elig["C0USDT"] = False
    w = f2_weights(f, elig.drop(columns="C1USDT"))
    assert (w["C0USDT"] == 0).all() and (w["C1USDT"] == 0).all()
