"""Funding-carry runner (no network; tmp data and report dirs only)."""
import json

import numpy as np
import pandas as pd
import pytest

from swingbot import config
from swingbot.data import futures, store
from swingbot.strategies import carry_run as cr


# ---------------------------------------------------------------- fixtures

def _curve(index, drift, vol, seed):
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumprod(1 + drift + rng.normal(0, vol, len(index))), index=index)


def _bars(close: np.ndarray, index, seed):
    rng = np.random.default_rng(seed)
    o = close * (1 + rng.normal(0, 0.002, len(index)))
    return pd.DataFrame({"open": o, "high": np.maximum(o, close) * 1.01,
                         "low": np.minimum(o, close) * 0.99, "close": close, "volume": 1e6},
                        index=pd.DatetimeIndex(index, name="date"))


COINS = ("BTCUSDT", "ETHUSDT", "XUSDT", "YUSDT")
SPOT_START, END = "2019-06-01", "2024-06-30"


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", d)
    return d


def _write_market(tmp_data, end=END):
    """Spot, perp and funding files for COINS; XUSDT has an older dead spot segment."""
    days = pd.date_range(SPOT_START, end, freq="D", name="date")
    rng = np.random.default_rng(0)
    for j, sym in enumerate(COINS):
        spot = 100 * np.cumprod(1 + rng.normal(0.0005, 0.03, len(days)))
        perp = spot * (1 + rng.normal(0, 0.0005, len(days)))
        sb, pb = _bars(spot, days, j), _bars(perp, days, 10 + j)
        if sym == "XUSDT":  # dead first listing, 60-day gap, relisted 2019-11-01
            sb = sb[(sb.index < "2019-08-01") | (sb.index >= "2019-11-01")]
        store.save_bars(sym, "crypto", sb)
        store.save_bars(sym, "perp", pb[pb.index >= "2019-09-10"])
        t = pd.date_range("2019-09-10", f"{end} 16:00", freq="8h")
        rate = 0.0001 * (1 + j) + rng.normal(0, 0.00002, len(t))
        if sym == "YUSDT":
            rate = -np.abs(rate)  # never qualifies for F2
        futures.save_funding(sym, pd.DataFrame({"funding_time": t, "rate": rate}))
    # a coin with funding but no perp klines: not in the carry universe
    t = pd.date_range("2019-09-10", f"{end} 16:00", freq="8h")
    futures.save_funding("ZUSDT", pd.DataFrame({"funding_time": t, "rate": 0.001}))
    store.save_bars("ZUSDT", "crypto", _bars(np.full(len(days), 5.0), days, 99))
    pd.DataFrame({"symbol": list(COINS) + ["ZUSDT"], "market": "crypto"}).to_parquet(
        tmp_data / "universe_meta.parquet")


def _panel(end=END):
    """Long panel like build_panel's: XUSDT@2019-06-01 is the dead segment."""
    days = pd.date_range("2019-06-01", end, freq="D", name="date")
    rows = []
    for sym in COINS + ("ZUSDT",):
        for d in days:
            if sym == "XUSDT" and d < pd.Timestamp("2019-11-01"):
                if d < pd.Timestamp("2019-08-01"):
                    rows.append((d, "XUSDT@2019-06-01", True))
                continue
            rows.append((d, sym, sym != "ETHUSDT" or d >= pd.Timestamp("2021-01-01")))
    return pd.DataFrame(rows, columns=["date", "asset", "eligible"])


def _write_dev_panel(tmp_data):
    p = _panel()
    p = p[p["date"] < pd.Timestamp(config.HOLDOUT_START)]
    (tmp_data / "panel").mkdir(exist_ok=True)
    p.to_parquet(tmp_data / "panel" / "crypto.parquet")


# ---------------------------------------------------------------- criteria

IDX = pd.date_range("2021-12-01", "2023-12-31", freq="D")
VAL = ("2022-01-01", "2023-12-31")
SUNDAYS = IDX[IDX.day_of_week == 6]


def _btc():
    return 100 * _curve(IDX, 0.0005, 0.04, 2)


def test_criteria_smooth_carry_passes_with_info_columns():
    eq = pd.Series(np.cumprod(np.full(len(IDX), 1 + 0.10 / 365)), index=IDX)
    eq.iloc[-200:] *= 0.995  # small dip so drawdown is non-zero
    r = cr.evaluate(eq, VAL, SUNDAYS, _btc())
    assert r["pass"] and r["reason"] == ""
    assert r["cagr"] > cr.MIN_CAGR and r["sharpe"] >= cr.MIN_SHARPE
    assert abs(r["max_drawdown"]) < cr.MAX_DRAWDOWN
    assert r["start"] == SUNDAYS[SUNDAYS >= "2022-01-01"][0]
    assert r["end"] == pd.Timestamp("2023-12-31")
    assert r["worst_week"] < 0
    assert np.isfinite(r["btc_total_return"]) and r["btc_max_drawdown"] < 0


def test_worst_week_ignores_the_base_point():
    up = pd.Series(np.cumprod(np.full(len(IDX), 1.001)), index=IDX)
    s = up.loc["2022-01-02":]  # starts on a Sunday
    assert cr._worst_week(s) == pytest.approx(1.001 ** 7 - 1)


def test_criteria_each_failure_is_named():
    low = pd.Series(np.cumprod(np.full(len(IDX), 1 + 0.02 / 365)), index=IDX)
    r = cr.evaluate(low, VAL, SUNDAYS, _btc())
    assert not r["pass"] and "CAGR" in r["reason"] and "Sharpe" not in r["reason"]

    noisy = _curve(IDX, 0.0, 0.02, 3)  # no drift: poor Sharpe, deep drawdowns
    r = cr.evaluate(noisy, VAL, SUNDAYS, _btc())
    assert not r["pass"] and "Sharpe" in r["reason"] and "drawdown" in r["reason"]

    crash = pd.Series(np.cumprod(np.full(len(IDX), 1 + 0.30 / 365)), index=IDX)
    crash.loc["2023-03-01":"2023-03-05"] *= 0.80  # one 20% hole, otherwise smooth
    r = cr.evaluate(crash, VAL, SUNDAYS, _btc())
    assert not r["pass"] and r["reason"] == "max drawdown >= 15%"


def test_criteria_rebases_at_first_rebalance_in_period():
    eq = pd.Series(np.linspace(1.0, 3.0, len(IDX)), index=IDX)
    valid = pd.DatetimeIndex(["2022-03-06", "2022-03-13"])
    r = cr.evaluate(eq, VAL, valid, _btc())
    assert r["start"] == pd.Timestamp("2022-03-06")
    expect = eq.loc["2023-12-31"] / eq.loc["2022-03-06"] - 1
    assert r["total_return"] == pytest.approx(expect)


def test_criteria_no_rebalance_in_period_fails():
    eq = pd.Series(1.0, index=IDX)
    r = cr.evaluate(eq, VAL, pd.DatetimeIndex([]), _btc())
    assert not r["pass"] and "no valid rebalance" in r["reason"]


def test_check_criteria_nan_fails():
    fails = cr.check_criteria({"cagr": float("nan"), "sharpe": float("nan"),
                               "max_drawdown": float("nan")})
    assert len(fails) == 3


# ---------------------------------------------------------------- inputs

def test_carry_coins_ignores_dead_segments_and_needs_perp(tmp_data):
    _write_market(tmp_data)
    p = _panel()
    p.loc[p["asset"] == "YUSDT", "eligible"] = False
    assert cr.available_perps() == {"BTCUSDT", "ETHUSDT", "XUSDT", "YUSDT"}
    assert cr.carry_coins(p, ["F1"]) == ["BTCUSDT", "ETHUSDT"]
    assert cr.carry_coins(p, ["F2"]) == ["BTCUSDT", "ETHUSDT", "XUSDT"]  # Z has no perp, Y never eligible
    assert cr.carry_coins(p, ["F1", "F2"]) == ["BTCUSDT", "ETHUSDT", "XUSDT"]


def test_load_inputs_dev_never_sees_holdout(tmp_data, monkeypatch):
    _write_market(tmp_data)
    p = _panel()
    p = p[p["date"] < pd.Timestamp(config.HOLDOUT_START)]
    flags = []
    real_bars, real_fund = store.load_bars, futures.load_funding
    monkeypatch.setattr(store, "load_bars",
                        lambda s, m, allow_holdout=False: flags.append(allow_holdout) or real_bars(s, m, allow_holdout))
    monkeypatch.setattr(futures, "load_funding",
                        lambda s, allow_holdout=False: flags.append(allow_holdout) or real_fund(s, allow_holdout))
    f = cr.load_inputs(p, ["F1", "F2"], allow_holdout=False)
    assert flags and not any(flags)
    assert not (tmp_data / "holdout_log.txt").exists()
    for k in ("spot_open", "spot_close", "perp_open", "perp_close", "funding", "eligible"):
        assert f[k].index.max() < pd.Timestamp(config.HOLDOUT_START), k
        assert "XUSDT@2019-06-01" not in f[k].columns
    assert list(f["spot_close"].columns) == ["BTCUSDT", "ETHUSDT", "XUSDT", "YUSDT"]
    assert f["spot_close"]["XUSDT"].first_valid_index() == pd.Timestamp("2019-11-01")
    assert not f["eligible"].loc["2020-06-01", "ETHUSDT"]
    # daily funding comes from futures.daily_funding (its day convention is owned there)
    day = futures.daily_funding(futures.load_funding("BTCUSDT"))
    assert f["funding"].loc["2020-02-02", "BTCUSDT"] == pytest.approx(day.loc["2020-02-02"])


# ---------------------------------------------------------------- dev mode

def test_run_dev_writes_gate_and_tables_without_holdout(tmp_data, tmp_path, monkeypatch):
    _write_market(tmp_data)
    _write_dev_panel(tmp_data)
    monkeypatch.setattr(cr, "build_panel", lambda *a, **k: pytest.fail("dev must not build a panel"))
    reports = tmp_path / "reports"
    df, gate = cr.run_dev(reports, stamp="T")
    assert set(gate) == {"F1", "F2"} and all(isinstance(v, bool) for v in gate.values())
    assert json.loads((reports / cr.GATE_NAME).read_text()) == gate
    assert (reports / "funding_dev_T.csv").exists() and (reports / "funding_dev_T.txt").exists()
    assert set(zip(df["strategy"], df["period"])) == {(s, p) for s in ("F1", "F2")
                                                     for p in ("dev", "validation")}
    for s in ("F1", "F2"):
        v = df[(df["strategy"] == s) & (df["period"] == "validation")].iloc[0]
        assert gate[s] == bool(v["pass"])
    assert pd.to_datetime(df["end"]).max() < pd.Timestamp(config.HOLDOUT_START)
    for c in ("negative_funding_share", "avg_gross_funding_yield", "turnover_per_year",
              "worst_week", "btc_total_return", "btc_max_drawdown"):
        assert c in df.columns
    f2 = df[df["strategy"] == "F2"].iloc[0]
    assert f2["negative_funding_share"] == pytest.approx(0.0)  # Y (negative) is never held
    assert f2["avg_gross_funding_yield"] > 0
    assert not (tmp_data / "holdout_log.txt").exists()
    assert not (tmp_data / cr.RECORD_NAME).exists()


# ---------------------------------------------------------------- final mode

def _final_env(tmp_data, tmp_path, monkeypatch, gate):
    _write_market(tmp_data)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / cr.GATE_NAME).write_text(json.dumps(gate))
    record = tmp_data / cr.RECORD_NAME
    calls = []

    def fake_build(market, allow_holdout=False, **kw):
        calls.append(("panel", allow_holdout, record.exists()))
        return _panel()

    real_bars, real_fund = store.load_bars, futures.load_funding

    def bars_spy(s, m, allow_holdout=False):
        calls.append((f"{m}/{s}", allow_holdout, record.exists()))
        return real_bars(s, m, allow_holdout)

    def fund_spy(s, allow_holdout=False):
        calls.append((f"funding/{s}", allow_holdout, record.exists()))
        return real_fund(s, allow_holdout)

    monkeypatch.setattr(cr, "build_panel", fake_build)
    monkeypatch.setattr(store, "load_bars", bars_spy)
    monkeypatch.setattr(futures, "load_funding", fund_spy)
    return reports, record, calls


def test_final_writes_record_before_any_holdout_load(tmp_data, tmp_path, monkeypatch):
    reports, record, calls = _final_env(tmp_data, tmp_path, monkeypatch, {"F1": True, "F2": True})
    df = cr.run_final(reports, stamp="T")
    assert calls[0][0] == "panel"
    assert all(c[1] and c[2] for c in calls)  # every load: holdout allowed, record already there
    assert any(c[0] == "perp/BTCUSDT" for c in calls) and any(c[0] == "funding/XUSDT" for c in calls)
    rec = json.loads(record.read_text())
    assert rec["strategies"] == ["F1", "F2"]
    assert rec["end_date"] == END
    assert {r["strategy"] for r in rec["results"]} == {"F1", "F2"}
    assert set(df["strategy"]) == {"F1", "F2"}
    assert (df["period"] == "final").all()
    assert set(df["verdict"]) <= {"confirmed", "not confirmed"}
    assert (pd.to_datetime(df["start"]) >= pd.Timestamp(config.HOLDOUT_START)).all()
    assert (reports / "funding_final_T.csv").exists() and (reports / "funding_final_T.txt").exists()


def test_final_runs_only_strategies_that_passed(tmp_data, tmp_path, monkeypatch):
    reports, record, calls = _final_env(tmp_data, tmp_path, monkeypatch, {"F1": False, "F2": True})
    seen = []
    real = cr.run_strategy
    monkeypatch.setattr(cr, "run_strategy", lambda s, f: seen.append(s) or real(s, f))
    df = cr.run_final(reports, stamp="T")
    assert seen == ["F2"]
    assert list(df["strategy"]) == ["F2"]
    assert json.loads(record.read_text())["strategies"] == ["F2"]


def test_final_second_run_refused_without_loading(tmp_data, tmp_path, monkeypatch):
    reports, record, calls = _final_env(tmp_data, tmp_path, monkeypatch, {"F1": True, "F2": False})
    first = cr.run_final(reports, stamp="A")
    rec = json.loads(record.read_text())
    n = len(calls)
    again = cr.run_final(reports, stamp="B")
    assert len(calls) == n  # nothing loaded
    assert json.loads(record.read_text()) == rec
    assert "refused" in again.iloc[0]["reason"]
    assert again.iloc[0]["verdict"] == first.iloc[0]["verdict"]


def test_final_nothing_passed_spends_nothing(tmp_data, tmp_path, monkeypatch, capsys):
    reports, record, calls = _final_env(tmp_data, tmp_path, monkeypatch, {"F1": False, "F2": False})
    df = cr.run_final(reports, stamp="T")
    assert df.empty and calls == [] and not record.exists()
    assert not (tmp_data / "holdout_log.txt").exists()
    assert "no strategy passed validation" in capsys.readouterr().out.lower()


def test_final_without_gate_refuses(tmp_data, tmp_path):
    with pytest.raises(SystemExit):
        cr.run_final(tmp_path / "reports")
    assert not (tmp_data / cr.RECORD_NAME).exists()


def test_final_preflight_missing_files_leaves_no_record(tmp_data, tmp_path, monkeypatch):
    reports, record, calls = _final_env(tmp_data, tmp_path, monkeypatch, {"F1": True})
    (tmp_data / "raw" / "perp" / "ETHUSDT.parquet").unlink()
    with pytest.raises(SystemExit):
        cr.run_final(reports)
    assert not record.exists() and calls == []


def test_final_error_is_saved_into_record(tmp_data, tmp_path, monkeypatch):
    reports, record, calls = _final_env(tmp_data, tmp_path, monkeypatch, {"F1": True})

    def boom(*a, **k):
        assert record.exists()
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(cr, "build_panel", boom)
    df = cr.run_final(reports, stamp="T")
    assert "disk on fire" in json.loads(record.read_text())["error"]
    assert df.iloc[0]["verdict"] == "not confirmed" and "error" in df.iloc[0]["reason"]


def test_final_interrupt_is_recorded_and_reraised(tmp_data, tmp_path, monkeypatch):
    reports, record, calls = _final_env(tmp_data, tmp_path, monkeypatch, {"F1": True})

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cr, "build_panel", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cr.run_final(reports)
    assert "interrupted" in json.loads(record.read_text())["error"]
