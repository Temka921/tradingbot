"""Gated runner for the original strategies (no network, tmp data and report dirs only)."""
import json

import numpy as np
import pandas as pd
import pytest

from swingbot import config
from swingbot.data import store, universe
from swingbot.strategies import CASH, run


# ---------------------------------------------------------------- fixtures

def _curve(index, drift, vol, seed):
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumprod(1 + drift + rng.normal(0, vol, len(index))), index=index)


def _coins(start="2017-08-01", end="2023-12-31", k=8, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range(start, end, freq="D", name="date")
    n = len(d)
    cols = {"BTCUSDT": 100 * np.cumprod(1 + 0.001 + rng.normal(0, 0.02, n))}
    for j in range(k):
        cols[f"C{j}USDT"] = 10 * np.cumprod(1 + rng.normal(0.0004 * (j - 3), 0.04, n))
    closes = pd.DataFrame(cols, index=d)
    opens = closes.shift(1).fillna(closes) * (1 + rng.normal(0, 0.002, closes.shape))
    eligible = pd.DataFrame(rng.random(closes.shape) < 0.85, index=d, columns=closes.columns)
    fng = pd.Series(rng.integers(0, 100, n).astype(float), index=d)
    return opens, closes, eligible, fng


def _bars(index, seed, drift=0.0003):
    rng = np.random.default_rng(seed)
    c = 50 * np.cumprod(1 + drift + rng.normal(0, 0.015, len(index)))
    o = c * (1 + rng.normal(0, 0.003, len(index)))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99,
                         "close": c, "volume": 1e6}, index=index)


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", d)
    return d


# ---------------------------------------------------------------- evaluate

IDX = pd.date_range("2022-01-01", "2023-12-31", freq="D")
VAL = ("2022-01-01", "2023-12-31")


def test_evaluate_c1_beats_btc_passes_and_final_needs_positive_return():
    eq = _curve(IDX, 0.001, 0.005, 1)
    btc = 100 * _curve(IDX, 0.0005, 0.04, 2)
    r = run.evaluate("C1", eq, VAL, {"btc": btc})
    assert r["pass"] and r["reason"] == ""
    assert r["sharpe"] >= r["bench_sharpe"]
    assert abs(r["max_drawdown"]) < abs(r["bench_max_drawdown"])
    assert np.isfinite(r["t_stat"])

    losing = _curve(IDX, -0.0001, 0.004, 4)  # total return < 0, Sharpe above the crash
    crash = 100 * _curve(IDX, -0.003, 0.05, 4)
    assert run.evaluate("C3", losing, VAL, {"btc": crash})["pass"]  # validation: no return bar
    r = run.evaluate("C3", losing, VAL, {"btc": crash}, final=True)
    assert not r["pass"] and "total return" in r["reason"]


def test_evaluate_c1_fails_when_worse_than_btc():
    eq = _curve(IDX, 0.0, 0.03, 5)
    btc = 100 * _curve(IDX, 0.002, 0.01, 6)
    r = run.evaluate("C1", eq, VAL, {"btc": btc})
    assert not r["pass"] and "sharpe" in r["reason"] and "drawdown" in r["reason"]


def test_evaluate_c2_absolute():
    good = _curve(IDX, 0.001, 0.005, 7)
    assert run.evaluate("C2", good, VAL, {})["pass"]
    noisy = _curve(IDX, 0.0002, 0.03, 8)
    noisy = noisy / noisy.iloc[0]
    if noisy.iloc[-1] <= 1:  # make total return positive but Sharpe low
        noisy = noisy * np.linspace(1, 1.1 / noisy.iloc[-1], len(noisy))
    r = run.evaluate("C2", noisy, VAL, {})
    assert r["total_return"] > 0 and r["sharpe"] < 0.5
    assert not r["pass"] and "sharpe" in r["reason"]


def test_evaluate_uses_first_valid_date_and_rebases():
    eq = pd.Series(1.0, index=IDX)
    eq.loc["2022-03-06":] = np.linspace(2.0, 3.0, len(eq.loc["2022-03-06":]))
    valid = pd.DatetimeIndex(["2021-12-26", "2022-03-06", "2022-03-13"])
    r = run.evaluate("C2", eq, VAL, {}, valid_dates=valid)
    assert r["start"] == pd.Timestamp("2022-03-06")
    assert r["total_return"] == pytest.approx(3.0 / 2.0 - 1)
    none = run.evaluate("C2", eq, VAL, {}, valid_dates=pd.DatetimeIndex(["2021-01-03"]))
    assert not none["pass"] and "no valid" in none["reason"]


def test_evaluate_c1_slice_starting_before_equity_counts_flat_cash():
    eq = pd.Series(np.linspace(1.0, 1.5, len(IDX) - 1), index=IDX[1:])  # equity starts day after signal
    btc = 100 * _curve(IDX, 0.0, 0.04, 9)
    r = run.evaluate("C1", eq, VAL, {"btc": btc}, valid_dates=IDX[:1])
    assert r["start"] == IDX[0]
    assert r["total_return"] == pytest.approx(0.5)


def _seeds(totals):
    return pd.DataFrame({s: np.linspace(1, 1 + t, len(IDX)) for s, t in enumerate(totals)}, index=IDX)


def test_evaluate_s1_uses_medians_over_seeds():
    eq = _seeds([0.5, 0.1, 0.1, 0.1, 0.1])  # mean 0.18, median 0.1
    r = run.evaluate("S1", eq, VAL, {"baseline": _seeds([0.15] * 5)})
    assert r["total_return"] == pytest.approx(0.1)
    assert r["bench_total_return"] == pytest.approx(0.15)
    assert not r["pass"] and "baseline" in r["reason"]
    assert run.evaluate("S1", eq, VAL, {"baseline": _seeds([0.05] * 5)})["pass"]
    r = run.evaluate("S1", _seeds([-0.1] * 5), VAL, {"baseline": _seeds([-0.2] * 5)})
    assert not r["pass"] and "total return" in r["reason"]


def test_evaluate_s2_needs_return_and_sharpe_above_equal_weight():
    ewh = _curve(IDX, 0.0008, 0.002, 10)
    eq = _curve(IDX, 0.0012, 0.002, 11)
    r = run.evaluate("S2", eq, VAL, {"ewh": ewh, "home": {"^GSPC": 100 * ewh}})
    assert r["pass"]
    assert "^GSPC_total_return" in r and np.isfinite(r["^GSPC_total_return"])
    t = np.arange(len(IDX))  # 730 days; a 27-day wave is 0 at both ends
    risky = pd.Series(np.linspace(1, 2.0, len(IDX)) * (1 + 0.25 * np.sin(2 * np.pi * t / 27)), index=IDX)
    r = run.evaluate("S2", risky, VAL, {"ewh": ewh})
    assert r["total_return"] > r["bench_total_return"]
    assert not r["pass"] and "sharpe" in r["reason"]


def test_stock_verdict():
    assert run.stock_verdict({"japan": True, "europe": True}) == "confirmed"
    assert run.stock_verdict({"japan": True, "europe": False}) == "mixed"
    assert run.stock_verdict({"japan": False, "europe": True}) == "mixed"
    assert run.stock_verdict({"japan": False, "europe": False}) == "not confirmed"


# ---------------------------------------------------------------- inputs

def test_crypto_inputs_matches_panel_segments_and_pivots_eligible(tmp_data):
    a = pd.date_range("2018-01-01", periods=40, freq="D", name="date")
    b = pd.date_range("2018-03-01", periods=30, freq="D", name="date")
    store.save_bars("XUSDT", "crypto", pd.concat([_bars(a, 1), _bars(b, 2)]))
    store.save_bars("BTCUSDT", "crypto", _bars(a.union(b), 3))
    panel = pd.DataFrame({
        "date": [a[5], a[6], b[3], a[5]],
        "asset": ["XUSDT@2018-01-01", "XUSDT@2018-01-01", "XUSDT", "BTCUSDT"],
        "eligible": [True, False, True, True],
    })
    opens, closes, elig = run.crypto_inputs(panel, allow_holdout=False)
    assert set(closes.columns) == {"XUSDT@2018-01-01", "XUSDT", "BTCUSDT"}
    assert closes["XUSDT"].first_valid_index() == b[0]
    assert closes["XUSDT@2018-01-01"].last_valid_index() == a[-1]
    assert elig.dtypes.eq(bool).all()
    assert bool(elig.loc[a[5], "XUSDT@2018-01-01"]) and not bool(elig.loc[a[6], "XUSDT@2018-01-01"])
    assert not bool(elig.loc[a[7], "XUSDT@2018-01-01"])  # missing -> False
    assert list(opens.columns) == list(closes.columns)


def test_foreign_inputs_glitch_filters_and_maps_home(tmp_data, tmp_path, monkeypatch):
    udir = tmp_path / "universe"
    udir.mkdir()
    monkeypatch.setattr(universe, "UNIVERSE_DIR", udir)
    pd.DataFrame({"symbol": ["A.DE", "B.PA", "C.L"], "market": "europe",
                  "home_index": ["^GDAXI", "^FCHI", "^FTSE"]}).to_csv(udir / "europe.csv", index=False)
    idx = pd.bdate_range("2010-01-04", periods=30, name="date")
    spiky = _bars(idx, 1)
    spiky.iloc[10, spiky.columns.get_loc("close")] *= 3.0  # one-day spike that reverses
    store.save_bars("A.DE", "europe", spiky)
    store.save_bars("B.PA", "europe", _bars(idx, 2))  # C.L has no file
    for i, sym in enumerate(["^GDAXI", "^FCHI", "^FTSE"]):
        store.save_bars(sym, "europe", _bars(idx, 10 + i))
    bars, index_close, home = run.foreign_inputs("europe")
    assert set(bars) == {"A.DE", "B.PA"}
    assert home == {"A.DE": "^GDAXI", "B.PA": "^FCHI"}
    assert set(index_close) >= {"^GDAXI", "^FCHI"}
    assert bars["A.DE"]["close"].iloc[10] == pytest.approx(spiky["close"].iloc[9])


def test_foreign_inputs_missing_universe_never_downloads(tmp_data, tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path / "nothing")
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("network"))
    with pytest.raises(FileNotFoundError):
        run.foreign_inputs("japan")


# ---------------------------------------------------------------- dev: C3 choice

def test_c3_choice_written_before_validation_and_reused(tmp_path, monkeypatch):
    opens, closes, eligible, fng = _coins()
    choice = tmp_path / "reports" / "originals_c3_choice.json"
    calls = []
    real = run.evaluate

    def spy(strategy, equity, period, benchmarks, **kw):
        calls.append((strategy, tuple(period), choice.exists()))
        return real(strategy, equity, period, benchmarks, **kw)

    monkeypatch.setattr(run, "evaluate", spy)
    rows, gate = run.crypto_dev(opens, closes, eligible, fng, choice)
    assert choice.exists()
    stored = json.loads(choice.read_text())
    assert stored["c3_mode"] in ("contrarian", "confirming")
    assert set(stored["discovery_sharpes"]) == {"contrarian", "confirming"}
    s = stored["discovery_sharpes"]
    best = max(s, key=lambda k: -np.inf if s[k] is None or np.isnan(s[k]) else s[k])
    assert stored["c3_mode"] == best
    val = run.PERIODS["crypto"]["validation"]
    c3_val = [c for c in calls if c[0] == "C3" and c[1] == tuple(val)]
    assert c3_val and all(c[2] for c in c3_val)
    # no validation-period call happens while the choice is being made
    first_c3 = next(i for i, c in enumerate(calls) if c[0] == "C3")
    assert all(c[1] != tuple(val) for c in calls[first_c3:] if not c[2])
    assert set(gate) == {"C1", "C2", "C3"}
    assert {r["strategy"] for r in rows} == {"C1", "C2", "C3"}
    assert {r["period"] for r in rows} == {"discovery", "validation"}

    # reuse: a stored choice is never recomputed
    other = "confirming" if stored["c3_mode"] == "contrarian" else "contrarian"
    choice.write_text(json.dumps({"c3_mode": other, "discovery_sharpes": {}}))
    modes = []
    real_c3 = run.cs.c3_weights

    def c3_spy(*a, mode, **k):
        modes.append(mode)
        return real_c3(*a, mode=mode, **k)

    monkeypatch.setattr(run.cs, "c3_weights", c3_spy)
    run.crypto_dev(opens, closes, eligible, fng, choice)
    assert modes and set(modes) == {other}
    assert json.loads(choice.read_text())["c3_mode"] == other


# ---------------------------------------------------------------- final mode

def test_final_runs_only_strategies_that_passed(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    seen = {}
    monkeypatch.setattr(run, "crypto_final", lambda s, r: seen.setdefault("crypto", list(s)) and [])
    monkeypatch.setattr(run, "stock_final", lambda s: seen.setdefault("stock", list(s)) and [])
    (reports / "originals_gate.json").write_text(json.dumps(
        {"C1": True, "C2": False, "C3": False, "S1": False, "S2": True}))
    run.run_final(reports, stamp="t1")
    assert seen == {"crypto": ["C1"], "stock": ["S2"]}

    seen.clear()
    (reports / "originals_gate.json").write_text(json.dumps({k: False for k in run.ALL}))
    run.run_final(reports, stamp="t2")
    assert seen == {}
    assert (reports / "originals_final_t2.txt").exists()


def test_final_without_gate_refuses(tmp_path):
    with pytest.raises(SystemExit):
        run.run_final(tmp_path, stamp="x")


def _crypto_final_env(tmp_data, tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame({"symbol": ["BTCUSDT"]}).to_parquet(tmp_data / "universe_meta.parquet")
    opens, closes, eligible, fng = _coins(end="2024-12-31")
    store.save_bars("BTCUSDT", "crypto", _bars(closes.index, 1))
    calls = []
    record = tmp_data / run.RECORD_NAME

    def fake_build(market, allow_holdout=False, **kw):
        calls.append((market, allow_holdout, record.exists()))
        return pd.DataFrame({"date": [closes.index[0]], "asset": ["BTCUSDT"], "eligible": [True]})

    monkeypatch.setattr(run, "build_panel", fake_build)
    monkeypatch.setattr(run, "crypto_inputs", lambda panel, allow_holdout: (opens, closes, eligible))
    return reports, calls, record


def test_crypto_final_writes_record_before_build_and_refuses_second_run(tmp_data, tmp_path, monkeypatch):
    reports, calls, record = _crypto_final_env(tmp_data, tmp_path, monkeypatch)
    rows = run.crypto_final(["C1", "C2"], reports)
    assert calls == [("crypto", True, True)]
    rec = json.loads(record.read_text())
    assert rec["strategies"] == ["C1", "C2"]
    assert rec["end_date"] == "2024-12-31"
    assert {r["strategy"] for r in rec["results"]} == {"C1", "C2"}
    assert {r["strategy"] for r in rows} == {"C1", "C2"}
    assert all(r["verdict"] in ("confirmed", "not confirmed") for r in rows)
    assert all(pd.Timestamp(r["start"]) >= pd.Timestamp("2024-01-01") for r in rows)
    with pytest.raises(SystemExit):
        run.crypto_final(["C1"], reports)
    assert len(calls) == 1


def test_crypto_final_preflight_missing_files_leaves_no_record(tmp_data, tmp_path, monkeypatch):
    reports, calls, record = _crypto_final_env(tmp_data, tmp_path, monkeypatch)
    with pytest.raises(SystemExit):  # C3 needs the stored choice and the FGI file
        run.crypto_final(["C3"], reports)
    assert not record.exists() and calls == []


def test_crypto_final_error_is_saved_into_record(tmp_data, tmp_path, monkeypatch):
    reports, calls, record = _crypto_final_env(tmp_data, tmp_path, monkeypatch)

    def boom(*a, **k):
        assert record.exists()
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(run, "build_panel", boom)
    rows = run.crypto_final(["C1"], reports)
    rec = json.loads(record.read_text())
    assert "disk on fire" in rec["error"]
    assert rows and rows[0]["verdict"] == "not confirmed" and "error" in rows[0]["reason"]


def test_stock_final_verdicts_per_market(monkeypatch):
    idx = pd.bdate_range("2005-01-03", periods=10)

    def fake_inputs(market):
        return {"X": None}, {}, {}

    monkeypatch.setattr(run, "_foreign_ready", lambda m: True)
    monkeypatch.setattr(run, "foreign_inputs", fake_inputs)
    outcome = {("S1", "japan"): True, ("S1", "europe"): False,
               ("S2", "japan"): True, ("S2", "europe"): True}

    def fake_eval_market(strategy, market, inputs):
        return {"strategy": strategy, "market": market, "pass": outcome[(strategy, market)],
                "reason": "", "start": idx[0], "end": idx[-1]}

    monkeypatch.setattr(run, "stock_final_market", fake_eval_market)
    rows = run.stock_final(["S1", "S2"])
    v = {(r["strategy"], r["market"]): r["verdict"] for r in rows}
    assert v[("S1", "japan")] == v[("S1", "europe")] == "mixed"
    assert v[("S2", "japan")] == "confirmed"


def test_stock_final_skips_when_a_market_is_missing(monkeypatch):
    monkeypatch.setattr(run, "_foreign_ready", lambda m: m == "europe")
    monkeypatch.setattr(run, "foreign_inputs", lambda m: pytest.fail("must not load any market"))
    rows = run.stock_final(["S1"])
    assert rows and rows[0]["verdict"] == "not run" and "japan" in rows[0]["reason"]


# ---------------------------------------------------------------- stock runs (synthetic)

def test_s1_and_s2_runs_on_synthetic_universe():
    idx = pd.bdate_range("2004-01-01", periods=900, name="date")
    bars = {f"S{i}": _bars(idx, 100 + i, drift=0.0002 * (i % 7 - 3)) for i in range(60)}
    index_close = {"^GSPC": _bars(idx, 7)["close"]}
    home = {s: "^GSPC" for s in bars}
    eq, base, valid = run.s1_run(bars, index_close, home, cost=0.001)
    assert list(eq.columns) == list(base.columns) == list(run.SEEDS)
    assert valid[0] >= idx[199]
    seq, ewh, svalid = run.s2_run(bars, index_close, home, cost_per_side=0.0005)
    assert seq.notna().all() and ewh.notna().all()
    assert svalid[0] >= idx[251]
    r = run.evaluate("S2", seq, (None, None), {"ewh": ewh}, valid_dates=svalid)
    assert r["start"] == svalid[0]


def test_run_dev_writes_gate_choice_and_tables_without_holdout(tmp_data, tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    (tmp_data / "panel").mkdir()
    pd.DataFrame({"date": pd.to_datetime(["2023-12-31", "2024-01-01"]), "asset": ["BTCUSDT"] * 2,
                  "eligible": [True, True]}).to_parquet(tmp_data / "panel" / "crypto.parquet")
    opens, closes, eligible, fng = _coins()
    seen = {}

    def fake_crypto_inputs(panel, allow_holdout):
        seen["allow_holdout"], seen["max_date"] = allow_holdout, panel["date"].max()
        return opens, closes, eligible

    idx = pd.bdate_range("2015-01-01", "2023-12-29", name="date")
    bars = {f"S{i}": _bars(idx, 100 + i) for i in range(55)}
    monkeypatch.setattr(run, "crypto_inputs", fake_crypto_inputs)
    monkeypatch.setattr(run.foreign, "load_fng", lambda: fng)
    monkeypatch.setattr(run, "ensure_index", lambda *a, **k: None)
    monkeypatch.setattr(run, "us_stock_inputs",
                        lambda: (bars, {"^GSPC": _bars(idx, 7)["close"]}, {s: "^GSPC" for s in bars}))
    df, gate = run.run_dev(reports, stamp="t")
    assert seen == {"allow_holdout": False, "max_date": pd.Timestamp("2023-12-31")}
    assert set(gate) == set(run.ALL)
    assert json.loads((reports / run.GATE_NAME).read_text()) == gate
    assert (reports / run.CHOICE_NAME).exists()
    assert (reports / "originals_dev_t.csv").exists()
    text = (reports / "originals_dev_t.txt").read_text()
    assert "C3 direction" in text and "Validation gate" in text
    assert set(df["strategy"]) == set(run.ALL)
