import runpy
import sys

import numpy as np
import pandas as pd

from swingbot import config
from swingbot.analysis.validation import GATE_COLS
from swingbot.data import store
from swingbot.report.build import CDN_URL, SHORT_NOTE, SUSPECT_LABEL
from swingbot.report.finalists import build_finalists_report
from tests.test_report import make_market_panel
from tests.test_validation import INFO, synth


def _bars(start="2014-06-01", periods=1500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, periods)))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "volume": 1.0}, index=idx)


def _save_benchmarks(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    store.save_bars("SPY", "stock", _bars(seed=1))
    store.save_bars("BTCUSDT", "crypto", _bars(seed=2))


def _etf(seed=3, **kw) -> pd.DataFrame:
    p = synth(seed=seed, **kw)
    p["market"] = "stock"
    p["is_etf"] = True
    return p


def _crypto(**kw) -> pd.DataFrame:
    return synth(**kw)


def _section(html: str, view: str) -> str:
    start = html.index(f'id="view-{view}"')
    nxt = html.find('id="view-', start + 10)
    end = nxt if nxt != -1 else html.find('id="holdout"', start)
    return html[start:end]


def test_persistent_crypto_signal(tmp_path, monkeypatch):
    _save_benchmarks(tmp_path, monkeypatch)
    panels = {"crypto": _crypto(a_disc=0.01, a_val=0.01), "stock": _etf()}
    out, gate = build_finalists_report(panels, tmp_path / "rep" / "finalists_report_x.html", horizon=10,
                                       as_of="x", signal_info=INFO)
    html = out.read_text(encoding="utf-8")
    assert list(gate.columns) == GATE_COLS
    assert CDN_URL in html and SHORT_NOTE in html
    assert "equal-slot sizing" in html and "slippage" in html
    crypto = _section(html, "crypto")
    assert "Finalists: 1 " in crypto
    assert "Robustness: 15 runs" in crypto
    assert "in-sample" in crypto and "2.5" in crypto and "97.5" in crypto
    assert "Finalists: 0 " in _section(html, "etf")
    assert "Eligible for the one-time holdout run" in html and "scripts/holdout.py" in html
    assert "Holdout locked" not in html
    assert (tmp_path / "rep" / "finalists_x.csv").exists()
    rob = pd.read_csv(tmp_path / "rep" / "finalists_robustness_x.csv")
    assert len(rob) == 15
    assert sorted(rob["max_positions"].unique()) == [8, 10, 12]
    assert sorted(rob["seed"].unique()) == [0, 1, 2, 3, 4]
    csv = pd.read_csv(tmp_path / "rep" / "finalists_x.csv")
    assert len(csv) == len(gate) and bool(csv["finalist"].any())
    assert (gate["holdout_eligible"]).sum() == 1


def test_no_persistent_signal_locks_holdout(tmp_path, monkeypatch):
    _save_benchmarks(tmp_path, monkeypatch)
    panels = {"crypto": _crypto(a_disc=0.01, a_val=0.0), "stock": _etf()}
    out, gate = build_finalists_report(panels, tmp_path / "finalists_report_y.html", as_of="y",
                                       signal_info=INFO)
    html = out.read_text(encoding="utf-8")
    assert "Holdout locked: no finalist from the primary views (ETF, crypto) passed validation" in html
    assert "Eligible for the one-time holdout run" not in html
    assert not gate["holdout_eligible"].any()
    assert (tmp_path / "finalists_y.csv").exists()
    assert len(pd.read_csv(tmp_path / "finalists_robustness_y.csv")) == 0


def test_stock_finalist_suspect_not_eligible(tmp_path, monkeypatch):
    _save_benchmarks(tmp_path, monkeypatch)
    stock = synth(a_disc=0.01, a_val=0.01)
    stock["market"] = "stock"
    stock["is_etf"] = False
    out, gate = build_finalists_report({"stock": stock}, tmp_path / "r.html", as_of="z", signal_info=INFO)
    html = out.read_text(encoding="utf-8")
    sec = _section(html, "stock")
    assert SUSPECT_LABEL in sec
    assert "not eligible for the holdout (survivorship)" in sec
    assert "Finalists: 1 " in sec
    assert "Holdout locked" in html
    assert gate["finalist"].sum() == 1 and not gate["holdout_eligible"].any()


def test_missing_benchmark_renders(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "empty")
    out, _ = build_finalists_report({"crypto": _crypto(a_disc=0.01, a_val=0.01)}, tmp_path / "m.html",
                                    as_of="m", signal_info=INFO)
    assert "benchmark unavailable" in out.read_text(encoding="utf-8")


def test_finalists_script(tmp_path, monkeypatch, capsys):
    _save_benchmarks(tmp_path, monkeypatch)
    data = tmp_path / "data"
    (data / "panel").mkdir(parents=True, exist_ok=True)
    make_market_panel("crypto", seed=2).to_parquet(data / "panel" / "crypto.parquet", index=False)
    monkeypatch.setattr(sys, "argv", ["finalists.py", "--horizon", "10"])
    runpy.run_path(str(config._REPO_ROOT / "scripts" / "finalists.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "finalists_report_" in out and ".csv" in out
