import json
import re

import numpy as np
import pandas as pd
import pytest

from swingbot import config
from swingbot import holdout
from swingbot.analysis.validation import GATE_COLS
from swingbot.data import store
from tests.test_validation import INFO, synth


def _gate_row(view="crypto", kind="solo", rule="A", direction="long", eligible=True) -> dict:
    r = {c: 0.0 for c in GATE_COLS}
    r.update(view=view, kind=kind, rule=rule, direction=direction, val_n=500, val_n_dates=300,
             finalist=True, holdout_eligible=eligible, reason="")
    return r


def _csv(tmp_path, rows):
    path = tmp_path / "finalists_x.csv"
    pd.DataFrame(rows, columns=GATE_COLS).to_csv(path, index=False)
    return path


def _bars(start="2014-06-01", periods=1500, seed=2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, periods)))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "volume": 1.0}, index=idx)


def _holdout_panel(**kw) -> pd.DataFrame:
    """synth() with its second half relabelled "holdout" (discovery rows stay, as in a real holdout panel)."""
    p = synth(**kw)
    p["split"] = pd.Categorical(np.where(p["split"] == "validation", "holdout", "discovery"))
    return p


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", d)
    return d


def _no_build(market, allow_holdout=False):
    raise AssertionError("build_panel must not be called")


def test_no_eligible_rows_loads_nothing(tmp_path, data_dir, monkeypatch, capsys):
    monkeypatch.setattr(holdout, "build_panel", _no_build)
    csv = _csv(tmp_path, [_gate_row(view="stock", eligible=False),
                          _gate_row(view="crypto", eligible=False)])
    assert holdout.run_holdout(csv, out_dir=tmp_path / "rep") is None
    assert "holdout locked: no eligible finalists" in capsys.readouterr().out
    assert not (data_dir / "holdout_log.txt").exists()
    assert not (data_dir / "holdout_used.json").exists()
    assert not (tmp_path / "rep").exists()


def test_string_flags_and_non_primary_view_are_not_eligible(tmp_path, data_dir, monkeypatch, capsys):
    monkeypatch.setattr(holdout, "build_panel", _no_build)
    path = tmp_path / "f.csv"
    rows = [_gate_row(view="stock", eligible=True), _gate_row(view="crypto", eligible=False)]
    df = pd.DataFrame(rows, columns=GATE_COLS)
    df["holdout_eligible"] = ["True", "False"]
    df["holdout_eligible"] = df["holdout_eligible"].astype(object)
    df.to_csv(path, index=False)
    assert holdout.run_holdout(path) is None
    assert "holdout locked" in capsys.readouterr().out
    assert not data_dir.exists()


def test_second_run_refused(tmp_path, data_dir, monkeypatch):
    monkeypatch.setattr(holdout, "build_panel", _no_build)
    data_dir.mkdir(parents=True)
    (data_dir / "holdout_used.json").write_text(
        json.dumps({"timestamp": "2026-11-20T10:00:00+00:00", "rules": []}), encoding="utf-8")
    log = data_dir / "holdout_log.txt"
    log.write_text("2026-11-20T10:00:00+00:00 holdout access crypto/BTCUSDT\n", encoding="utf-8")
    before = log.read_bytes()
    csv = _csv(tmp_path, [_gate_row()])
    with pytest.raises(RuntimeError, match=r"holdout already used on 2026-11-20T10:00:00\+00:00"):
        holdout.run_holdout(csv, out_dir=tmp_path / "rep")
    assert log.read_bytes() == before
    assert not (tmp_path / "rep").exists()


def test_marker_written_before_loading(tmp_path, data_dir, monkeypatch):
    def boom(market, allow_holdout=False):
        assert (data_dir / "holdout_used.json").exists()
        raise OSError("disk on fire")

    monkeypatch.setattr(holdout, "build_panel", boom)
    csv = _csv(tmp_path, [_gate_row()])
    with pytest.raises(OSError):
        holdout.run_holdout(csv, out_dir=tmp_path / "rep", signal_info=INFO)
    marker = json.loads((data_dir / "holdout_used.json").read_text(encoding="utf-8"))
    assert marker["timestamp"]
    assert marker["finalists_csv"] == str(csv)
    assert marker["rules"] == [{"view": "crypto", "kind": "solo", "rule": "A", "direction": "long"}]
    with pytest.raises(RuntimeError, match="holdout already used on"):
        holdout.run_holdout(csv, out_dir=tmp_path / "rep")


def test_happy_path_writes_report(tmp_path, data_dir, monkeypatch):
    store.save_bars("BTCUSDT", "crypto", _bars())
    calls = []
    panel = _holdout_panel(a_disc=0.01, a_val=0.01, ab_val=0.0)

    def fake_build(market, allow_holdout=False):
        calls.append((market, allow_holdout))
        return panel

    monkeypatch.setattr(holdout, "build_panel", fake_build)
    csv = _csv(tmp_path, [_gate_row(rule="A"), _gate_row(rule="S", direction="short"),
                          _gate_row(kind="pair", rule="A+B"),
                          _gate_row(view="stock", rule="A", eligible=False)])
    out = holdout.run_holdout(csv, out_dir=tmp_path / "rep", signal_info=INFO)
    assert out is not None and out.exists()
    assert out.parent == tmp_path / "rep" and re.fullmatch(r"holdout_report_.+\.html", out.name)
    assert calls == [("crypto", True)]
    html = out.read_text(encoding="utf-8")
    verdicts = re.findall(r'<td class="verdict">([^<]*)</td>', html)
    assert len(verdicts) == 3  # the stock row is never sent to the holdout
    assert verdicts[0] == "confirmed"          # A: strong long edge on the holdout rows
    assert verdicts[1] == "not confirmed"      # S: pure noise
    assert "A+B" in html
    # benchmark closes come through the logged holdout load
    log = (data_dir / "holdout_log.txt").read_text(encoding="utf-8")
    assert "holdout access crypto/BTCUSDT" in log
    assert (data_dir / "holdout_used.json").exists()
    with pytest.raises(RuntimeError):
        holdout.run_holdout(csv, out_dir=tmp_path / "rep", signal_info=INFO)


def test_view_missing_from_holdout_panel_is_not_confirmed(tmp_path, data_dir, monkeypatch):
    panel = _holdout_panel()
    monkeypatch.setattr(holdout, "build_panel", lambda market, allow_holdout=False: panel.assign(market=market))
    csv = _csv(tmp_path, [_gate_row(view="etf", rule="A")])  # stock market panel without ETFs
    out = holdout.run_holdout(csv, out_dir=tmp_path / "rep", signal_info=INFO)
    html = out.read_text(encoding="utf-8")
    assert re.findall(r'<td class="verdict">([^<]*)</td>', html) == ["not confirmed"]
    assert "not in the holdout panel" in html


def test_unknown_signal_refused_before_marker(tmp_path, data_dir, monkeypatch):
    monkeypatch.setattr(holdout, "build_panel", _no_build)
    csv = _csv(tmp_path, [_gate_row(kind="pair", rule="A+NOPE")])
    with pytest.raises(ValueError, match="NOPE"):
        holdout.run_holdout(csv, signal_info=INFO)
    assert not data_dir.exists()


def test_cli_prints_locked(tmp_path, data_dir, monkeypatch, capsys):
    import runpy
    import sys
    monkeypatch.setattr(holdout, "build_panel", _no_build)
    csv = _csv(tmp_path, [_gate_row(eligible=False)])
    monkeypatch.setattr(sys, "argv", ["holdout.py", "--finalists", str(csv)])
    runpy.run_path(str(config._REPO_ROOT / "scripts" / "holdout.py"), run_name="__main__")
    assert "holdout locked: no eligible finalists" in capsys.readouterr().out
    assert not data_dir.exists()
