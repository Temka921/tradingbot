import re
import runpy
import sys

import numpy as np
import pandas as pd

from swingbot import config
from swingbot.report.build import CDN_URL, SUSPECT_LABEL, SURVIVORSHIP, _fmt
from swingbot.report.combos import build_combo_report, jaccard_for_report
from swingbot.signals import SIGNAL_INFO
from tests.test_report import make_market_panel


def _panels():
    stock = make_market_panel("stock", assets=("SPY", "QQQ", "AAA", "BBB"), seed=1)
    stock["is_etf"] = stock["asset"].isin(["SPY", "QQQ"])
    return {"stock": stock, "crypto": make_market_panel("crypto", seed=2)}


def test_combo_report_renders(tmp_path):
    out = build_combo_report(_panels(), tmp_path / "combo_report_x.html", horizon=10, as_of="x")
    html = out.read_text(encoding="utf-8")
    assert SUSPECT_LABEL in html and CDN_URL in html
    for v in ("etf", "crypto", "stock"):
        for n in ("pairs", "triples", "pair_regimes", "jaccard"):
            assert (tmp_path / f"combo_{n}_{v}_x.csv").exists()
    assert "lift" in html.lower() and "hypothesis" in html.lower()
    assert "validation" in html.lower() and "suspect" in html.lower()
    assert "S&amp;P 500/400 constituents" in html          # shared survivorship text
    assert "{stock_mean}" not in html and "bps per 10 days" in html


def test_fmt_combo_columns():
    assert _fmt("edge_ab", 0.001) == "10.0"
    assert _fmt("lift", -0.00123) == "-12.3"
    assert _fmt("q_lift", 0.12345) == "0.123"
    assert _fmt("jaccard", 0.456) == "0.46"
    assert _fmt("lift", np.nan) == ""
    assert _fmt("n_ab", 12) == "12"


def test_jaccard_blank_diagonal_for_never_firing():
    p = make_market_panel("stock", seed=1)
    dead = list(SIGNAL_INFO)[1]
    p[dead] = False
    m = jaccard_for_report(p)
    assert np.isnan(m.loc[dead, dead])
    live = list(SIGNAL_INFO)[2]
    assert m.loc[live, live] == 1.0


def test_pair_table_shows_all_synergy_conflict_and_top30(tmp_path):
    build_combo_report(_panels(), tmp_path / "c.html", as_of="x")
    html = (tmp_path / "c.html").read_text(encoding="utf-8")
    assert "Jaccard" in html and "Redundancy" in html
    assert re.search(r"synergy: \d+", html) and re.search(r"conflict: \d+", html)


def test_analyze_script_writes_both_reports(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    (data / "panel").mkdir(parents=True)
    make_market_panel("stock").to_parquet(data / "panel" / "stock.parquet", index=False)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(sys, "argv", ["analyze.py", "--horizon", "10"])
    runpy.run_path(str(config._REPO_ROOT / "scripts" / "analyze.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "SKIP crypto" in out
    assert list((tmp_path / "reports").glob("solo_report_*.html"))
    combo = list((tmp_path / "reports").glob("combo_report_*.html"))
    assert combo and combo[0].name in out
