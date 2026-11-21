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


# ---------- final review F1/F3/F4 ----------

def test_combo_report_mean_ret_columns_and_wording(tmp_path):
    from swingbot.report.build import SHORT_NOTE
    from swingbot.report.combos import COMBO_CAVEAT
    html = build_combo_report(_panels(), tmp_path / "c.html", horizon=10, as_of="x").read_text(encoding="utf-8")
    assert "Mean return A&amp;B, absolute (bps)" in html and "Mean excess A&amp;B (bps)" in html
    assert html.index("Mean excess A&amp;B (bps)") < html.index("Mean return A&amp;B, absolute (bps)")
    assert SHORT_NOTE in html
    cav = html[html.index("Caveats."):html.index("How to read this")]
    assert SHORT_NOTE in cav
    # F3: both slice rules, printed threshold, accurate q wording
    # the regime paragraph renders only with synergy rows: render the template with one fake row
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    from swingbot.report import combos
    env = Environment(loader=FileSystemLoader(combos._TEMPLATES), autoescape=select_autoescape(["j2", "html"]))
    view = {"view": "crypto", "suspect": False, "primary": True, "pair_counts": {}, "triple_counts": {},
            "n_pairs": 0, "n_pairs_shown": 0, "n_triples": 0, "pair_rows": [], "triple_rows": [],
            "regime_rows": [{"insufficient": False, "cells": ["x"]}], "heat": "", "groups": []}
    page = env.get_template("combos.html.j2").render(
        views=[view], verdicts=[], redundant_jaccard=config.REDUNDANT_JACCARD, slice_rows=30, slice_dates=50)
    assert "fewer than 30 rows, or fewer than max(30, 5×horizon) dates (here 30 rows / 50 dates)" in page
    assert "at or above the threshold" not in html
    assert html.count(f"Jaccard at or above {config.REDUNDANT_JACCARD}") >= 2
    assert "0.5" in html
    assert ("The pair table has q (edge) and q (lift); the triple table has q (lift) only; "
            "the regime table has q (edge) only.") in COMBO_CAVEAT
    assert "separately for q (edge) and q (lift)" not in COMBO_CAVEAT


def test_select_pairs_direct():
    from swingbot.report.combos import select_pairs
    rows = [{"a": "s1", "verdict": "synergy", "insufficient": False, "edge_ab": 0.001},
            {"a": "s2", "verdict": "synergy", "insufficient": False, "edge_ab": 0.004},
            {"a": "c1", "verdict": "conflict", "insufficient": False, "edge_ab": 0.050}]
    rows += [{"a": f"ins{i}", "verdict": "insufficient", "insufficient": True, "edge_ab": 1.0 + i}
             for i in range(5)]
    rows += [{"a": f"o{i}", "verdict": "not significant" if i % 2 else "redundant",
              "insufficient": False, "edge_ab": i / 1000} for i in range(40)]
    rows += [{"a": "onan", "verdict": "not significant", "insufficient": False, "edge_ab": np.nan}]
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0)
    out = select_pairs(df)
    got = out["a"].tolist()
    assert got[:3] == ["s2", "s1", "c1"]                         # every synergy, then every conflict
    others = out[~out["verdict"].isin(["synergy", "conflict"])]
    assert len(others) == 30
    assert not others["insufficient"].any() and not any(a.startswith("ins") for a in others["a"])
    assert others["edge_ab"].tolist() == sorted(others["edge_ab"], reverse=True)
    assert others["a"].tolist() == [f"o{i}" for i in range(39, 9, -1)]
    assert len(select_pairs(df, top=5)) == 3 + 5
