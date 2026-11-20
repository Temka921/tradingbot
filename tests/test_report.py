import numpy as np
import pandas as pd

from swingbot.analysis.solo import solo_table
from swingbot.report.build import CAVEAT, CDN_URL, build_report, heatmap_matrix
from swingbot.signals import SIGNAL_INFO


def make_market_panel(market: str, n_dates: int = 300, assets=("A1", "A2", "A3"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_dates)
    p = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"]).to_frame(index=False)
    n = len(p)
    for name in SIGNAL_INFO:
        p[name] = rng.random(n) < 0.15
    rare = list(SIGNAL_INFO)[0]
    p[rare] = False
    p.loc[[3, 40, 90], rare] = True  # far below any minimum
    for side in ("long", "short"):
        for h in (5, 10, 20):
            p[f"excess_{side}_{h}"] = rng.normal(0, 0.02, n).astype("float32")
            p[f"ret_{side}_{h}"] = rng.normal(0, 0.02, n).astype("float32")
    p["market"] = market
    p["trend_regime"] = np.where(np.arange(n) % 2 == 0, "up", "down")
    p["vol_regime"] = np.where(np.arange(n) % 3 == 0, "high", "low")
    p["sector"] = "Tech"
    p["cap_bucket"] = "large"
    p["split"] = "discovery"
    for c in ("market", "trend_regime", "vol_regime", "sector", "cap_bucket", "split"):
        p[c] = p[c].astype("category")
    return p


def test_report_renders(tmp_path):
    panels = {"stock": make_market_panel("stock", seed=1), "crypto": make_market_panel("crypto", seed=2)}
    out = build_report(panels, tmp_path / "solo_report_2026-01-01.html", horizon=10)
    html = out.read_text(encoding="utf-8")
    assert out.exists()
    assert CAVEAT in html
    assert html.index(CAVEAT) < html.index("p-values test the edge versus the same-day average")
    for name in SIGNAL_INFO:
        assert name in html
    assert "plotly" in html and CDN_URL in html
    assert "Edge vs same-day avg (bps)" in html and "q (BH, this table)" in html
    assert "How to read this" in html
    assert list(tmp_path.glob("solo_*_*.csv"))


def test_report_insufficient_blank():
    panel = make_market_panel("stock")
    rare = list(SIGNAL_INFO)[0]
    tbl = solo_table(panel, horizon=10, group_by=["market", "trend_regime"])
    assert tbl[tbl.signal == rare]["insufficient"].all()
    m = heatmap_matrix(tbl, "trend_regime")
    assert m.loc[rare].isna().all()                      # blank, no number
    other = [s for s in SIGNAL_INFO if s != rare][0]
    assert m.loc[other].notna().all()
    assert len(m) == len(SIGNAL_INFO)


# ---------- review fixes ----------
import re
import runpy
import sys

import pytest

from swingbot import config
from swingbot.report.build import _fmt, _quality_count, sort_leaderboard


def _render(tmp_path, **kw):
    panels = {"stock": make_market_panel("stock", seed=1), "crypto": make_market_panel("crypto", seed=2)}
    return panels, build_report(panels, tmp_path / "r.html", horizon=10, as_of="2026-01-01", **kw)


def test_insufficient_blank_in_rendered_html(tmp_path):
    rare = list(SIGNAL_INFO)[0]
    panels = {"stock": make_market_panel("stock", seed=1)}
    p = panels["stock"]
    p["sector"] = p["sector"].astype(object)
    p.loc[p.index < 30, "sector"] = "Rare"
    html = build_report(panels, tmp_path / "r.html", as_of="2026-01-01").read_text(encoding="utf-8")
    seg = html[html.index("4. Segment tables"):]
    rows = re.findall(r'<tr class="insuf">(.*?)</tr>', seg, flags=re.S)
    assert rows
    for r in rows:
        cells = re.findall(r"<td>(.*?)</td>", r, flags=re.S)
        assert sum(1 for c in cells if c.strip()) == 6   # signal, sector, dir, n, n solo, dates only
        assert any(c.strip() == "" for c in cells)
    # heatmap hover for insufficient cells: only the insufficient message, no mean/t/q
    hov = re.findall(r'"([^"]*insufficient data \(n=\d+, dates=\d+\)[^"]*)"', html)
    assert hov and all("mean excess" not in h and "q=" not in h and "t=" not in h for h in hov)
    assert any(rare in h for h in hov)


def test_leaderboard_sort_order():
    t = pd.DataFrame({
        "signal": list("abcde"), "insufficient": [True, False, False, False, False],
        "q": [0.0, 0.2, 0.01, 0.01, np.nan], "edge_vs_all_days": [9.0, 1.0, 0.5, 0.7, 5.0]})
    assert list(sort_leaderboard(t)["signal"]) == ["d", "c", "b", "e", "a"]


def test_formatting_bps_and_profit_factor():
    assert _fmt("edge_vs_all_days", 0.001) == "10.0"
    assert _fmt("mean_all", -0.00123) == "-12.3"
    assert _fmt("mean_all", np.nan) == ""
    assert _fmt("profit_factor", np.inf) == "inf"
    assert _fmt("profit_factor", np.nan) == ""
    assert _fmt("profit_factor", 1.2345) == "1.23"


def test_csv_names(tmp_path):
    _render(tmp_path)
    names = {f.name for f in tmp_path.glob("*.csv")}
    exp = {f"solo_{n}_{m}_2026-01-01.csv" for m in ("stock", "crypto")
           for n in ("leaderboard", "heatmap_trend", "heatmap_vol", "segment")}
    assert names == exp


def test_mixed_market_panel_rejected(tmp_path):
    mixed = pd.concat([make_market_panel("stock", seed=1), make_market_panel("crypto", seed=2)],
                      ignore_index=True)
    for c in ("market", "trend_regime", "vol_regime", "sector", "cap_bucket", "split"):
        mixed[c] = mixed[c].astype(str)
    with pytest.raises(ValueError, match="only that market"):
        build_report({"stock": mixed}, tmp_path / "r.html")
    wrong = make_market_panel("crypto")
    with pytest.raises(ValueError, match="only that market"):
        build_report({"stock": wrong}, tmp_path / "r.html")


def test_quality_count_filters_market():
    q = pd.DataFrame({"symbol": ["A", "A", "B"], "market": ["stock", "crypto", None]})
    assert _quality_count(q, {"A", "B"}, "stock") == 2
    assert _quality_count(q, {"A", "B"}, "crypto") == 2
    assert _quality_count(None, {"A"}, "stock") is None


def test_analyze_script_missing_market(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    (data / "panel").mkdir(parents=True)
    make_market_panel("stock").to_parquet(data / "panel" / "stock.parquet", index=False)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(sys, "argv", ["analyze.py", "--horizon", "10"])
    runpy.run_path(str(config._REPO_ROOT / "scripts" / "analyze.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "SKIP crypto" in out
    assert list((tmp_path / "reports").glob("solo_report_*.html"))
