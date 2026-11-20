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
