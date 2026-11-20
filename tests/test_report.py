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
        # signal, sector, dir, verdict, n, n solo, dates only
        assert sum(1 for c in cells if c.strip()) == 7
        assert "insufficient" in cells
        assert any(c.strip() == "" for c in cells)
    # heatmap hover for insufficient cells: only the insufficient message, no mean/t/q
    hov = re.findall(r'"([^"]*insufficient data \(n=\d+, dates=\d+\)[^"]*)"', html)
    assert hov and all("mean excess" not in h and "q=" not in h and "t=" not in h for h in hov)
    assert any(rare in h for h in hov)


def test_leaderboard_sort_order():
    t = pd.DataFrame({
        "signal": list("abcde"), "insufficient": [True, False, False, False, False],
        "q": [0.0, 0.2, 0.01, 0.01, np.nan], "edge_vs_all_days": [9.0, 1.0, 0.5, 0.7, 5.0],
        "mean_all": [1.0, 1.0, 1.0, 1.0, 1.0]})
    assert list(sort_leaderboard(t)["signal"]) == ["d", "c", "b", "e", "a"]


def _verdict_fixture():
    return pd.DataFrame({
        "signal": ["neg", "lossnet", "cand", "cand2", "ns", "insuf", "nanq"],
        "insufficient": [False, False, False, False, False, True, False],
        "q": [0.001, 0.01, 0.05, 0.05, 0.4, 0.0, np.nan],
        "edge_vs_all_days": [-0.004, 0.002, 0.001, 0.003, 0.01, 0.05, 0.02],
        "mean_all": [0.002, -0.001, 0.0005, 0.0001, 0.02, 0.05, 0.02]})


def test_verdicts_and_sort():
    from swingbot.report.build import VERDICTS, add_verdicts, verdict_counts
    t = add_verdicts(_verdict_fixture()).set_index("signal")["verdict"]
    assert t["neg"] == "significant negative edge (avoid)"
    assert t["lossnet"] == "beats same-day avg but loses net of costs"
    assert t["cand"] == t["cand2"] == "candidate"
    assert t["ns"] == t["nanq"] == "not significant"
    assert t["insuf"] == "insufficient"
    counts = verdict_counts(add_verdicts(_verdict_fixture()))
    assert list(counts) == VERDICTS
    assert counts["candidate"] == 2 and counts["significant negative edge (avoid)"] == 1
    s = sort_leaderboard(_verdict_fixture())
    # candidates first (q asc, then edge desc), then avoid, then loses-net, then not significant, insufficient
    assert list(s["signal"]) == ["cand2", "cand", "neg", "lossnet", "ns", "nanq", "insuf"]
    assert "verdict" in s.columns


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
    q = pd.DataFrame({"symbol": ["A", "A", "B"], "market": ["stock", "crypto", None],
                      "issues": ["gap", "gap", "gap"]})
    assert _quality_count(q, {"A", "B"}, "stock") == 2
    assert _quality_count(q, {"A", "B"}, "crypto") == 2
    assert _quality_count(None, {"A"}, "stock") is None


def test_rendered_verdicts_not_counted_as_hits(tmp_path):
    panels = {"stock": make_market_panel("stock", seed=1)}
    p = panels["stock"]
    names = list(SIGNAL_INFO)
    neg, pos = names[1], names[2]
    for name, shift in ((neg, -0.03), (pos, 0.03)):
        col = f"excess_{SIGNAL_INFO[name][1]}_10"
        p[col] = p[col].astype("float64")
        p.loc[p[name], col] += shift
    # make every pos value net-negative while it still beats the same-day average
    pcol = f"excess_{SIGNAL_INFO[pos][1]}_10"
    p[pcol] = p[pcol] - 0.05
    html = build_report(panels, tmp_path / "r.html", as_of="2026-01-01").read_text(encoding="utf-8")
    lb = pd.read_csv(tmp_path / "solo_leaderboard_stock_2026-01-01.csv").set_index("signal")
    assert lb.loc[neg, "verdict"] == "significant negative edge (avoid)"
    assert lb.loc[pos, "verdict"] == "beats same-day avg but loses net of costs"
    assert (lb["verdict"] == "candidate").sum() == int(
        ((~lb["insufficient"]) & (lb["q"] < 0.10) & (lb["edge_vs_all_days"] > 0) & (lb["mean_all"] > 0)).sum())
    assert "with q &lt; 0.10" not in html
    assert re.search(r"candidate: \d+", html)
    assert "cost-neutral" in html
    assert lb.iloc[0]["verdict"] in ("candidate", "significant negative edge (avoid)")


def _etf_panel(seed=1):
    p = make_market_panel("stock", seed=seed)
    p["is_etf"] = p["asset"] == "A3"
    return p


def test_etf_excluded_and_control_table(tmp_path):
    p = _etf_panel()
    html = build_report({"stock": p}, tmp_path / "r.html", as_of="2026-01-01").read_text(encoding="utf-8")
    sig = list(SIGNAL_INFO)[1]
    exp_stock = int((p[sig] & ~p["is_etf"]).sum())
    exp_etf = int((p[sig] & p["is_etf"]).sum())
    lb = pd.read_csv(tmp_path / "solo_leaderboard_stock_2026-01-01.csv").set_index("signal")
    etf = pd.read_csv(tmp_path / "solo_etf_control_stock_2026-01-01.csv").set_index("signal")
    assert lb.loc[sig, "n_all"] == exp_stock
    assert etf.loc[sig, "n_all"] == exp_etf
    assert "verdict" in etf.columns
    for name in ("heatmap_trend", "heatmap_vol", "segment"):
        t = pd.read_csv(tmp_path / f"solo_{name}_stock_2026-01-01.csv")
        assert t.loc[t.signal == sig, "n_all"].sum() == exp_stock
    assert "ETF-only control" in html and "no constituent survivorship" in html


def test_survivorship_caveat_number(tmp_path):
    from swingbot.analysis.solo import clip_bounds, clip_values
    from swingbot.report.build import CAVEAT_EXTRA
    p = _etf_panel()
    p["excess_long_10"] = (p["excess_long_10"] + 0.0035).astype("float32")
    html = build_report({"stock": p}, tmp_path / "r.html", as_of="2026-01-01").read_text(encoding="utf-8")
    s = p[~p["is_etf"] & (p["split"] == "discovery")]
    v = s["excess_long_10"].to_numpy(dtype="float64")
    v = clip_values(v, clip_bounds(v))
    x = pd.Series(v).groupby(s["date"].to_numpy()).mean().mean() * 1e4
    txt = f"{x:+.1f} bps per 10 days"
    assert txt in html
    i_cav, i_surv, i_extra = html.index(CAVEAT), html.index(txt), html.index(CAVEAT_EXTRA)
    assert i_cav < i_surv < i_extra
    assert "S&amp;P 500/400 constituents" in html and "dip-buying" in html
    assert "top" in html and "selection on later success" in html


def _plotly_traces(html: str) -> list[list[dict]]:
    import json
    out, pos = [], 0
    while (i := html.find("Plotly.newPlot(", pos)) >= 0:
        j = html.index("[", i)
        data, _ = json.JSONDecoder().raw_decode(html[j:])
        out.append(data)
        pos = j
    return out


def test_heatmap_grey_for_not_significant(tmp_path):
    # 6 assets so every regime cell has several assets per date (else d = 0 and p is NaN)
    p = make_market_panel("stock", n_dates=600, assets=tuple(f"A{i}" for i in range(6)), seed=4)
    strong = list(SIGNAL_INFO)[1]
    col = f"excess_{SIGNAL_INFO[strong][1]}_10"
    p[col] = p[col].astype("float64")
    p.loc[p[strong], col] += 0.03
    html = build_report({"stock": p}, tmp_path / "r.html", as_of="2026-01-01").read_text(encoding="utf-8")
    assert ("Blue = signal beats same-day average, red = trails; grey = not significant (q ≥ 0.10); "
            "blank = insufficient data. BH correction is per heatmap.") in html
    figs = _plotly_traces(html)
    assert len(figs) == 2
    for traces in figs:
        by = {t["name"]: t for t in traces}
        sig, grey = by["significant"], by["not significant"]
        ys = sig["y"]
        r = ys.index(strong)
        assert all(v is not None for v in sig["z"][r]) and all(v is None for v in grey["z"][r])
        n_grey = 0
        for zs, zg in zip(sig["z"], grey["z"]):
            for a, b in zip(zs, zg):
                assert a is None or b is None          # never both
                n_grey += b is not None
        assert n_grey > 0
        # not-significant hover still shows the values
        assert any("not significant" in t and "mean excess" in t for row in grey["text"] for t in row)
    tbl = pd.read_csv(tmp_path / "solo_heatmap_trend_stock_2026-01-01.csv")
    ok = tbl[~tbl.insufficient]
    assert ((ok.q < 0.10).sum() > 0) and ((ok.q >= 0.10).sum() > 0)


def test_leaderboard_labels_notes_and_clip_bounds(tmp_path):
    from swingbot.analysis.solo import clip_bounds
    panels, out = _render(tmp_path)
    html = out.read_text(encoding="utf-8")
    assert "Signal leaderboard (all firings)" in html and "Solo leaderboard" not in html
    assert "Mean excess solo (date-weighted, bps)" in html and "n solo" in html
    assert re.search(r"bb_lower_close</code> and <code>zscore_below_2</code> are the same test", html)
    assert "Clip bounds" in html
    lo, hi = clip_bounds(panels["crypto"]["excess_long_10"].to_numpy(dtype="float64"))
    assert f"{lo * 100:.2f}% to {hi * 100:.2f}%" in html


def test_quality_count_only_nonempty_issues_deduped():
    q = pd.DataFrame({"symbol": ["A", "A", "B", "C", "D"],
                      "issues": ["gap", "gap", np.nan, "gap", ""],
                      "market": ["stock", np.nan, "stock", "crypto", "stock"]})
    assert _quality_count(q, {"A", "B", "C", "D"}, "stock") == 1
    assert _quality_count(q, {"A", "B", "C", "D"}, "crypto") == 2


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
