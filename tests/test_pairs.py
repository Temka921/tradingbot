import numpy as np
import pandas as pd
import pytest

from swingbot.analysis.pairs import (PAIR_COLS, PAIR_VERDICTS, build_context, group_stats,
                                     pair_table)
from swingbot.analysis.stats import TestResult, diff_test

INFO = {"A": ("meanrev", "long"), "B": ("trend", "long"), "S": ("trend", "short")}


def synth(plant_a: float = 0.0, plant_b: float = 0.0, plant_ab: float = 0.0,
          n_dates: int = 400, n_assets: int = 20, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_dates)
    assets = [f"X{i}" for i in range(n_assets)]
    p = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"]).to_frame(index=False)
    n = len(p)
    for name in INFO:
        p[name] = rng.random(n) < 0.15
    for side in ("long", "short"):
        for h in (5, 10, 20):
            p[f"excess_{side}_{h}"] = rng.normal(0, 0.02, n)
            p[f"ret_{side}_{h}"] = rng.normal(0, 0.02, n)
    a, b = p["A"].to_numpy(), p["B"].to_numpy()
    y = p["excess_long_10"].to_numpy().copy()
    y[a] += plant_a
    y[b] += plant_b
    y[a & b] += plant_ab
    p["excess_long_10"] = y
    p["market"] = "crypto"
    p["sector"] = "Tech"
    p["cap_bucket"] = "large"
    p["split"] = "discovery"
    for c in ("market", "sector", "cap_bucket", "split"):
        p[c] = p[c].astype("category")
    return p


def test_diff_test_arithmetic():
    a = TestResult(0.03, 0.01, 3, 0.0, 100, 200); b = TestResult(0.01, 0.01, 1, 0.3, 100, 200)
    r = diff_test(a, b, 10)
    assert r.mean == pytest.approx(0.02) and r.se == pytest.approx(np.sqrt(2) * 0.01)
    assert r.t == pytest.approx(0.02 / (np.sqrt(2) * 0.01))
    assert np.isnan(diff_test(a, TestResult(0.0, np.nan, np.nan, np.nan, 1, 1), 10).p)


def test_diff_test_counts_and_df():
    a = TestResult(0.03, 0.01, 3, 0.0, 100, 200); b = TestResult(0.01, 0.01, 1, 0.3, 50, 60)
    r = diff_test(a, b, 10)
    assert r.n == 150 and r.n_dates == 60
    from scipy import stats as st
    assert r.p == pytest.approx(2 * st.t.sf(abs(r.t), 60 // 20 - 1))


def test_synergy_detected():
    p = synth(plant_ab=+0.03)          # A, B fire independently ~15%; rows with both get +3%
    t = pair_table(p, horizon=10, signal_info=INFO)
    r = t[(t.a == "A") & (t.b == "B")].iloc[0]
    assert r.verdict == "synergy" and r.lift > 0.02 and r.q_lift < 0.10
    assert r.cond_a_given_b == pytest.approx(r.edge_ab - r.edge_b_only)


def test_no_lift_when_a_carries_it():
    p = synth(plant_a=+0.03)           # every A row +3%, B adds nothing
    r = pair_table(p, 10, signal_info=INFO).query("a == 'A' and b == 'B'").iloc[0]
    assert r.verdict != "synergy" and abs(r.lift) < 0.01


def test_conflict_detected():
    p = synth(plant_a=+0.03, plant_b=+0.03, plant_ab=-0.06)
    r = pair_table(p, 10, signal_info=INFO).query("a == 'A' and b == 'B'").iloc[0]
    assert r.verdict == "conflict"


def test_same_direction_only_and_columns():
    t = pair_table(synth(), 10, signal_info=INFO)
    assert set(zip(t.a, t.b)) == {("A", "B")}            # S (short) never pairs with longs
    assert list(t.columns) == PAIR_COLS


def test_a_never_alone_is_redundant_not_synergy():
    p = synth(); p["A"] = p["B"]                           # identical firings
    r = pair_table(p, 10, signal_info=INFO).iloc[0]
    assert r.verdict == "redundant" and r.jaccard == 1.0 and np.isnan(r.lift)


def test_ineligible_rows_never_counted():
    p = synth(plant_ab=+0.03); p["eligible"] = False
    from swingbot.analysis.views import universe_views
    v = universe_views({"crypto": p.assign(market="crypto")})
    assert "crypto" not in v                               # nothing eligible -> no view


def test_signal_never_fires_is_insufficient_not_crash():
    p = synth(); p["A"] = False
    r = pair_table(p, 10, signal_info=INFO).iloc[0]
    assert r.n_ab == 0 and r.insufficient and r.verdict == "insufficient"
    assert np.isnan(r.lift)


def test_empty_panel_gives_insufficient_rows():
    t = pair_table(synth().iloc[:0], 10, signal_info=INFO)
    assert list(t.columns) == PAIR_COLS
    assert len(t) == 1 and t.insufficient.all() and (t.verdict == "insufficient").all()


def test_verdict_order():
    assert PAIR_VERDICTS == ("insufficient", "synergy", "conflict", "redundant", "not significant")


def test_group_stats_matches_pandas():
    p = synth(plant_a=0.01)
    ctx = build_context(p, horizon=10, signal_info=INFO)
    g = group_stats(ctx, ctx.fires["A"], "long")
    from swingbot.analysis.solo import clip_bounds, clip_values
    raw = p["excess_long_10"].to_numpy()
    p = p.assign(y=clip_values(raw, clip_bounds(raw)))
    sub = p[p["A"]]
    daily = sub.groupby("date")["y"].mean()
    base = p.groupby("date")["y"].mean()
    assert g.n == len(sub) and g.n_dates == len(daily)
    assert g.mean == pytest.approx(daily.mean())
    assert g.edge.mean == pytest.approx((daily - base.reindex(daily.index)).mean())


def test_rows_mask_restricts_rows():
    p = synth()
    m = (p["asset"] == "X0").to_numpy()
    ctx = build_context(p, horizon=10, signal_info=INFO, rows_mask=m)
    assert len(ctx.date_code) == int(m.sum())
