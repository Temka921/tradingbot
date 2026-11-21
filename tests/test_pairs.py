import numpy as np
import pandas as pd
import pytest

from swingbot import config
from swingbot.analysis.pairs import (PAIR_COLS, PAIR_VERDICTS, TRIPLE_COLS, REGIME_COLS,
                                     add_pair_verdicts, build_context, group_stats, pair_regime_table,
                                     pair_table, triple_table)
from swingbot.analysis.redundancy import jaccard_matrix, redundancy_groups
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


INFO3 = {"A": ("meanrev", "long"), "B": ("trend", "long"), "C": ("volume", "long"),
         "S": ("trend", "short")}


def synth3(plant_abc: float = 0.0, fire: float = 0.35, n_dates: int = 600, n_assets: int = 20,
           seed: int = 11) -> pd.DataFrame:
    """Three long signals A, B, C (and one short S) firing independently at ``fire``.

    ``plant_abc`` is added to excess_long_10 only on rows where A, B and C all fire.
    The fire rate and size are chosen so the ABC group clears MIN_N rows and dates.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_dates)
    assets = [f"X{i}" for i in range(n_assets)]
    p = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"]).to_frame(index=False)
    n = len(p)
    for name in INFO3:
        p[name] = rng.random(n) < fire
    for side in ("long", "short"):
        for h in (5, 10, 20):
            p[f"excess_{side}_{h}"] = rng.normal(0, 0.02, n)
            p[f"ret_{side}_{h}"] = rng.normal(0, 0.02, n)
    abc = (p["A"] & p["B"] & p["C"]).to_numpy()
    y = p["excess_long_10"].to_numpy().copy()
    y[abc] += plant_abc
    p["excess_long_10"] = y
    p["split"] = pd.Categorical(["discovery"] * n)
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


def _row(**kw):
    base = {"a": "A", "b": "B", "family_a": "meanrev", "family_b": "trend", "direction": "long",
            "jaccard": 0.1, "n_ab": 500, "n_dates_ab": 300, "n_a_only": 500, "n_b_only": 500,
            "mean_ab": 0.01, "edge_ab": 0.01, "t_ab": 3.0, "p_ab": 0.01, "q_ab": 0.01,
            "edge_a_only": 0.0, "edge_b_only": 0.0, "cond_a_given_b": 0.01, "cond_b_given_a": 0.01,
            "lift": 0.01, "t_lift": 3.0, "p_lift": 0.01, "q_lift": 0.01,
            "insufficient": False, "verdict": ""}
    base.update(kw)
    return base


def test_significant_lift_is_never_redundant():
    # significant positive lift but the combo itself loses money, overlap high
    t = pd.DataFrame([_row(jaccard=0.8, mean_ab=-0.002, lift=0.01, q_lift=0.01),
                      _row(jaccard=0.8, q_lift=0.5)], columns=PAIR_COLS)
    v = add_pair_verdicts(t, 10)["verdict"].tolist()
    assert v == ["not significant", "redundant"]


def test_verdicts_on_concatenated_frame_need_no_attrs():
    t = pd.concat([pd.DataFrame([_row(n_a_only=0, insufficient=True, lift=np.nan, q_lift=np.nan)],
                                columns=PAIR_COLS),
                   pd.DataFrame([_row(n_a_only=0, n_dates_ab=150, insufficient=True,
                                      lift=np.nan, q_lift=np.nan)], columns=PAIR_COLS)],
                  ignore_index=True)
    assert add_pair_verdicts(t, 10)["verdict"].tolist() == ["redundant", "redundant"]
    # h = 40 needs 200 combo dates: the second row is now too small to be called redundant
    assert add_pair_verdicts(t, 40)["verdict"].tolist() == ["redundant", "insufficient"]


# --- Task 7: triples, regime slices, redundancy map ------------------------------------------

def test_triple_synergy():
    p = synth3(plant_abc=+0.04)        # three long signals A, B, C; +4% only when all three fire
    pairs = pair_table(p, 10, signal_info=INFO3)
    t = triple_table(p, pairs, 10, signal_info=INFO3)
    # pairs carry part of the effect, so lifts > 0 and the triple is evaluated
    assert len(t) == 1 and t.iloc[0].lift > 0 and t.iloc[0].verdict in ("synergy", "not significant")


def test_triple_synergy_columns_and_values():
    p = synth3(plant_abc=+0.04)
    t = triple_table(p, pair_table(p, 10, signal_info=INFO3), 10, signal_info=INFO3)
    assert list(t.columns) == TRIPLE_COLS and "jaccard" not in t.columns
    r = t.iloc[0]
    assert (r.a, r.b, r.c, r.direction) == ("A", "B", "C", "long")
    assert not r.insufficient and r.n_abc >= config.MIN_N
    assert r.lift == pytest.approx(r.edge_abc - r.edge_best_pair_only)
    assert r.lift > 0.03 and r.verdict == "synergy"
    assert r.best_pair in ("A+B", "A+C", "B+C")


def test_triple_empty_when_no_positive_pairs():
    p = synth3()
    t = triple_table(p, pair_table(p, 10, signal_info=INFO3).assign(lift=-1.0), 10, signal_info=INFO3)
    assert t.empty and "lift" in t.columns


def test_triple_needs_all_three_pairs():
    p = synth3(plant_abc=+0.04)
    pairs = pair_table(p, 10, signal_info=INFO3)
    pairs = pairs[~((pairs.a == "B") & (pairs.b == "C"))]
    assert triple_table(p, pairs, 10, signal_info=INFO3).empty
    ins = pair_table(p, 10, signal_info=INFO3)
    ins.loc[(ins.a == "A") & (ins.b == "C"), "insufficient"] = True
    assert triple_table(p, ins, 10, signal_info=INFO3).empty


def test_triple_cap(monkeypatch):
    p = synth3(plant_abc=+0.04)
    monkeypatch.setattr(config, "MAX_TRIPLES", 0)
    t = triple_table(p, pair_table(p, 10, signal_info=INFO3), 10, signal_info=INFO3)
    assert t.empty and list(t.columns) == TRIPLE_COLS
    assert config.MAX_TRIPLES == 0
    monkeypatch.undo()
    assert config.MAX_TRIPLES == 500


def test_triple_never_redundant_when_c_always_fires_with_ab():
    p = synth3(plant_abc=+0.04)
    p["C"] = p["C"] | (p["A"] & p["B"])          # A and B never fire together without C
    pairs = pair_table(p, 10, signal_info=INFO3).assign(lift=0.01, insufficient=False)
    t = triple_table(p, pairs, 10, signal_info=INFO3)
    assert len(t) == 1 and t.iloc[0].insufficient and t.iloc[0].verdict == "insufficient"
    assert np.isnan(t.iloc[0].lift)


def test_pair_regime_table_same_regime_baseline():
    p = synth(plant_ab=+0.03); p["trend_regime"] = np.where(p.index % 2 == 0, "up", "down")
    pairs = pair_table(p, 10, signal_info=INFO)
    r = pair_regime_table(p, pairs, 10, signal_info=INFO)
    assert set(r.regime) == {"up", "down"} and (r.edge_ab > 0.02).all()


def test_pair_regime_table_skips_unknown_and_columns():
    p = synth(plant_ab=+0.03)
    p["trend_regime"] = np.where(p.index % 3 == 0, "unknown", np.where(p.index % 2 == 0, "up", "down"))
    r = pair_regime_table(p, pair_table(p, 10, signal_info=INFO), 10, signal_info=INFO)
    assert list(r.columns) == REGIME_COLS and "unknown" not in set(r.regime)
    assert (r.q_ab >= r.p_ab).all() and not r.insufficient.any()


def test_pair_regime_table_slice_insufficient():
    p = synth(plant_ab=+0.03)
    p["trend_regime"] = np.where(p["date"] < p["date"].iloc[0] + pd.Timedelta(days=20), "rare", "up")
    r = pair_regime_table(p, pair_table(p, 10, signal_info=INFO), 10, signal_info=INFO)
    assert r.set_index("regime").loc["rare", "insufficient"]
    assert not r.set_index("regime").loc["up", "insufficient"]


def test_pair_regime_table_empty_without_synergy():
    p = synth(); p["trend_regime"] = "up"
    r = pair_regime_table(p, pair_table(p, 10, signal_info=INFO), 10, signal_info=INFO)
    assert r.empty and list(r.columns) == REGIME_COLS


def test_redundancy_groups():
    jac = pd.DataFrame([[1, .9, 0], [.9, 1, .1], [0, .1, 1]], index=list("XYZ"), columns=list("XYZ"), dtype=float)
    assert redundancy_groups(jac, 0.5) == [["X", "Y"]]


def test_redundancy_groups_transitive_and_order():
    names = list("PQRST")
    m = np.eye(5)
    for i, j in [(0, 3), (3, 4), (1, 2)]:        # P-S-T chain, Q-R pair
        m[i, j] = m[j, i] = 0.6
    m[0, 1] = m[1, 0] = np.nan
    jac = pd.DataFrame(m, index=names, columns=names)
    assert redundancy_groups(jac, 0.5) == [["P", "S", "T"], ["Q", "R"]]
    assert redundancy_groups(jac) == [["P", "S", "T"], ["Q", "R"]]   # default REDUNDANT_JACCARD
    assert redundancy_groups(jac, 0.7) == []


def test_jaccard_matrix_duplicate_signals():
    # Real-registry sanity check (Task 9): bb_lower_close and zscore_below_2 must end up in
    # one redundancy group on real data.
    p = synth(); p["B"] = p["A"]
    j = jaccard_matrix(p, signal_info=INFO)
    assert j.loc["A", "B"] == 1.0 and j.loc["A", "A"] == 1.0


def test_jaccard_matrix_values_and_split():
    p = synth()
    p["split"] = pd.Categorical(np.where(p.index % 4 == 0, "validation", "discovery"))
    p.loc[p["split"] == "validation", "A"] = True       # must not leak into discovery
    p["S"] = False
    j = jaccard_matrix(p, signal_info=INFO)
    d = p[p["split"] == "discovery"]
    a, b = d["A"].to_numpy(), d["B"].to_numpy()
    assert list(j.index) == list(INFO) and list(j.columns) == list(INFO)
    assert j.loc["A", "B"] == pytest.approx((a & b).sum() / (a | b).sum())
    assert j.loc["A", "B"] == j.loc["B", "A"]
    assert j.loc["A", "S"] == 0.0 and j.loc["S", "S"] == 1.0
    p2 = p.copy(); p2["B"] = False
    assert np.isnan(jaccard_matrix(p2, signal_info=INFO).loc["B", "S"])
