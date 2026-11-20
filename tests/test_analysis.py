import warnings

import numpy as np
import pandas as pd
import pytest

from swingbot import config
from swingbot.analysis.solo import solo_mask, solo_table
from swingbot.analysis.stats import TestResult, bh_qvalues, clustered_mean_test

INFO = {
    "A": ("trend", "long"),
    "B": ("trend", "long"),
    "C": ("meanrev", "short"),
    "R": ("volume", "long"),   # rare
    "Z": ("sr", "long"),       # never fires
}


def make_panel(n_dates=400, assets=("X", "Y", "Z"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    p = idx.to_frame(index=False)
    n = len(p)
    p["A"] = rng.random(n) < 0.3
    p["B"] = rng.random(n) < 0.3
    p["C"] = rng.random(n) < 0.3
    p["R"] = False
    p.loc[[5, 50, 500], "R"] = True
    p["Z"] = False
    for side in ("long", "short"):
        for h in (5, 10, 20):
            p[f"excess_{side}_{h}"] = rng.normal(0, 0.02, n).astype("float32")
            p[f"ret_{side}_{h}"] = rng.normal(0, 0.02, n).astype("float32")
    p["market"] = "stock"
    p["trend_regime"] = np.where(np.arange(n) % 2 == 0, "up", "down")
    p["split"] = "discovery"
    for c in ("market", "trend_regime", "split"):
        p[c] = p[c].astype("category")
    return p


# ---------- stats ----------

def test_clustered_mean_known():
    rng = np.random.default_rng(1)
    dates = pd.Series(pd.bdate_range("2000-01-01", periods=2000))
    vals = pd.Series(rng.normal(0.01, 0.02, 2000))
    r = clustered_mean_test(vals, dates, horizon=10)
    assert isinstance(r, TestResult)
    assert r.mean == pytest.approx(0.01, abs=0.002)
    assert r.p < 0.05
    assert r.n == 2000 and r.n_dates == 2000

    zero = clustered_mean_test(pd.Series(np.zeros(2000)), dates, horizon=10)
    assert np.isnan(zero.p) or zero.p > 0.5


def test_clustered_mean_averages_per_date_and_drops_nan():
    dates = pd.Series(pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-01",
                                      "2020-01-02", "2020-01-02"]))
    vals = pd.Series([0.3, 0.0, 0.2, np.nan, 0.4])
    r = clustered_mean_test(vals, dates, horizon=5)
    # per-date means: 0.1, 0.4, 0.3 -> equal-weighted by date
    assert r.mean == pytest.approx((0.1 + 0.4 + 0.3) / 3)
    assert r.n == 4 and r.n_dates == 3


def test_clustered_collapses_same_day():
    dates = pd.Series([pd.Timestamp("2020-01-02")] * 100)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = clustered_mean_test(pd.Series(np.full(100, 0.01)), dates, horizon=10)
    assert r.n_dates == 1 and r.n == 100
    assert np.isnan(r.se) and np.isnan(r.t) and np.isnan(r.p)
    assert r.mean == pytest.approx(0.01)


def test_clustered_empty():
    r = clustered_mean_test(pd.Series([], dtype=float), pd.Series([], dtype="datetime64[ns]"), 10)
    assert r.n == 0 and r.n_dates == 0 and np.isnan(r.mean) and np.isnan(r.p)


def test_bh_qvalues_known():
    q = bh_qvalues(np.array([0.01, 0.04, 0.03, 0.2]))
    np.testing.assert_allclose(np.round(q, 4), [0.04, 0.0533, 0.0533, 0.2])
    q2 = bh_qvalues(np.array([0.01, np.nan, 0.04, 0.03, 0.2]))
    assert np.isnan(q2[1])
    np.testing.assert_allclose(np.round(q2[[0, 2, 3, 4]], 4), [0.04, 0.0533, 0.0533, 0.2])
    assert bh_qvalues(np.array([])).shape == (0,)
    assert np.isnan(bh_qvalues(np.array([np.nan, np.nan]))).all()


# ---------- solo ----------

def test_solo_mask_other_family():
    p = pd.DataFrame({"A": [True, True, True], "B": [True, False, False],
                      "C": [False, True, False], "R": False, "Z": False})
    m = solo_mask(p, "A", signal_info=INFO)
    assert m.tolist() == [True, False, True]   # A+B same family -> solo; A+C -> not solo
    assert solo_mask(p, "C", signal_info=INFO).tolist() == [False, False, False]


def test_solo_table_columns_and_direction_metric():
    p = make_panel()
    # make C's short metric strongly positive and its long metric strongly negative
    p["excess_short_10"] = np.float32(0.05)
    p["excess_long_10"] = np.float32(-0.05)
    t = solo_table(p, horizon=10, signal_info=INFO)
    expected = ["signal", "family", "direction", "n_all", "n_solo", "mean_all", "mean_solo",
                "edge_vs_all_days", "median_all", "hit_rate", "profit_factor", "t", "p", "q",
                "occ_per_year", "insufficient"]
    assert list(t.columns) == expected
    row = t.set_index("signal").loc["C"]
    assert row["direction"] == "short" and row["family"] == "meanrev"
    assert row["mean_all"] == pytest.approx(0.05, rel=1e-5)
    assert row["edge_vs_all_days"] == pytest.approx(0.0, abs=1e-6)
    assert row["hit_rate"] == 1.0 and np.isinf(row["profit_factor"])
    a = t.set_index("signal").loc["A"]
    assert a["mean_all"] == pytest.approx(-0.05, rel=1e-5)


def test_solo_table_values_match_manual():
    p = make_panel()
    t = solo_table(p, horizon=10, signal_info=INFO).set_index("signal")
    a = p[p["A"]]
    r = clustered_mean_test(a["excess_long_10"].astype(float), a["date"], 10)
    base = clustered_mean_test(p["excess_long_10"].astype(float), p["date"], 10)
    row = t.loc["A"]
    assert row["n_all"] == len(a)
    assert row["mean_all"] == pytest.approx(r.mean)
    assert row["p"] == pytest.approx(r.p)
    assert row["edge_vs_all_days"] == pytest.approx(r.mean - base.mean)
    solo = a[~a["C"]]
    assert row["n_solo"] == len(solo)
    v = a["excess_long_10"].astype(float)
    assert row["hit_rate"] == pytest.approx((v > 0).mean())
    assert row["median_all"] == pytest.approx(v.median())
    assert row["profit_factor"] == pytest.approx(v[v > 0].sum() / -v[v < 0].sum())
    span = (p["date"].max() - p["date"].min()).days / 365.25
    assert row["occ_per_year"] == pytest.approx(len(a) / span)
    assert not row["insufficient"]
    assert np.isfinite(t["q"]).sum() == np.isfinite(t["p"]).sum()


def test_solo_table_rare_signal():
    p = make_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = solo_table(p, horizon=10, signal_info=INFO).set_index("signal")
    r = t.loc["R"]
    assert r["n_all"] == 3 and r["insufficient"]
    for c in ("mean_all", "median_all", "hit_rate", "t", "p", "q"):
        assert np.isfinite(r[c]) or np.isnan(r[c])
    z = t.loc["Z"]
    assert z["n_all"] == 0 and z["insufficient"]
    assert np.isnan(z["profit_factor"]) and np.isnan(z["mean_all"]) and np.isnan(z["q"])


def test_solo_table_groups():
    p = make_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = solo_table(p, horizon=10, group_by=["market", "trend_regime"], signal_info=INFO)
    assert list(t.columns[:5]) == ["signal", "family", "direction", "market", "trend_regime"]
    assert len(t) == len(INFO) * 2
    assert set(map(tuple, t[["market", "trend_regime"]].astype(str).values)) == {
        ("stock", "up"), ("stock", "down")}
    # rows under MIN_N_SLICE flagged; above not
    a_up = t[(t.signal == "A") & (t.trend_regime == "up")].iloc[0]
    sub = p[p["A"] & (p["trend_regime"] == "up")]
    assert a_up["n_all"] == len(sub)
    assert a_up["insufficient"] == (len(sub) < config.MIN_N_SLICE)
    assert t.loc[t.signal == "R", "insufficient"].all()


def test_solo_table_single_group_key_and_nan_keys():
    p = make_panel()
    p["trend_regime"] = p["trend_regime"].astype(object)
    p.loc[p.index[:30], "trend_regime"] = np.nan   # NaN keys dropped, not a group
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = solo_table(p, horizon=10, group_by=["trend_regime"], signal_info=INFO)
    assert len(t) == len(INFO) * 2
    a = t[(t.signal == "A") & (t.trend_regime == "down")].iloc[0]
    assert a["n_all"] == int((p["A"] & (p["trend_regime"] == "down")).sum())


def test_split_filter():
    p = make_panel()
    p["split"] = p["split"].cat.add_categories(["validation"])
    p.loc[p["date"] >= p["date"].iloc[len(p) // 2], "split"] = "validation"
    n_disc = int((p["A"] & (p["split"] == "discovery")).sum())
    t = solo_table(p, horizon=10, signal_info=INFO, split="discovery").set_index("signal")
    assert t.loc["A", "n_all"] == n_disc
    tv = solo_table(p, horizon=10, signal_info=INFO, split="validation").set_index("signal")
    assert tv.loc["A", "n_all"] == int((p["A"] & (p["split"] == "validation")).sum())


def test_solo_table_missing_signal_column_raises():
    p = make_panel().drop(columns=["Z"])
    with pytest.raises(KeyError):
        solo_table(p, horizon=10, signal_info=INFO)


def test_solo_table_ret_metric():
    p = make_panel()
    p["ret_short_5"] = np.float32(0.01)
    t = solo_table(p, horizon=5, metric="ret", signal_info=INFO).set_index("signal")
    assert t.loc["C", "mean_all"] == pytest.approx(0.01, rel=1e-5)
