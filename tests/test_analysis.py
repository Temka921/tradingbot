import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats as scipy_stats

from swingbot import config
from swingbot.analysis.solo import solo_mask, solo_table
from swingbot.analysis.stats import (TestResult, bh_qvalues, clustered_mean_test,
                                     date_matched_edge_test, per_date_means)

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
    expected = ["signal", "family", "direction", "n_all", "n_solo", "n_dates", "mean_all", "mean_solo",
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
    # statistics are computed on values clipped at pooled 0.1/99.9 percentiles of the split
    v_all = p["excess_long_10"].to_numpy(dtype="float64")
    lo, hi = np.percentile(v_all, (0.1, 99.9))
    p["excess_long_10"] = np.clip(v_all, lo, hi)
    a = p[p["A"]]
    r = clustered_mean_test(a["excess_long_10"].astype(float), a["date"], 10)
    # independent date-matched edge: per-date signal mean minus per-date all-rows mean
    sig_d = a.groupby("date")["excess_long_10"].mean().astype(float)
    all_d = p.groupby("date")["excess_long_10"].mean().astype(float)
    d = (sig_d - all_d.reindex(sig_d.index)).sort_index().to_numpy()
    fit = sm.OLS(d, np.ones((len(d), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 20}, use_t=True)
    row = t.loc["A"]
    assert row["n_all"] == len(a)
    assert row["n_dates"] == a["date"].nunique()
    assert row["mean_all"] == pytest.approx(r.mean)
    assert row["edge_vs_all_days"] == pytest.approx(d.mean())
    assert row["t"] == pytest.approx(fit.tvalues[0])
    assert row["p"] == pytest.approx(2 * scipy_stats.t.sf(abs(fit.tvalues[0]), max(len(d) // 20 - 1, 1)))
    assert row["n_solo"] == int((p["A"] & solo_mask(p, "A", signal_info=INFO)).sum())
    assert row["n_solo"] == int((p["A"] & ~p["C"] & ~p["R"] & ~p["Z"]).sum())
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


# ---------- fix round 1: HAC settings, date-matched edge, n_dates ----------

def test_hac_settings_and_min_dates():
    rng = np.random.default_rng(3)
    dates = pd.Series(pd.bdate_range("2001-01-01", periods=300))
    y = rng.normal(0.001, 0.02, 300)
    r = clustered_mean_test(pd.Series(y), dates, horizon=10)
    fit = sm.OLS(y, np.ones((300, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 20}, use_t=True)
    assert r.se == pytest.approx(fit.bse[0]) and r.t == pytest.approx(fit.tvalues[0])
    # p from a t distribution with effective df = 300 // 20 - 1 = 14, not 299
    assert r.p == pytest.approx(2 * scipy_stats.t.sf(abs(fit.tvalues[0]), 14))
    assert r.p > fit.pvalues[0]
    # fewer than 5*horizon dates -> no p-value; exactly 5*horizon -> p-value
    short = clustered_mean_test(pd.Series(y[:49]), dates[:49], horizon=10)
    assert np.isnan(short.p) and np.isnan(short.se) and np.isfinite(short.mean)
    assert np.isfinite(clustered_mean_test(pd.Series(y[:50]), dates[:50], horizon=10).p)


def test_hac_sees_date_sorted_input():
    rng = np.random.default_rng(4)
    n = 600
    dates = pd.bdate_range("2003-01-01", periods=n)
    e = rng.normal(0, 1, n + 9)
    y = np.convolve(e, np.ones(10), "valid")[:n] * 0.01 + 0.002   # strongly autocorrelated
    perm = rng.permutation(n)
    shuffled = clustered_mean_test(pd.Series(y[perm]), pd.Series(dates[perm]), horizon=10)
    ordered = sm.OLS(y, np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 20}, use_t=True)
    wrong = sm.OLS(y[perm], np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 20}, use_t=True)
    assert shuffled.se == pytest.approx(ordered.bse[0])
    assert abs(wrong.bse[0] - ordered.bse[0]) > 0.1 * ordered.bse[0]  # order matters


def test_date_matched_edge_test_unsorted_signal_index():
    base = pd.Series([0.0, 0.1, 0.2], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    sig = pd.Series([0.5, 0.2], index=pd.to_datetime(["2020-01-03", "2020-01-01"]))
    r = date_matched_edge_test(sig, base, horizon=1, n=2)
    assert r.mean == pytest.approx((0.3 + 0.2) / 2) and r.n_dates == 2


def test_insufficient_when_few_dates_many_occurrences():
    dates = pd.bdate_range("2010-01-01", periods=400)
    assets = [f"S{i}" for i in range(40)]
    p = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"]).to_frame(index=False)
    rng = np.random.default_rng(5)
    p["excess_long_10"] = rng.normal(0, 0.02, len(p)).astype("float32")
    p["split"] = pd.Categorical(["discovery"] * len(p))
    p["F"] = p["date"].isin(dates[[10, 100, 200, 300, 390]])   # 200 firings on 5 dates
    t = solo_table(p, horizon=10, signal_info={"F": ("trend", "long")}).iloc[0]
    assert t["n_all"] == 200 and t["n_dates"] == 5
    assert t["insufficient"]
    assert np.isnan(t["p"]) and np.isnan(t["q"])


def test_edge_removes_common_drift():
    """With metric=ret, a market-wide drift must not make a random signal significant."""
    p = make_panel(n_dates=1500, assets=tuple(f"S{i}" for i in range(10)), seed=7)
    p["ret_long_10"] = (p["ret_long_10"] + 0.01).astype("float32")
    t = solo_table(p, horizon=10, metric="ret", signal_info=INFO).set_index("signal")
    a = t.loc["A"]
    assert a["mean_all"] == pytest.approx(0.01, abs=0.002)
    assert abs(a["edge_vs_all_days"]) < 0.002
    assert a["p"] > 0.01
    # the old mean != 0 test would have called it a sure thing
    sub = p[p["A"]]
    assert clustered_mean_test(sub["ret_long_10"].astype(float), sub["date"], 10).p < 1e-6


def test_edge_detects_real_edge():
    p = make_panel(n_dates=1500, assets=tuple(f"S{i}" for i in range(10)), seed=8)
    p.loc[p["A"], "excess_long_10"] += np.float32(0.01)
    a = solo_table(p, horizon=10, signal_info=INFO).set_index("signal").loc["A"]
    assert a["edge_vs_all_days"] == pytest.approx(0.007, abs=0.002)  # 0.01 * (1 - share firing)
    assert a["p"] < 1e-6


def _markov_runs(rng, n_dates, n_assets, run_len=10.0, occupancy=0.03):
    """On/off Markov firing: mean on-run ``run_len`` days, long-run share ``occupancy``."""
    p_off = 1.0 / run_len
    p_on = p_off * occupancy / (1.0 - occupancy)
    u = rng.random((n_dates, n_assets))
    s = np.zeros((n_dates, n_assets), dtype=bool)
    s[0] = u[0] < occupancy
    for t in range(1, n_dates):
        s[t] = np.where(s[t - 1], u[t] >= p_off, u[t] < p_on)
    return s.ravel()


def _null_size(regime, reps, seed, h=10, n_dates=1500, n_assets=10):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2000-01-01", periods=n_dates).values, n_assets)
    rejections = 0
    for _ in range(reps):
        r = rng.normal(0, 0.01, (n_dates + h, n_assets)) + rng.normal(0, 0.01, (n_dates + h, 1))
        c = np.cumsum(r, axis=0)
        v = (c[h:] - c[:-h])[:n_dates].ravel()          # overlapping h-day forward sums
        if regime == "iid":
            f = rng.random(n_dates * n_assets) < 0.3
        else:
            f = _markov_runs(rng, n_dates, n_assets)
        res = date_matched_edge_test(per_date_means(v[f], dates[f]),
                                     per_date_means(v, dates), h, int(f.sum()))
        rejections += res.p < 0.05
    return rejections / reps


def test_edge_test_null_size():
    """Null calibration, dense iid firings: size <= 8% at nominal 5%."""
    assert _null_size("iid", reps=200, seed=123) <= 0.08


def test_edge_test_null_size_sparse_runs():
    """Null calibration, sparse Markov runs (~10-day runs, ~3% occupancy): size <= 8% at 5%."""
    assert _null_size("markov", reps=300, seed=7) <= 0.08


def test_clip_outlier_does_not_dominate_edge():
    p = make_panel(n_dates=1500, assets=tuple(f"S{i}" for i in range(10)), seed=11)
    base = solo_table(p, horizon=10, signal_info=INFO).set_index("signal").loc["A", "edge_vs_all_days"]
    row = p.index[p["A"]][500]
    p.loc[row, "excess_long_10"] = np.float32(300.0)       # a 300x data splice on one date
    a = solo_table(p, horizon=10, signal_info=INFO).set_index("signal").loc["A"]
    assert abs(a["edge_vs_all_days"]) < 0.002
    assert abs(a["edge_vs_all_days"] - base) < 0.0005
    assert a["p"] > 1e-3


def test_clip_symmetric_pooled_percentiles():
    from swingbot.analysis.solo import CLIP_PCT, clip_bounds
    p = make_panel(n_dates=1500, assets=tuple(f"S{i}" for i in range(10)), seed=12)
    p.loc[p.index[p["A"]][100], "excess_long_10"] = np.float32(300.0)
    p.loc[p.index[p["A"]][200], "excess_long_10"] = np.float32(-300.0)
    p["split"] = p["split"].astype(object)
    p.loc[p.index[:50], "split"] = "validation"
    v = p.loc[p["split"] == "discovery", "excess_long_10"].to_numpy(dtype="float64")
    lo, hi = np.percentile(v, CLIP_PCT)
    assert CLIP_PCT == (0.1, 99.9)
    assert clip_bounds(v) == pytest.approx((lo, hi))
    t = solo_table(p, horizon=10, signal_info=INFO)
    assert t.attrs["clip_bounds"]["excess_long_10"] == pytest.approx((lo, hi))
    assert t.attrs["clip_bounds"]["excess_short_10"][0] < 0 < t.attrs["clip_bounds"]["excess_short_10"][1]
    # both tails clipped: a +300 and a -300 cancel out of A's mean instead of swinging it
    a = t.set_index("signal").loc["A"]
    assert abs(a["mean_all"]) < 0.002 and abs(a["edge_vs_all_days"]) < 0.002


def test_insufficient_when_dates_below_test_minimum():
    """Rows with >= MIN_N_SLICE dates but < 5*horizon dates have NaN p, so must be insufficient."""
    dates = pd.bdate_range("2010-01-01", periods=400)
    assets = [f"S{i}" for i in range(5)]
    p = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"]).to_frame(index=False)
    rng = np.random.default_rng(9)
    p["excess_long_10"] = rng.normal(0, 0.02, len(p)).astype("float32")
    p["excess_long_20"] = rng.normal(0, 0.02, len(p)).astype("float32")
    p["market"] = pd.Categorical(["stock"] * len(p))
    p["split"] = pd.Categorical(["discovery"] * len(p))
    p["F"] = p["date"].isin(dates[::10][:40]) & p["asset"].isin(assets[:3])   # 40 dates x 3 = 120
    info = {"F": ("trend", "long")}
    r10 = solo_table(p, horizon=10, group_by=["market"], signal_info=info).iloc[0]
    assert r10["n_dates"] == 40 and r10["n_all"] == 120
    assert np.isnan(r10["p"]) and r10["insufficient"]
    # 60 dates on 3 of 5 assets (so the date-matched d series is not constant):
    # enough dates for h=10, not for h=20
    p["F"] = p["date"].isin(dates[::4][:60]) & p["asset"].isin(assets[:3])
    r10 = solo_table(p, horizon=10, group_by=["market"], signal_info=info).iloc[0]
    r20 = solo_table(p, horizon=20, group_by=["market"], signal_info=info).iloc[0]
    assert np.isfinite(r10["p"]) and not r10["insufficient"]
    assert np.isnan(r20["p"]) and r20["insufficient"]


# ---------- universe views (option C) ----------
def test_universe_views_split_and_eligible():
    from swingbot.analysis.views import universe_views
    from tests.test_report import make_market_panel
    stock = make_market_panel("stock", assets=("SPY", "AAA"))
    stock["is_etf"] = stock["asset"] == "SPY"
    crypto = make_market_panel("crypto", assets=("BTCUSDT", "XUSDT"))
    crypto["eligible"] = crypto["asset"] == "BTCUSDT"
    v = universe_views({"stock": stock, "crypto": crypto})
    assert list(v) == ["etf", "crypto", "stock"]
    assert set(v["etf"]["asset"]) == {"SPY"} and set(v["stock"]["asset"]) == {"AAA"}
    assert set(v["crypto"]["asset"]) == {"BTCUSDT"}
    assert (v["etf"]["view"] == "etf").all()
    assert str(v["etf"]["view"].dtype) == "category"


def test_universe_views_missing_columns_and_empty_views():
    from swingbot.analysis.views import universe_views
    from tests.test_report import make_market_panel
    v = universe_views({"stock": make_market_panel("stock")})   # no is_etf, no eligible
    assert list(v) == ["stock"] and len(v["stock"]) == 900


def test_universe_views_rejects_wrong_market():
    import pytest
    from swingbot.analysis.views import universe_views
    from tests.test_report import make_market_panel
    with pytest.raises(ValueError, match="only that market"):
        universe_views({"stock": make_market_panel("crypto")})


def test_universe_views_does_not_filter_on_delisted():
    from swingbot.analysis.views import universe_views
    from tests.test_report import make_market_panel
    c = make_market_panel("crypto", assets=("A", "B"))
    c["delisted"] = c["asset"] == "B"
    v = universe_views({"crypto": c})
    assert set(v["crypto"]["asset"]) == {"A", "B"} and "delisted" in v["crypto"]
