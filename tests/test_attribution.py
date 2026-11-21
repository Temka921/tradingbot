import numpy as np
import pandas as pd
import pytest

from swingbot import config
from swingbot.analysis.attribution import (REGIME_FEATURES, feature_matrix, fit_discovery_score_validation,
                                           fit_model, walk_forward)

INFO = {"A": ("meanrev", "long"), "B": ("trend", "long"), "S": ("trend", "short")}


def synth_ml(n_dates=1500, n_assets=30, plant=0.03, seed=0, leak=False):
    # business dates from 2010; signals A (long), B (long), S (short) ~ Bernoulli(0.2);
    # excess_long_10 ~ N(0, 0.02) + plant on rows where A fires; trend_regime random in {up, down, unknown};
    # split = "discovery"; leak=True instead plants on rows whose date is in the first 11 dates of each year
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_dates)
    assets = [f"X{i}" for i in range(n_assets)]
    p = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"]).to_frame(index=False)
    n = len(p)
    for name in INFO:
        p[name] = rng.random(n) < 0.2
    y = rng.normal(0, 0.02, n)
    if leak:
        d = pd.Series(dates)
        first11 = set(d.groupby(d.dt.year).head(11))
        y[p["date"].isin(first11).to_numpy()] += plant
    else:
        y[p["A"].to_numpy()] += plant
    p["excess_long_10"] = y
    p["trend_regime"] = pd.Categorical(rng.choice(["up", "down", "unknown"], n))
    p["split"] = "discovery"
    return p


def test_regime_features_constant():
    assert REGIME_FEATURES == ["trend_regime", "vol_regime", "market_regime", "adx_regime"]


def test_feature_matrix_columns():
    p = synth_ml(200, 5)
    X = feature_matrix(p, signal_info=INFO)
    assert list(X.columns[:3]) == ["A", "B", "S"]
    assert "trend_regime=up" in X and "trend_regime=unknown" not in X
    assert X.dtypes.eq("float32").all() and len(X) == len(p)


def test_feature_matrix_values_and_index():
    p = synth_ml(50, 4).iloc[::-1]                 # unusual row order must be kept
    X = feature_matrix(p, signal_info=INFO)
    assert X.index.equals(p.index)
    assert list(X.columns) == ["A", "B", "S", "trend_regime=down", "trend_regime=up"]
    assert (X["A"].to_numpy() == p["A"].to_numpy().astype("float32")).all()
    assert (X["trend_regime=up"].to_numpy() == (p["trend_regime"] == "up").to_numpy()).all()


def test_walk_forward_learns_planted_signal():
    out = walk_forward(synth_ml(), horizon=10, signal_info=INFO)
    assert set(out["fold"]) == {2013, 2014, 2015}        # 1500 bdays ~ 2010-2015, 3 training years first
    assert list(out.columns) == ["date", "asset", "fold", "pred", "target"]
    p = synth_ml()[["date", "asset", "A"]]
    m = out.merge(p, on=["date", "asset"])
    assert m.loc[m.A == 1, "pred"].mean() - m.loc[m.A == 0, "pred"].mean() > 0.01
    assert set(out.attrs["models"]) == {2013, 2014, 2015}


def test_embargo_excludes_last_h_plus_1_dates():
    p = synth_ml(); out = walk_forward(p, horizon=10, signal_info=INFO)
    tr = out.attrs["train_dates"][2014]                    # dates used to train the 2014 model
    first_test = out.loc[out.fold == 2014, "date"].min()
    all_dates = np.sort(p["date"].unique())
    cutoff = all_dates[np.searchsorted(all_dates, first_test) - 11]
    assert tr.max() < cutoff


def test_embargo_on_leaky_panel_every_fold():
    # the target is planted on the first 11 dates of each year; no training date of a fold may be
    # a test date or one of the h+1 dates just before the fold
    p = synth_ml(leak=True); out = walk_forward(p, horizon=10, signal_info=INFO)
    all_dates = np.sort(p["date"].unique())
    for year, tr in out.attrs["train_dates"].items():
        test = out.loc[out.fold == year, "date"]
        i = np.searchsorted(all_dates, test.min())
        banned = set(all_dates[i - 11:])
        assert not banned.intersection(tr)
        assert pd.Timestamp(tr.max()).year < year


def test_target_clipped_with_training_bounds():
    p = synth_ml()
    p.loc[p.index[-1], "excess_long_10"] = 5.0             # extreme test outlier
    out = walk_forward(p, horizon=10, signal_info=INFO)
    assert out["target"].max() < 1.0


def test_nan_target_rows_dropped():
    p = synth_ml()
    p.loc[p.index[::7], "excess_long_10"] = np.nan
    out = walk_forward(p, horizon=10, signal_info=INFO)
    assert out["target"].notna().all()
    nan_keys = p.loc[p["excess_long_10"].isna(), ["date", "asset"]]
    assert out.merge(nan_keys, on=["date", "asset"]).empty


def test_small_fold_skipped_not_crash():
    out = walk_forward(synth_ml(300, 3), horizon=10, signal_info=INFO)
    assert out.empty and out.attrs["skipped"]
    assert list(out.columns) == ["date", "asset", "fold", "pred", "target"]


def test_fold_with_too_few_test_dates_skipped():
    # 1330 bdays end in early Feb 2015: the 2015 fold has < 5*h test dates
    out = walk_forward(synth_ml(1330, 10), horizon=10, signal_info=INFO)
    assert set(out["fold"]) == {2013, 2014}
    assert [y for y, _ in out.attrs["skipped"]] == [2015]


def test_missing_regime_value_in_training_still_gets_column():
    p = synth_ml()
    late = p["date"] >= "2015-01-01"
    p["trend_regime"] = p["trend_regime"].astype(str)
    p.loc[late & (p["trend_regime"] == "up"), "trend_regime"] = "sideways"   # only in test years
    out = walk_forward(p, horizon=10, signal_info=INFO)
    assert "trend_regime=sideways" in out.attrs["models"][2013].feature_name_


def test_fit_model_subsamples(monkeypatch):
    monkeypatch.setattr(config, "ML_MAX_TRAIN_ROWS", 1000)
    p = synth_ml(200, 10)
    X = feature_matrix(p, signal_info=INFO)
    m = fit_model(X, p["excess_long_10"].to_numpy())
    assert m.n_features_in_ == X.shape[1]
    assert list(m.feature_name_) == list(X.columns)


def test_ml_config():
    assert config.ML_PARAMS == {"objective": "regression", "num_leaves": 15, "learning_rate": 0.05,
                                "n_estimators": 200, "min_child_samples": 500, "subsample": 0.8,
                                "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
                                "random_state": 0, "verbose": -1}
    assert config.ML_MAX_TRAIN_ROWS == 1_500_000
    assert config.SHAP_ROWS == 50_000 and config.SHAP_INTERACTION_ROWS == 3_000


def test_validation_scoring():
    p = synth_ml(); p.loc[p.date >= "2014-01-01", "split"] = "validation"
    model, pred = fit_discovery_score_validation(p, 10, signal_info=INFO)
    assert (pred["fold"] == "validation").all() and pred["date"].min() >= pd.Timestamp("2014-01-01")
    assert list(pred.columns) == ["date", "asset", "fold", "pred", "target"]
    m = pred.merge(p[["date", "asset", "A"]], on=["date", "asset"])
    assert m.loc[m.A, "pred"].mean() - m.loc[~m.A, "pred"].mean() > 0.01


# ---- Task 2: out-of-sample metrics ----
from swingbot.analysis.attribution import oos_metrics  # noqa: E402

OOS_COLS = ["fold", "n", "n_dates", "ic_mean", "ic_t", "ic_p", "spread_mean", "spread_t", "spread_p",
            "top_mean", "top_ret_n", "insufficient"]


def _oos_frame(n_dates=60, n_assets=20, fold=2013, seed=0, pred_from_target=False, start="2013-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_dates)
    df = pd.MultiIndex.from_product([dates, range(n_assets)], names=["date", "asset"]).to_frame(index=False)
    df["asset"] = df["asset"].astype(str)
    df["fold"] = fold
    df["target"] = rng.normal(0, 0.02, len(df))
    df["pred"] = df["target"] if pred_from_target else rng.normal(0, 1, len(df))
    return df[["date", "asset", "fold", "pred", "target"]]


def test_ic_perfect_and_null():
    m = oos_metrics(_oos_frame(pred_from_target=True), horizon=5)
    assert list(m.columns) == OOS_COLS
    row = m.loc[m.fold == 2013].iloc[0]
    assert row.ic_mean == pytest.approx(1.0)
    null = oos_metrics(_oos_frame(n_dates=300, n_assets=40, seed=3), horizon=5)
    assert abs(null.loc[null.fold == "all", "ic_mean"].iloc[0]) < 0.05


def test_constant_pred_gives_nan_ic():
    f = _oos_frame()
    f["pred"] = 0.5
    m = oos_metrics(f, horizon=5).iloc[0]
    assert np.isnan(m.ic_mean) and np.isnan(m.ic_t) and np.isnan(m.ic_p)


def test_ic_nan_on_dates_with_fewer_than_5_rows():
    f = _oos_frame(n_assets=4, pred_from_target=True)
    assert np.isnan(oos_metrics(f, horizon=5).iloc[0].ic_mean)


def test_spread_matches_hand_example():
    d1, d2 = pd.Timestamp("2013-01-02"), pd.Timestamp("2013-01-03")
    rows = []
    for d, tg, top_asset in ((d1, np.arange(10) / 100.0, 3), (d2, np.arange(10)[::-1] / 50.0, 7)):
        for i in range(10):
            rows.append((d, str(i), 2013, 10.0 if i == top_asset else float(i % 3) - 5.0, tg[i]))
    f = pd.DataFrame(rows, columns=["date", "asset", "fold", "pred", "target"])
    m = oos_metrics(f, horizon=1).iloc[0]
    s1 = 0.03 - np.arange(10).mean() / 100.0
    s2 = (np.arange(10)[::-1] / 50.0)[7] - (np.arange(10) / 50.0).mean()
    assert m.spread_mean == pytest.approx((s1 + s2) / 2)
    assert m.top_mean == pytest.approx((0.03 + 0.04) / 2)
    assert m.top_ret_n == 2 and m.n == 20 and m.n_dates == 2
    assert m.insufficient            # 2 dates < 5*1


def test_top_decile_takes_ceil_with_ties_deterministic():
    f = _oos_frame(n_dates=10, n_assets=25)            # ceil(2.5) = 3 rows per date
    f["pred"] = 1.0
    m = oos_metrics(f, horizon=1).iloc[0]
    assert m.top_ret_n == 30
    assert m.equals(oos_metrics(f, horizon=1).iloc[0])


def test_all_row_pools_walk_forward_only():
    a = _oos_frame(fold=2013, start="2013-01-01")
    b = _oos_frame(fold=2014, start="2014-01-01", seed=1)
    v = _oos_frame(fold="validation", start="2022-01-03", seed=2)
    m = oos_metrics(pd.concat([v, b, a], ignore_index=True), horizon=5)
    assert list(m.fold) == [2013, 2014, "all", "validation"]
    assert m.loc[m.fold == "all", "n"].iloc[0] == len(a) + len(b)
    assert m.loc[m.fold == "validation", "n"].iloc[0] == len(v)
    assert not m.loc[m.fold == "all", "insufficient"].iloc[0]       # 120 dates >= 25


def test_oos_empty_input():
    m = oos_metrics(pd.DataFrame({"date": [], "asset": [], "fold": [], "pred": [], "target": []}), 5)
    assert list(m.columns) == OOS_COLS and m.empty


def test_oos_on_walk_forward_output():
    out = walk_forward(synth_ml(), horizon=10, signal_info=INFO)
    m = oos_metrics(out, horizon=10)
    assert list(m.fold) == [2013, 2014, 2015, "all"]
    assert m.loc[m.fold == "all", "ic_mean"].iloc[0] > 0
