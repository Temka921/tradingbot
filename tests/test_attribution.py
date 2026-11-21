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
