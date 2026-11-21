"""ML attribution: an embargoed walk-forward LightGBM on signal and regime features.

Features at row t are the boolean signal columns and one-hot regime labels,
all known at the close of t. The target is the clipped ``excess_long_{h}``
(forward-looking by design). Out-of-sample predictions come only from
walk-forward folds (one per calendar year of the split); the validation split
is scored only by a model fitted on all discovery rows.

Embargo: a fold's training rows have ``date`` at least ``h + 1`` distinct
panel dates before its first test date, so no training target overlaps the
test period.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from swingbot import config
from swingbot.analysis.solo import clip_bounds, clip_values
from swingbot.signals import SIGNAL_INFO

REGIME_FEATURES = ["trend_regime", "vol_regime", "market_regime", "adx_regime"]
PRED_COLS = ["date", "asset", "fold", "pred", "target"]
UNKNOWN = "unknown"
PREDICT_CHUNK = 500_000


def _info(signal_info: dict[str, tuple[str, str]] | None) -> dict[str, tuple[str, str]]:
    return SIGNAL_INFO if signal_info is None else signal_info


def _regime_values(panel: pd.DataFrame) -> dict[str, list[str]]:
    """Sorted non-"unknown", non-NaN values per regime column present in ``panel``."""
    out: dict[str, list[str]] = {}
    for col in REGIME_FEATURES:
        if col not in panel.columns:
            continue
        vals = pd.unique(panel[col].dropna().astype(str))
        out[col] = sorted(v for v in vals if v != UNKNOWN)
    return out


def _features(panel: pd.DataFrame, info: dict, regime_values: dict[str, list[str]]) -> pd.DataFrame:
    """float32 feature frame with a fixed column set (signals, then regime one-hots)."""
    missing = [s for s in info if s not in panel.columns]
    if missing:
        raise KeyError(f"signal columns missing from panel: {missing}")
    cols: dict[str, np.ndarray] = {}
    for name in info:
        cols[name] = panel[name].to_numpy(dtype=bool).astype("float32")
    for col, values in regime_values.items():
        labels = panel[col].astype(object).to_numpy() if col in panel.columns else None
        for v in values:
            if labels is None:
                cols[f"{col}={v}"] = np.zeros(len(panel), dtype="float32")
            else:
                cols[f"{col}={v}"] = (labels == v).astype("float32")
    return pd.DataFrame(cols, index=panel.index)


def feature_matrix(panel: pd.DataFrame, signal_info=None) -> pd.DataFrame:
    """Signals as 0/1 then ``f"{regime}={value}"`` one-hots (no "unknown"); float32, same index."""
    return _features(panel, _info(signal_info), _regime_values(panel))


def _sample_positions(n: int) -> np.ndarray | None:
    """Sorted random positions (seed 0) when ``n`` exceeds ``ML_MAX_TRAIN_ROWS``, else None."""
    if n <= config.ML_MAX_TRAIN_ROWS:
        return None
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(n, size=config.ML_MAX_TRAIN_ROWS, replace=False))


def fit_model(X: pd.DataFrame, y: np.ndarray) -> lgb.LGBMRegressor:
    """LightGBM with ``config.ML_PARAMS`` on at most ``ML_MAX_TRAIN_ROWS`` random rows (seed 0)."""
    y = np.asarray(y, dtype="float64")
    pos = _sample_positions(len(X))
    if pos is not None:
        X, y = X.iloc[pos], y[pos]
    model = lgb.LGBMRegressor(**config.ML_PARAMS)
    model.fit(X, y)
    return model


def _predict(model: lgb.LGBMRegressor, X: pd.DataFrame) -> np.ndarray:
    if not len(X):
        return np.empty(0, dtype="float64")
    parts = [model.predict(X.iloc[i:i + PREDICT_CHUNK]) for i in range(0, len(X), PREDICT_CHUNK)]
    return np.concatenate(parts)


def _empty(attrs: dict) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "asset": pd.Series(dtype=object),
                        "fold": pd.Series(dtype=object), "pred": pd.Series(dtype="float64"),
                        "target": pd.Series(dtype="float64")})
    out.attrs.update(attrs)
    return out


def _embargo_cutoff(dates: np.ndarray, first_test: pd.Timestamp, horizon: int) -> pd.Timestamp | None:
    """Training dates must be < the returned date (the (h+1)-th distinct date before ``first_test``)."""
    before = dates[dates < np.datetime64(first_test)]
    if len(before) <= horizon + 1:
        return None
    return pd.Timestamp(before[-(horizon + 1)])


def _fit_and_score(train: pd.DataFrame, test: pd.DataFrame, target: str, info: dict,
                   regime_values: dict[str, list[str]]) -> tuple[lgb.LGBMRegressor, np.ndarray, np.ndarray]:
    """Fit on ``train`` (clip bounds from all its rows; features built on the fit sample only)."""
    y_tr_raw = train[target].to_numpy(dtype="float64")
    bounds = clip_bounds(y_tr_raw)
    pos = _sample_positions(len(train))        # sample before densifying features
    if pos is not None:
        train, y_tr_raw = train.iloc[pos], y_tr_raw[pos]
    model = fit_model(_features(train, info, regime_values), clip_values(y_tr_raw, bounds))
    y_te = clip_values(test[target].to_numpy(dtype="float64"), bounds)
    pred = _predict(model, _features(test, info, regime_values))
    return model, pred, y_te


def _frame(test: pd.DataFrame, fold, pred: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"date": test["date"].to_numpy(), "asset": test["asset"].to_numpy(),
                         "fold": fold, "pred": pred, "target": y})


def walk_forward(panel: pd.DataFrame, horizon: int = 10, split: str = "discovery",
                 signal_info=None, min_train_years: int = 3) -> pd.DataFrame:
    """Out-of-sample predictions, one fold per calendar year after the first ``min_train_years``.

    Fold Y trains on split rows dated before ``Y-01-01`` minus the last ``horizon + 1``
    distinct split dates before the first test date, and tests on the split rows of
    year Y. Rows with NaN target are dropped from both. The target is
    ``excess_long_{horizon}`` clipped at bounds from the training rows. A fold with
    fewer than ``config.MIN_N`` training rows or fewer than ``5 * horizon`` test dates
    is skipped. attrs: ``skipped`` [(year, reason)] (year None when the split has no
    fold year at all), ``models`` {year: model}, ``train_dates`` {year: dates}.
    """
    info = _info(signal_info)
    target = f"excess_long_{horizon}"
    sp = panel.loc[(panel["split"] == split).to_numpy()]
    regime_values = _regime_values(sp)          # same one-hot set for every fold
    all_dates = np.sort(sp["date"].unique())
    sp = sp.loc[sp[target].notna().to_numpy()]
    years = sp["date"].dt.year.to_numpy()
    uniq = np.unique(years)
    skipped: list[tuple] = []
    models: dict[int, lgb.LGBMRegressor] = {}
    train_dates: dict[int, np.ndarray] = {}
    frames: list[pd.DataFrame] = []
    if len(uniq) <= min_train_years:
        skipped.append((None, f"split has {len(uniq)} calendar year(s); need more than "
                              f"min_train_years={min_train_years}"))
    for year in uniq[min_train_years:]:
        year = int(year)
        test = sp.loc[years == year]
        n_test_dates = test["date"].nunique()
        if n_test_dates < 5 * horizon:
            skipped.append((year, f"{n_test_dates} test dates < {5 * horizon}"))
            continue
        cutoff = _embargo_cutoff(all_dates, test["date"].min(), horizon)
        train = sp.iloc[:0] if cutoff is None else sp.loc[(sp["date"] < cutoff).to_numpy()]
        if len(train) < config.MIN_N:
            skipped.append((year, f"{len(train)} training rows < {config.MIN_N}"))
            continue
        model, pred, y_te = _fit_and_score(train, test, target, info, regime_values)
        models[year] = model
        train_dates[year] = np.sort(train["date"].unique())
        frames.append(_frame(test, year, pred, y_te))
    attrs = {"skipped": skipped, "models": models, "train_dates": train_dates}
    if not frames:
        return _empty(attrs)
    out = pd.concat(frames, ignore_index=True)
    out.attrs.update(attrs)
    return out


def fit_discovery_score_validation(panel: pd.DataFrame, horizon: int = 10, signal_info=None
                                   ) -> tuple[lgb.LGBMRegressor | None, pd.DataFrame]:
    """Fit on all discovery rows (embargoed before the first validation date), score validation.

    Returns the model and predictions in ``PRED_COLS`` with ``fold="validation"``.
    The same embargo as ``walk_forward`` applies: discovery dates within the last
    ``horizon + 1`` distinct dates before the first validation date are dropped.
    Returns ``(None, empty)`` with ``attrs["skipped"]`` when there are no validation
    rows or fewer than ``config.MIN_N`` training rows.
    """
    info = _info(signal_info)
    target = f"excess_long_{horizon}"
    split = panel["split"].astype(str).to_numpy()
    both = panel.loc[np.isin(split, ["discovery", "validation"])]
    regime_values = _regime_values(both)
    disc = panel.loc[split == "discovery"]
    val = panel.loc[split == "validation"]
    all_dates = np.sort(both["date"].unique())
    disc = disc.loc[disc[target].notna().to_numpy()]
    val = val.loc[val[target].notna().to_numpy()]
    if val.empty:
        return None, _empty({"skipped": [("validation", "no validation rows")]})
    cutoff = _embargo_cutoff(all_dates, val["date"].min(), horizon)
    train = disc.iloc[:0] if cutoff is None else disc.loc[(disc["date"] < cutoff).to_numpy()]
    if len(train) < config.MIN_N:
        return None, _empty({"skipped": [("validation", f"{len(train)} training rows < {config.MIN_N}")]})
    model, pred, y_val = _fit_and_score(train, val, target, info, regime_values)
    out = _frame(val, "validation", pred, y_val)
    out.attrs.update({"skipped": [], "train_dates": np.sort(train["date"].unique())})
    return model, out
