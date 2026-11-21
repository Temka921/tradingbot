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
from swingbot.analysis.stats import hac_test
from swingbot.signals import SIGNAL_INFO

REGIME_FEATURES = ["trend_regime", "vol_regime", "market_regime", "adx_regime"]
PRED_COLS = ["date", "asset", "fold", "pred", "target"]
UNKNOWN = "unknown"
PREDICT_CHUNK = 500_000


def _info(signal_info: dict[str, tuple[str, str]] | None) -> dict[str, tuple[str, str]]:
    return SIGNAL_INFO if signal_info is None else signal_info


def _regime_values(panel: pd.DataFrame, pos: np.ndarray | None = None) -> dict[str, list[str]]:
    """Sorted non-"unknown", non-NaN values per regime column present in ``panel`` (rows ``pos`` if given)."""
    out: dict[str, list[str]] = {}
    for col in REGIME_FEATURES:
        if col not in panel.columns:
            continue
        ser = panel[col] if pos is None else panel[col].iloc[pos]
        vals = pd.unique(ser.dropna().astype(str))
        out[col] = sorted(v for v in vals if v != UNKNOWN)
    return out


def _features(panel: pd.DataFrame, info: dict, regime_values: dict[str, list[str]],
              pos: np.ndarray | None = None) -> pd.DataFrame:
    """float32 feature frame with a fixed column set (signals, then regime one-hots).

    With ``pos`` only those integer row positions are densified, one column at a
    time, so a full-width copy of the panel is never made.
    """
    missing = [s for s in info if s not in panel.columns]
    if missing:
        raise KeyError(f"signal columns missing from panel: {missing}")
    n = len(panel) if pos is None else len(pos)
    cols: dict[str, np.ndarray] = {}
    for name in info:
        arr = panel[name].to_numpy(dtype=bool)
        cols[name] = (arr if pos is None else arr[pos]).astype("float32")
    for col, values in regime_values.items():
        labels = None
        if col in panel.columns:
            ser = panel[col] if pos is None else panel[col].iloc[pos]
            labels = ser.astype(object).to_numpy()
        for v in values:
            if labels is None:
                cols[f"{col}={v}"] = np.zeros(n, dtype="float32")
            else:
                cols[f"{col}={v}"] = (labels == v).astype("float32")
    index = panel.index if pos is None else panel.index[pos]
    return pd.DataFrame(cols, index=index)


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


def _fit_and_score(panel: pd.DataFrame, train_pos: np.ndarray, test_pos: np.ndarray, target: str,
                   info: dict, regime_values: dict[str, list[str]]
                   ) -> tuple[lgb.LGBMRegressor, np.ndarray, np.ndarray]:
    """Fit on panel rows ``train_pos`` (clip bounds from all of them; features built on the fit sample only).

    Works on integer positions and only the needed columns, so no full-width
    copy of the train or test rows is made.
    """
    y_all = panel[target].to_numpy(dtype="float64")
    y_tr_raw = y_all[train_pos]
    bounds = clip_bounds(y_tr_raw)
    spos = _sample_positions(len(train_pos))        # sample before densifying features
    if spos is not None:
        train_pos, y_tr_raw = train_pos[spos], y_tr_raw[spos]
    model = fit_model(_features(panel, info, regime_values, train_pos), clip_values(y_tr_raw, bounds))
    y_te = clip_values(y_all[test_pos], bounds)
    pred = _predict(model, _features(panel, info, regime_values, test_pos))
    return model, pred, y_te


def _frame(panel: pd.DataFrame, test_pos: np.ndarray, fold, pred: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"date": panel["date"].to_numpy()[test_pos], "asset": panel["asset"].to_numpy()[test_pos],
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
    dates = panel["date"].to_numpy()
    in_split = np.flatnonzero((panel["split"] == split).to_numpy())
    regime_values = _regime_values(panel, in_split)          # same one-hot set for every fold
    all_dates = np.sort(pd.unique(dates[in_split]))
    sp = in_split[panel[target].to_numpy()[in_split] == panel[target].to_numpy()[in_split]]   # drop NaN target
    sp_dates = dates[sp]
    years = pd.DatetimeIndex(sp_dates).year.to_numpy()
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
        test_pos = sp[years == year]
        test_d = sp_dates[years == year]
        n_test_dates = len(pd.unique(test_d))
        if n_test_dates < 5 * horizon:
            skipped.append((year, f"{n_test_dates} test dates < {5 * horizon}"))
            continue
        cutoff = _embargo_cutoff(all_dates, pd.Timestamp(test_d.min()), horizon)
        train_pos = sp[:0] if cutoff is None else sp[sp_dates < np.datetime64(cutoff)]
        if len(train_pos) < config.MIN_N:
            skipped.append((year, f"{len(train_pos)} training rows < {config.MIN_N}"))
            continue
        model, pred, y_te = _fit_and_score(panel, train_pos, test_pos, target, info, regime_values)
        models[year] = model
        train_dates[year] = np.sort(pd.unique(dates[train_pos]))
        frames.append(_frame(panel, test_pos, year, pred, y_te))
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
    dates = panel["date"].to_numpy()
    y = panel[target].to_numpy(dtype="float64")
    both = np.flatnonzero(np.isin(split, ["discovery", "validation"]))
    regime_values = _regime_values(panel, both)
    all_dates = np.sort(pd.unique(dates[both]))
    ok = ~np.isnan(y)
    disc_pos = np.flatnonzero((split == "discovery") & ok)
    val_pos = np.flatnonzero((split == "validation") & ok)
    if not len(val_pos):
        return None, _empty({"skipped": [("validation", "no validation rows")]})
    cutoff = _embargo_cutoff(all_dates, pd.Timestamp(dates[val_pos].min()), horizon)
    train_pos = disc_pos[:0] if cutoff is None else disc_pos[dates[disc_pos] < np.datetime64(cutoff)]
    if len(train_pos) < config.MIN_N:
        return None, _empty({"skipped": [("validation", f"{len(train_pos)} training rows < {config.MIN_N}")]})
    model, pred, y_val = _fit_and_score(panel, train_pos, val_pos, target, info, regime_values)
    out = _frame(panel, val_pos, "validation", pred, y_val)
    out.attrs.update({"skipped": [], "train_dates": np.sort(pd.unique(dates[train_pos]))})
    return model, out


# ---- out-of-sample metrics ----
OOS_COLS = ["fold", "n", "n_dates", "ic_mean", "ic_t", "ic_p", "spread_mean", "spread_t", "spread_p",
            "top_mean", "top_ret_n", "insufficient"]
IC_MIN_ROWS = 5
TOP_FRAC = 0.1


def _per_date(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date rank IC, top-decile spread and top-decile mean target (vectorised groupby, no date loop)."""
    pred = df["pred"].to_numpy(dtype="float64")
    tgt = df["target"].to_numpy(dtype="float64")
    g = pd.DataFrame({"date": df["date"].to_numpy(), "p": pred, "t": tgt})
    by = g.groupby("date", sort=True)
    n = by["p"].transform("size").to_numpy()
    # Spearman = Pearson of average ranks within date
    rp = by["p"].rank(method="average").to_numpy()
    rt = by["t"].rank(method="average").to_numpy()
    g["rp"], g["rt"] = rp, rt
    mp = g.groupby("date", sort=True)["rp"].transform("mean").to_numpy()
    mt = g.groupby("date", sort=True)["rt"].transform("mean").to_numpy()
    dp, dt_ = rp - mp, rt - mt
    g["pp"], g["tt"], g["pt"] = dp * dp, dt_ * dt_, dp * dt_
    s = g.groupby("date", sort=True)[["pp", "tt", "pt"]].sum()
    size = by.size()
    pmax, pmin = by["p"].max(), by["p"].min()
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = s["pt"] / np.sqrt(s["pp"] * s["tt"])
    bad = (size < IC_MIN_ROWS) | (pmax == pmin) | ~np.isfinite(ic)
    ic = ic.where(~bad)
    # top 10% of pred (>= 1 row), ties broken by order of appearance
    k = np.maximum(np.ceil(TOP_FRAC * n), 1)
    top = by["p"].rank(method="first", ascending=False).to_numpy() <= k
    g["top_t"] = np.where(top, tgt, np.nan)
    top_mean = g.groupby("date", sort=True)["top_t"].mean()
    top_n = g.groupby("date", sort=True)["top_t"].count()
    all_mean = by["t"].mean()
    return pd.DataFrame({"ic": ic, "spread": top_mean - all_mean, "top_mean": top_mean, "top_n": top_n})


def _oos_row(fold, df: pd.DataFrame, horizon: int) -> dict:
    d = _per_date(df)
    n_dates = len(d)
    ic = hac_test(d["ic"].dropna().sort_index(), horizon, len(df))
    sp = hac_test(d["spread"].dropna().sort_index(), horizon, len(df))
    return {"fold": fold, "n": int(len(df)), "n_dates": int(n_dates),
            "ic_mean": ic.mean if ic.n_dates else np.nan, "ic_t": ic.t, "ic_p": ic.p,
            "spread_mean": sp.mean if sp.n_dates else np.nan, "spread_t": sp.t, "spread_p": sp.p,
            "top_mean": float(d["top_mean"].mean()) if n_dates else np.nan,
            "top_ret_n": int(d["top_n"].sum()), "insufficient": bool(n_dates < 5 * horizon)}


def oos_metrics(pred: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Out-of-sample rank IC and top-decile spread per fold, plus an ``"all"`` row.

    ``"all"`` pools every fold except ``"validation"`` (never pooled with walk-forward).
    Rows: folds sorted by year, ``"all"``, then ``"validation"`` if present. No q-values.
    """
    if pred.empty:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in OOS_COLS})
    is_val = pred["fold"].astype(str).to_numpy() == "validation"
    wf = pred.loc[~is_val]
    rows = []
    for fold in sorted(pd.unique(wf["fold"])):
        rows.append(_oos_row(fold, wf.loc[wf["fold"] == fold], horizon))
    if len(wf):
        rows.append(_oos_row("all", wf, horizon))
    if is_val.any():
        rows.append(_oos_row("validation", pred.loc[is_val], horizon))
    out = pd.DataFrame(rows, columns=OOS_COLS)
    out["fold"] = out["fold"].astype(object)
    return out


# ---- SHAP attribution ----
SHAP_SUMMARY_COLS = ["feature", "kind", "mean_abs_shap", "mean_shap_when_on", "n_on", "rank"]
SHAP_PAIR_COLS = ["a", "b", "mean_abs_interaction", "mean_interaction_when_both", "n_both"]


def _shap_sample(X: pd.DataFrame, rows: int) -> pd.DataFrame:
    """Random rows (numpy default_rng(0), no replacement, original order kept) when ``len(X) > rows``."""
    if len(X) <= rows:
        return X
    pos = np.sort(np.random.default_rng(0).choice(len(X), size=rows, replace=False))
    return X.iloc[pos]


def _explainer(model):
    import shap
    # no background dataset: tree_path_dependent, fast and deterministic
    return shap.TreeExplainer(model)


def shap_summary(model, X: pd.DataFrame, signal_info=None) -> pd.DataFrame:
    """Per-feature SHAP importance on a ``config.SHAP_ROWS`` sample of ``X`` (seed 0).

    Columns: feature, kind ("signal" if in ``signal_info`` else "regime"),
    mean_abs_shap, mean_shap_when_on (mean SHAP over rows where the feature is 1;
    NaN when ``n_on == 0``), n_on (within the sample), rank (1 = largest mean_abs_shap).
    """
    info = _info(signal_info)
    Xs = _shap_sample(X, config.SHAP_ROWS)
    phi = np.asarray(_explainer(model).shap_values(Xs), dtype="float64")
    on = Xs.to_numpy() == 1
    n_on = on.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        when_on = np.where(n_on > 0, (phi * on).sum(axis=0) / np.maximum(n_on, 1), np.nan)
    out = pd.DataFrame({"feature": list(X.columns),
                        "kind": ["signal" if c in info else "regime" for c in X.columns],
                        "mean_abs_shap": np.abs(phi).mean(axis=0) if len(Xs) else np.zeros(X.shape[1]),
                        "mean_shap_when_on": when_on, "n_on": n_on.astype(int)})
    out = out.sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[SHAP_SUMMARY_COLS]


def shap_pair_interactions(model, X: pd.DataFrame, signal_info=None, top: int = 30) -> pd.DataFrame:
    """Top same-direction signal x signal SHAP interactions on a ``config.SHAP_INTERACTION_ROWS`` sample.

    Pairs (a, b) follow ``signal_info`` order (a before b) and share a direction.
    ``mean_abs_interaction`` and ``mean_interaction_when_both`` use the raw
    off-diagonal entry ``phi[:, i, j]`` as returned by shap, which is one symmetric
    half of the pair's interaction (phi[i, j] == phi[j, i]); it is NOT doubled.
    ``mean_interaction_when_both`` is NaN when ``n_both == 0`` (rows where both are 1).
    Sorted by ``mean_abs_interaction`` descending, cut to ``top``.
    """
    info = _info(signal_info)
    sigs = [c for c in X.columns if c in info]
    pairs = [(a, b) for i, a in enumerate(sigs) for b in sigs[i + 1:] if info[a][1] == info[b][1]]
    Xs = _shap_sample(X, config.SHAP_INTERACTION_ROWS)
    if not pairs:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in SHAP_PAIR_COLS})
    phi = np.asarray(_explainer(model).shap_interaction_values(Xs), dtype="float64")
    idx = {c: i for i, c in enumerate(X.columns)}
    arr = Xs.to_numpy() == 1
    rows = []
    for a, b in pairs:
        v = phi[:, idx[a], idx[b]]
        both = arr[:, idx[a]] & arr[:, idx[b]]
        n_both = int(both.sum())
        rows.append((a, b, float(np.abs(v).mean()) if len(v) else 0.0,
                     float(v[both].mean()) if n_both else np.nan, n_both))
    out = pd.DataFrame(rows, columns=SHAP_PAIR_COLS)
    out = out.sort_values("mean_abs_interaction", ascending=False, kind="stable")
    return out.head(top).reset_index(drop=True)
