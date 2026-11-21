# Phase 5: Model Attribution (LightGBM + SHAP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit a walk-forward gradient-boosting model on the signal matrix plus regime labels for each view (etf, crypto, stock). Measure honestly whether it predicts out of sample. Then use SHAP to attribute what drives it: per signal, and per signal pair through SHAP interaction values.

**Architecture:** One new module, `analysis/attribution.py`, owns the feature matrix, walk-forward training, out-of-sample metrics and SHAP summaries. `report/ml.py` and `templates/ml.html.j2` render a third report; `scripts/attribute.py` runs it. The module consumes the views from `analysis/views.py`, and the Phase 4 pair table is used only for the comparison column.

**Tech Stack:** Python 3.15, pandas, numpy, lightgbm, shap, statsmodels (`stats.hac_test`), plotly (CDN), jinja2, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-swing-research-bot-design.md` (§5 "Model attribution", plus "Decisions after Phase 1"). Obey `CLAUDE.md`.

## Global Constraints

- No lookahead. Features at row t are the signal and regime columns, which are known at the close of t. The target is `excess_long_{h}`, which is forward-looking by design.
- Walk-forward is the only way any reported out-of-sample number may be produced. A model never sees a test row's date, or any date within the embargo, during training.
- Embargo: training rows must have `date` at least `h + 1` distinct panel dates before the first test date.
- The holdout stays locked; the 2024+ data is never loaded.
- Discovery split for walk-forward. The validation split is used only to score a model fitted on all discovery rows.
- Fixed hyperparameters, no tuning. They are pinned in `config.ML_PARAMS`:
  `{"objective": "regression", "num_leaves": 15, "learning_rate": 0.05, "n_estimators": 200, "min_child_samples": 500, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0, "random_state": 0, "verbose": -1}`.
- `config.ML_MAX_TRAIN_ROWS = 1_500_000`: a random row sample (seed 0) per fit when a training set is larger.
- `config.SHAP_ROWS = 50_000` and `config.SHAP_INTERACTION_ROWS = 3_000`: random samples, seed 0.
- The target is clipped with the solo rule (pooled 0.1/99.9 percentiles over the training rows), using `solo.clip_bounds` and `solo.clip_values`.
- Tests never hit the network; the suite must stay under 2 minutes.
- If lightgbm or shap cannot be installed on Python 3.15, stop and report BLOCKED. CLAUDE.md allows a 3.12 venv via `uv`, but the controller decides that.

## Review Focus

1. **Lookahead through the embargo:** no training row may fall inside the last `h+1` dates before a test fold. Test this with a panel where the target leaks into the next fold.
2. **A view or fold with too little data** (crypto 2018 has about 12 coins per day): the fold is skipped with a reason. It must not crash, and must not report an IC computed from fewer than `5*h` dates.
3. **A signal that never fires in a view:** its SHAP rows are 0 or NaN, and nothing raises.
4. **A constant prediction** (a model that learned nothing): the per-date rank IC is NaN for those dates, not 0 and not an error.
5. **Memory on the stock view** (about 4.4M discovery rows): fit on a sample, and predict in chunks if needed. Never densify a copy of the whole panel per fold.

---

### Task 1: Dependencies, feature matrix and walk-forward predictions

**Files:**
- Modify: `pyproject.toml` (add `lightgbm`, `shap`), `src/swingbot/config.py`
- Create: `src/swingbot/analysis/attribution.py`
- Test: `tests/test_attribution.py`

**Interfaces:**
- Produces:
  - `REGIME_FEATURES = ["trend_regime", "vol_regime", "market_regime", "adx_regime"]`
  - `feature_matrix(panel: pd.DataFrame, signal_info=None) -> pd.DataFrame`: float32 columns. Each signal in `SIGNAL_INFO` becomes a 0/1 column. Each regime label in `REGIME_FEATURES` becomes one-hot columns named `f"{col}={value}"` for every value except `"unknown"`. A regime column that is missing from the panel is skipped. Row order and index are the same as `panel`.
  - `walk_forward(panel, horizon=10, split="discovery", signal_info=None, min_train_years=3) -> pd.DataFrame`:
    - Columns: `date, asset, fold, pred, target`.
    - One fold per calendar year Y of the split, starting after the first `min_train_years` years.
    - Training rows are split rows with date before `Y-01-01`, excluding the last `horizon+1` distinct dates before the first test date.
    - Test rows are split rows in year Y.
    - Rows with NaN target are dropped from both training and test.
    - `target` is the clipped `excess_long_{h}`: bounds from training rows, applied to both training and test.
    - A fold with fewer than `config.MIN_N` training rows, or fewer than `5*horizon` test dates, is skipped. Its reason goes into `out.attrs["skipped"]`, a list of `(year, reason)`.
    - `out.attrs["models"]` maps year to the fitted `LGBMRegressor`.
  - `fit_model(X: pd.DataFrame, y: np.ndarray) -> lightgbm.LGBMRegressor`: uses `config.ML_PARAMS`, subsampled to `ML_MAX_TRAIN_ROWS`.
  - `fit_discovery_score_validation(panel, horizon=10, signal_info=None) -> tuple[LGBMRegressor, pd.DataFrame]`: fits on all discovery rows and returns the model plus predictions on validation rows, in the same columns as `walk_forward` with `fold="validation"`.
  - config: `ML_PARAMS`, `ML_MAX_TRAIN_ROWS`, `SHAP_ROWS`, `SHAP_INTERACTION_ROWS`.

- [ ] **Step 1: Write the failing tests**

```python
def synth_ml(n_dates=1500, n_assets=30, plant=0.03, seed=0, leak=False):
    # business dates from 2010; signals A (long), B (long), S (short) ~ Bernoulli(0.2);
    # excess_long_10 ~ N(0, 0.02) + plant on rows where A fires; trend_regime random in {up, down, unknown};
    # split = "discovery"; leak=True instead plants on rows whose date is in the first 11 dates of each year
    ...

def test_feature_matrix_columns():
    p = synth_ml(200, 5)
    X = feature_matrix(p, signal_info=INFO)
    assert list(X.columns[:3]) == ["A", "B", "S"]
    assert "trend_regime=up" in X and "trend_regime=unknown" not in X
    assert X.dtypes.eq("float32").all() and len(X) == len(p)

def test_walk_forward_learns_planted_signal():
    out = walk_forward(synth_ml(), horizon=10, signal_info=INFO)
    assert set(out["fold"]) == {2013, 2014, 2015}        # 1500 bdays ~ 2010-2015, 3 training years first
    p = synth_ml()[["date", "asset", "A"]]
    m = out.merge(p, on=["date", "asset"])
    assert m.loc[m.A == 1, "pred"].mean() - m.loc[m.A == 0, "pred"].mean() > 0.01

def test_embargo_excludes_last_h_plus_1_dates():
    p = synth_ml(); out = walk_forward(p, horizon=10, signal_info=INFO)
    tr = out.attrs["train_dates"][2014]                    # dates used to train the 2014 model
    first_test = out.loc[out.fold == 2014, "date"].min()
    all_dates = np.sort(p["date"].unique())
    cutoff = all_dates[np.searchsorted(all_dates, first_test) - 11]
    assert tr.max() < cutoff

def test_small_fold_skipped_not_crash():
    out = walk_forward(synth_ml(300, 3), horizon=10, signal_info=INFO)
    assert out.empty and out.attrs["skipped"]

def test_validation_scoring():
    p = synth_ml(); p.loc[p.date >= "2014-01-01", "split"] = "validation"
    model, pred = fit_discovery_score_validation(p, 10, signal_info=INFO)
    assert (pred["fold"] == "validation").all() and pred["date"].min() >= pd.Timestamp("2014-01-01")
```

Also store `out.attrs["train_dates"]` (year → numpy array of the training dates) so the embargo is testable.

- [ ] **Step 2:** `.venv\Scripts\python -m pip install lightgbm shap`. Run the tests; expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `pytest -q`; expect PASS.
- [ ] **Step 5: Commit** with message `feat(ml): feature matrix and embargoed walk-forward LightGBM`.

---

### Task 2: Out-of-sample metrics

**Files:** Modify `src/swingbot/analysis/attribution.py`; test `tests/test_attribution.py`.

**Interfaces:**
- Consumes: `walk_forward` / `fit_discovery_score_validation` output and `stats.hac_test`.
- Produces: `oos_metrics(pred: pd.DataFrame, horizon: int) -> pd.DataFrame`, with one row per fold plus a row `fold="all"` that covers all walk-forward folds (validation is never pooled with walk-forward folds).
  - Columns: `fold, n, n_dates, ic_mean, ic_t, ic_p, spread_mean, spread_t, spread_p, top_mean, top_ret_n, insufficient`.
  - `ic`: per-date Spearman rank correlation between `pred` and `target`. NaN on dates with fewer than 5 rows or a constant `pred`. The per-date series goes to `hac_test`.
  - `spread`: per date, mean target of rows in the top 10% of `pred` (at least 1 row) minus the mean target of all rows that date. Tested with `hac_test`.
  - `top_mean`: date-weighted mean target of the top-10% rows.
  - `insufficient` = `n_dates < 5*h`.
  - No q-values. Folds are few, and the "all" row is the decision row.

- [ ] **Step 1: Write the failing tests:**
  - `test_ic_perfect_and_null`: `pred == target` gives IC 1.0 on every date; `pred` independent of the target gives `|ic_mean| < 0.05`.
  - `test_constant_pred_gives_nan_ic`.
  - `test_spread_matches_hand_example`: 2 dates × 10 rows, hand-computed.
  - `test_all_row_pools_walk_forward_only`: a frame with folds 2013, 2014 and "validation" gives an "all" row whose n is the sum of 2013 and 2014 only.
- [ ] **Step 2:** Run them; expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `pytest -q`; expect PASS.
- [ ] **Step 5: Commit** with message `feat(ml): out-of-sample rank IC and top-decile spread with HAC tests`.

---

### Task 3: SHAP attribution

**Files:** Modify `src/swingbot/analysis/attribution.py`; test `tests/test_attribution.py`.

**Interfaces:**
- Produces:
  - `shap_summary(model, X: pd.DataFrame, signal_info=None) -> pd.DataFrame`, with one row per feature.
    - Columns: `feature, kind ("signal"|"regime"), mean_abs_shap, mean_shap_when_on, n_on, rank`.
    - It uses `shap.TreeExplainer(model).shap_values` on a `config.SHAP_ROWS` sample (seed 0).
    - `mean_shap_when_on` is the mean SHAP over rows where the feature is 1, NaN when `n_on == 0`.
    - Sorted by `mean_abs_shap` descending; `rank` starts at 1.
  - `shap_pair_interactions(model, X, signal_info=None, top=30) -> pd.DataFrame`, with columns `a, b, mean_abs_interaction, mean_interaction_when_both, n_both`.
    - It uses `TreeExplainer.shap_interaction_values` on a `config.SHAP_INTERACTION_ROWS` sample.
    - Signal × signal off-diagonal pairs only, same direction only (reusing `SIGNAL_INFO` directions as Phase 4 does), with a < b in registry order.
    - Sorted by `mean_abs_interaction` descending, then cut to the top `top`.
    - `mean_interaction_when_both` is NaN when `n_both == 0`.

- [ ] **Step 1: Write the failing tests:**
  - `test_shap_ranks_planted_signal_first`: `synth_ml` with `plant=0.03` on A, fit with `fit_model` on all rows; `shap_summary` ranks A first, with `mean_shap_when_on > 0`.
  - `test_shap_interaction_finds_planted_pair`: plant +0.04 only on A∧B rows (A and B alone add nothing); the top row is (A, B) with `mean_interaction_when_both > 0`.
  - `test_never_firing_signal_no_crash`: add a column `Z` that is all 0 to `signal_info`; Z's `mean_shap_when_on` is NaN and `n_on == 0`.
  - Monkeypatch `config.SHAP_ROWS` and `SHAP_INTERACTION_ROWS` small in the tests to keep them fast.
- [ ] **Step 2:** Run them; expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `pytest -q`; expect PASS.
- [ ] **Step 5: Commit** with message `feat(ml): SHAP importance and pair interactions`.

---

### Task 4: ML report and `scripts/attribute.py`

**Files:**
- Create: `src/swingbot/report/ml.py`, `src/swingbot/report/templates/ml.html.j2`, `scripts/attribute.py`
- Test: `tests/test_ml_report.py`

**Interfaces:**
- Consumes: everything above; `universe_views`, `SUSPECT_LABEL`, `CDN_URL`, `stock_mean_text`, `_fmt`/`_bps` from `report/build.py`; `pair_table` from `analysis/pairs.py`.
- Produces: `build_ml_report(panels, out_path, horizon=10, as_of=None) -> Path`. For each view:
  1. **Out-of-sample table:** the `oos_metrics` rows for the walk-forward folds plus "all", plus the validation row from `fit_discovery_score_validation` (shown separately), and the skipped folds with reasons.
  2. **Comparison line:** the "all" row's `spread_mean` against the best solo candidate's edge in the same view (from `solo_table`, verdict candidate, largest edge; "none" if no candidate). This is the fair "ML vs rules" comparison the spec asks for.
  3. **SHAP importance:** bar chart (plotly) plus table, from the model fitted on all discovery rows.
  4. **Top 30 SHAP pair interactions:** each row joined with the Phase 4 pair verdict and lift (`pair_table` on the same view) in extra columns `phase4_verdict, phase4_lift`.
  5. **Caveat box:**
     - the survivorship text (reused);
     - "SHAP explains the model, not the market; a feature the model leans on is not proof of an edge";
     - "the validation row is the only out-of-sample score outside discovery";
     - fixed hyperparameters, no tuning;
     - the stock view is labelled suspect.

  CSVs: `ml_{oos|shap|interactions}_{view}_{stamp}.csv`.
- `scripts/attribute.py --horizon 10`: loads the panels like `analyze.py`, builds `reports/ml_report_{stamp}.html` and prints the path.

- [ ] **Step 1: Write the failing test:** `test_ml_report_renders` uses tiny synthetic panels: a stock panel with 2 ETFs and 2 stocks, a crypto panel, a few hundred dates from 2010, and `excess_*`/`ret_*` columns. It asserts:
  - the CSVs exist for each view that has enough data;
  - `SUSPECT_LABEL`, `CDN_URL` and the "SHAP explains the model" text appear in the HTML;
  - a view whose folds were all skipped renders its skip reasons instead of crashing.

  Keep the test under about 20 s: monkeypatch `config.ML_PARAMS["n_estimators"]` to 20 and the SHAP rows small.
- [ ] **Step 2:** Run it; expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `pytest -q`; expect PASS.
- [ ] **Step 5: Commit** with message `feat(report): ML attribution report and attribute.py`.

---

### Task 5: Real run (controller)

- [ ] Run `python scripts/attribute.py --horizon 10` in the background. Record the runtime and the per-view "all" and validation rows.
- [ ] Do a sanity check: on the stock view, SHAP should rank the Phase 1 dip-buying signals high, if the model found what the solo analysis found.
- [ ] Final whole-branch review of the Phase 5 range.
