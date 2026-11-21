# Phase 6: Validation Gate, Portfolio Backtest, Holdout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn discovery hypotheses into finalists with a formal gate on the validation split. Then backtest them as a simple portfolio against buy-and-hold, check robustness, and run the one-time 2024+ holdout only for finalists from the primary views.

**Architecture:**
- `analysis/validation.py` re-tests the discovery candidates (solo) and synergy pairs on the validation split, and marks finalists.
- `backtest/portfolio.py` simulates a capacity-limited, fixed-horizon portfolio from panel outcomes. No new price layer is needed, because the outcomes are already net of costs.
- `report/finalists.py` renders the gate, the backtests and the robustness checks.
- `scripts/holdout.py` is the only code path that ever loads 2024+ data. It refuses to run without primary-view finalists, and refuses to run twice.

**Tech Stack:** Python 3.15, pandas, numpy, plotly (CDN), jinja2, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-swing-research-bot-design.md` (§6 Splits, §7 Portfolio backtest, "Decisions after Phase 1": the holdout runs automatically only on finalists that passed validation). Obey `CLAUDE.md`.

## Global Constraints

- **Holdout:** 2024+ data is loaded only by `scripts/holdout.py`, which goes through `build_panel(market, allow_holdout=True)`. Every access is logged by `store.load_bars`. It runs at most once: it writes `data/holdout_used.json` and refuses if that file exists.
- **Holdout eligibility:** only finalists from `views.PRIMARY` (etf, crypto). Stock-view finalists are reported but never sent to the holdout, because today's index members are survivorship-biased in every period.
- **Horizon:** h = 10, the horizon all discovery tables used. No other horizon is gated, which avoids extra multiple testing.
- **Finalist rule** (validation split, same view):
  - sufficient (`MIN_N`, `max(MIN_N, 5h)` dates);
  - edge > 0;
  - q < 0.10, where q is BH over all hypotheses gated in that view;
  - date-weighted mean excess > 0;
  - mean absolute return (`mean_ret`) > 0.
- **Hypotheses gated per view:**
  - discovery solo rows with verdict "candidate";
  - discovery pair rows with verdict "synergy". For a pair, the validation test is the pair's combo edge (`edge_ab`, `q_ab` recomputed), plus lift > 0 on validation.
- **Backtest:**
  - Fixed horizon `h=10`. Entry at the open of t+1, exit at the close of t+h. The trade return is the panel's `ret_{direction}_{h}`, which is already net of `config.COSTS`.
  - Equal-weight slots: `max_positions` slots, default 10, each holding 1/max_positions of equity at entry.
  - Each day, new signals fill the free slots in random order (`numpy default_rng(seed)`, default seed 0).
  - A trade's P&L is booked on its exit date.
  - No leverage, no reinvestment within a trade.
  - Benchmark: buy-and-hold of `config.BENCHMARK[market]` over the same dates, from its close prices via `store.load_bars` (holdout flag as the caller passes).
- **Robustness grid:**
  - `max_positions` in {8, 10, 12} (±20%);
  - seeds 0–4;
  - per-calendar-year returns;
  - a 2.5/97.5-percentile clipped variant of the mean trade return, as a tail check. The final review found long-breakout crypto edges are tail-driven.
- Ruling recorded in the plan: the spec's ATR stop and 1%-risk sizing are replaced by fixed-horizon equal slots. That is the exit the statistics actually tested. An ATR-exit variant would need exit dates the panel does not store; it is deferred.
- Tests never hit the network.

## Review Focus

1. **The holdout script runs when no primary finalist exists, or runs a second time:** it must refuse both and load nothing. Assert that `holdout_log.txt` is unchanged.
2. **A finalist that fires 0 times in some year:** per-year rows show 0 trades, not NaN errors.
3. **More signals than free slots on a day:** selection is random but reproducible by seed. A trade never occupies a slot past its exit, and capacity is never exceeded.
4. **A trade whose `ret_*` is NaN** (end of data): it is skipped and never booked.
5. **A pair hypothesis whose validation A∧B is insufficient:** it is not a finalist, and the reason is shown.

---

### Task 1: Validation gate

**Files:** create `src/swingbot/analysis/validation.py`; test `tests/test_validation.py`.

**Interfaces:**
- Consumes: `solo_table`, `report.build.add_verdicts`, `pair_table`, `stats.bh_qvalues`, `views.PRIMARY`.
- Produces:
  - `GATE_COLS = ["view", "kind", "rule", "direction", "disc_edge", "disc_q", "val_n", "val_n_dates", "val_mean", "val_mean_ret", "val_edge", "val_lift", "val_p", "val_q", "finalist", "holdout_eligible", "reason"]`
  - `validation_gate(view_panel: pd.DataFrame, view: str, horizon: int = 10, signal_info=None) -> pd.DataFrame` (GATE_COLS):
    - `kind` is "solo" or "pair";
    - `rule` is the signal name, or `"a+b"`;
    - discovery tables are computed with `split="discovery"`, validation tables with `split="validation"`;
    - `val_q` is BH over this view's gated rows' `val_p`;
    - `val_lift` is NaN for solo rows;
    - `reason` is "" for finalists, otherwise the first failing condition, in this order: "insufficient on validation", "edge <= 0", "q >= 0.10", "mean excess <= 0", "mean return <= 0", "lift <= 0";
    - `holdout_eligible` = `finalist and view in PRIMARY`;
    - no hypotheses → an empty frame with GATE_COLS.

- [ ] **Step 1: Write the failing tests.** Use a synthetic view panel with discovery and validation rows and the `INFO` style of `tests/test_pairs.py`:
  - `test_persistent_signal_is_finalist`: an effect planted in both splits gives `finalist=True`, and `holdout_eligible` depends on the view name ("crypto" True, "stock" False).
  - `test_discovery_only_effect_fails_on_validation`: an effect planted in discovery only gives `finalist=False`, with `reason` in {"edge <= 0", "q >= 0.10"}.
  - `test_negative_absolute_return_not_finalist`: positive excess but negative `ret` on validation gives reason "mean return <= 0".
  - `test_pair_hypothesis_gated`: a planted A∧B synergy in both splits gives a pair row "A+B" that is a finalist.
  - `test_no_hypotheses_empty`.
- [ ] **Step 2:** Run them; expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `pytest -q`; expect PASS.
- [ ] **Step 5: Commit** — `feat(validation): gate discovery candidates and synergies on the validation split`.

---

### Task 2: Portfolio backtester

**Files:** create `src/swingbot/backtest/__init__.py`, `src/swingbot/backtest/portfolio.py`; test `tests/test_portfolio.py`.

**Interfaces:**
- Produces:
  - `rule_mask(panel, rule: str) -> np.ndarray`: "a+b" → A∧B; otherwise the single signal column.
  - `simulate(panel, rule: str, direction: str, split: str, horizon: int = 10, max_positions: int = 10, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]`:
    - `trades` columns: `date` (signal day), `asset`, `entry_idx`, `exit_date`, `ret`;
    - `equity` is a `pd.Series` indexed by every distinct panel date of the split, starting at 1.0, with booked P&L applied on exit dates.
    - Exit date = the panel date h distinct asset bars after the signal date. Use that asset's own subsequent dates in the panel. If fewer than h remain, skip the trade.
    - Slot accounting: a slot is busy from the signal date + 1 through the exit date inclusive.
    - Each booked trade adds `slot_value * ret`, where `slot_value = equity_at_entry / max_positions`.
  - `metrics(equity: pd.Series, bench_close: pd.Series | None = None) -> dict`:
    - keys `total_return, cagr, max_drawdown, sharpe_monthly, n_trades, exposure, bench_total_return, bench_cagr, bench_max_drawdown`;
    - `sharpe_monthly` = mean / std of month-end equity returns × sqrt(12);
    - `exposure` = mean fraction of busy slots;
    - bench values are NaN when `bench_close` is None.
  - `yearly_returns(equity) -> pd.Series` (calendar year → return).

- [ ] **Step 1: Write the failing tests:**
  - `test_single_trade_booking`: one asset, one firing, ret = +10%, max_positions=10 → equity rises 1% on the exit date and is flat elsewhere.
  - `test_capacity_never_exceeded_and_seeded`: 50 assets all firing every day, max_positions=5 → at most 5 open trades at any time; same seed gives identical trades, a different seed gives different ones.
  - `test_nan_ret_skipped`.
  - `test_metrics_hand_example`: an equity series with a known drawdown and CAGR.
  - `test_yearly_returns_zero_trade_year`: a year with no trades returns 0.0, not NaN.
- [ ] **Step 2:** Run them; expect FAIL.
- [ ] **Step 3: Implement.** Use a day loop over dates, with an array of slot end-dates. Performance must handle about 1M signal rows over about 4,500 dates in under 30 s: pre-group candidate rows by date.
- [ ] **Step 4:** Run `pytest -q`; expect PASS.
- [ ] **Step 5: Commit** — `feat(backtest): capacity-limited fixed-horizon portfolio simulator`.

---

### Task 3: Finalists report, robustness and `scripts/finalists.py`

**Files:** create `src/swingbot/report/finalists.py`, `src/swingbot/report/templates/finalists.html.j2`, `scripts/finalists.py`; test `tests/test_finalists_report.py`.

**Interfaces:**
- Consumes: `validation_gate`, `simulate`, `metrics`, `yearly_returns`, `universe_views`, `SUSPECT_LABEL`, `CDN_URL`, the report helpers, and `store.load_bars` for benchmark closes (holdout locked).
- Produces: `build_finalists_report(panels, out_path, horizon=10, as_of=None) -> tuple[Path, pd.DataFrame]`. It returns the path and the concatenated gate table, which is also written to `reports/finalists_{stamp}.csv`. For each view:
  1. **Gate table:** every gated hypothesis with its reason.
  2. For each finalist, from any view, the stock view labelled suspect:
     - the validation-split backtest (metrics against benchmark buy-and-hold, equity chart with the benchmark line, yearly returns);
     - the discovery-split backtest for reference, labelled "in-sample";
     - the robustness table (max_positions {8, 10, 12} × seeds 0–4 → validation total return and Sharpe; min/median/max);
     - the tail check: mean trade return raw vs clipped at 2.5/97.5.
  3. **Holdout status line:**
     - "Holdout locked: no finalist from the primary views (ETF, crypto) passed validation", or
     - "Eligible for the one-time holdout run: …" listing the eligible rules, and naming `scripts/holdout.py`.

  The caveat box reuses the survivorship text and `SHORT_NOTE`. It also says that fixed-horizon equal-slot sizing replaces ATR/1%-risk sizing (see the plan ruling), and that capacity/slippage beyond `config.COSTS` is not modelled.
- `scripts/finalists.py --horizon 10`: builds `reports/finalists_report_{stamp}.html` and the CSV, then prints both paths.

- [ ] **Step 1: Write the failing test.** Synthetic panels with one persistent planted crypto signal and nothing in the ETFs. Benchmark close bars are saved to a tmp `DATA_DIR` via `store.save_bars`. Assert that:
  - the report says crypto has 1 finalist and that it is eligible for the holdout;
  - the ETF section has 0 finalists;
  - the robustness table has 15 runs;
  - the CSV exists.

  A second test has no persistent signal and asserts the "Holdout locked" sentence.
- [ ] **Step 2:** Run it; expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `pytest -q`; expect PASS.
- [ ] **Step 5: Commit** — `feat(report): finalists report with validation backtests and robustness`.

---

### Task 4: One-time holdout runner

**Files:** create `scripts/holdout.py` and `src/swingbot/holdout.py` (the logic, so it is testable); test `tests/test_holdout.py`.

**Interfaces:**
- Produces: `run_holdout(finalists_csv: Path, horizon: int = 10, out_dir: Path | None = None) -> Path | None`.
  - Reads the gate CSV and keeps rows with `holdout_eligible == True`.
  - If there are none, it prints "holdout locked: no eligible finalists", returns None and loads nothing.
  - If `config.DATA_DIR / "holdout_used.json"` exists, it raises `RuntimeError("holdout already used on <date>")`.
  - Otherwise:
    - writes `holdout_used.json` **first**, with timestamp and rules;
    - builds the holdout panels for the needed markets via `build_panel(market, allow_holdout=True)` and `universe_views`;
    - for each eligible rule, runs `simulate(..., split="holdout")` and the solo/pair edge test on `split="holdout"`;
    - writes `reports/holdout_report_{stamp}.html` (a short page: per rule, the edge, mean_ret, backtest metrics vs benchmark, and a one-line verdict "confirmed" if edge > 0 and p < 0.10 and mean_ret > 0, otherwise "not confirmed");
    - returns the path.
  - `scripts/holdout.py --finalists reports/finalists_<stamp>.csv` calls it.

- [ ] **Step 1: Write the failing tests** (tmp `DATA_DIR`, synthetic saved bars, `build_panel` monkeypatched to a synthetic holdout panel where convenient):
  - `test_no_eligible_rows_loads_nothing`: the CSV has only stock finalists → returns None, and neither `holdout_log.txt` nor `holdout_used.json` is created.
  - `test_second_run_refused`: an existing `holdout_used.json` → RuntimeError, and `holdout_log.txt` is unchanged.
  - `test_marker_written_before_loading`: monkeypatch `build_panel` to raise → `holdout_used.json` still exists afterwards. A crashed run still counts as used, so nobody "retries" after peeking.
  - `test_happy_path_writes_report`: one eligible synthetic rule → the report exists and contains "confirmed" or "not confirmed".
- [ ] **Step 2:** Run them; expect FAIL.
- [ ] **Step 3: Implement.** `split_of` already labels 2024+ rows "holdout", so `split="holdout"` works on a holdout panel.
- [ ] **Step 4:** Run `pytest -q`; expect PASS.
- [ ] **Step 5: Commit** — `feat(holdout): one-time, logged, finalist-only holdout runner`.

---

### Task 5: Real run (controller)

- [ ] Run `python scripts/finalists.py --horizon 10`. Record the gate results per view.
- [ ] If any `holdout_eligible` finalist exists, run `python scripts/holdout.py --finalists reports/finalists_<stamp>.csv` exactly once. The user pre-authorised this ("Only if finalists pass"). Otherwise leave the holdout locked.
- [ ] Final whole-branch review of the Phase 6 range, then report to the user.
