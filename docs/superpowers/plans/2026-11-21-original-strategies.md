# Original Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the five pre-registered strategies (C1, C2, C3, S1, S2), the new-market and Fear & Greed data, and a runner. The runner applies the fixed gates: dev, then validation, then a one-time final test.

**Architecture:**
- `data/foreign.py`: Japan/Europe universes, the glitch filter, the Fear & Greed fetcher.
- `strategies/`: weight and mask builders, plus an extended portfolio engine `backtest/rebalance.py` (cash is free to move; short funding).
- `scripts/originals.py`: runs the gates and writes the reports.

**Tech Stack:** Python 3.15, pandas, numpy, requests, yfinance, pytest.

**Spec:** `docs/superpowers/specs/2026-11-21-original-strategies-design.md`. It is binding and holds every rule, cost, criterion and resolution. Also obey `CLAUDE.md`.

## Global Constraints

- No lookahead: decide at close t, trade at open t+1. Every weight or mask builder needs a truncation (prefix-invariance) test.
- The crypto 2024+ holdout is loaded only by `scripts/originals.py --final`, via `build_panel("crypto", allow_holdout=True)`. The record file `data/holdout_use_crypto_originals.json` is written before loading.
- **Costs:**
  - crypto 0.30% per round trip; Japan and Europe 0.20%; US 0.10%.
  - Charged per side on turnover of non-cash assets only. Cash (USDT, S2 cash) is free and earns 0%.
- **C2:** short funding/borrow is 10%/yr on current short notional, accrued daily.
- Japan and Europe are fresh final-test markets: nothing but the final runner may look at their data.
- Tests never hit the network: monkeypatch `requests.get` and `yf.download`.

## Review Focus

1. **Lookahead in weekly/monthly signals:** C1/C3 rank on Sunday close and trade Monday open; S2 ranks at month-end close. Covered by truncation tests.
2. **FGI lag:** C3 uses the index value dated t−1. A test plants a value on day t and asserts that it has no effect on the decision at t.
3. **C2 short accounting:** a short loses when price rises; funding is charged on current notional; cash is never charged a cost.
4. **A delisted coin held across a rebalance:** valued at its last close, then sold at the next rebalance. No NaN may propagate into equity.
5. **S2 with fewer than 50 valid stocks:** holds cash.

---

### Task 1: New-market data, Fear & Greed, glitch filter, holdout scope

**Files:**
- create `src/swingbot/data/foreign.py`;
- modify `src/swingbot/config.py`, `src/swingbot/data/store.py`, `src/swingbot/data/download.py`, `scripts/download.py`;
- test `tests/test_foreign.py`.

**Interfaces (produces):**
- `config.COSTS` gains `"japan": 0.0020` and `"europe": 0.0020`.
- `config.HOLDOUT_MARKETS = {"stock", "crypto"}`. `store.load_bars` applies the holdout cut and the log only for markets in this set. For other markets it returns the full history without logging.
- `config.HOME_INDEX = {"japan": "^N225", ".DE": "^GDAXI", ".PA": "^FCHI", ".L": "^FTSE", "us": "^GSPC"}`.
- `foreign.japan_universe() -> pd.DataFrame` (columns `symbol, market, home_index`):
  - cached at `universe/japan.csv`;
  - built on a cache miss from the Wikipedia "Nikkei 225" constituents table;
  - symbols are `<4-digit code>.T`; market is `"japan"`; home_index is `"^N225"`.
- `foreign.europe_universe() -> pd.DataFrame`:
  - cached at `universe/europe.csv`;
  - built from the Wikipedia DAX (`.DE`), CAC 40 (`.PA`) and FTSE 100 (`.L`) constituent tables;
  - market is `"europe"`, home_index comes from the suffix;
  - duplicate company names across lists keep the first (home) listing.
- `foreign.download_market(market: str) -> list[str]`:
  - fetches each symbol plus the home index symbols with the existing `fetch_yahoo` and `fetch_with_retry`;
  - saves via `store.save_bars(symbol, market, df)`; index files are saved under the same market;
  - drops bars dated on or after the download date (UTC);
  - returns OK/SKIP log lines.
- `foreign.glitch_filter(df: pd.DataFrame) -> pd.DataFrame` (design resolution 5):
  - a day is a glitch if |close/prev − 1| > 0.60 **and** the price reverses by more than half that move within the next 2 bars;
  - glitch closes (and the same day's open, high and low) are set to NaN, then forward-filled;
  - returns a copy.
- `foreign.fetch_fng() -> pd.Series`:
  - GET `https://api.alternative.me/fng/?limit=0&format=json`;
  - returns a float Series indexed by tz-naive date (UTC day), sorted;
  - `foreign.load_fng()` / `save_fng()` read and write `data/raw/sentiment/fng.parquet`.
- `scripts/download.py` gains `--market japan|europe|fng`.

**Steps:**
- [ ] **Step 1:** Write failing tests:
  - `test_holdout_scope`: a japan-market file keeps 2025 rows without logging; a stock-market file is still cut and logged.
  - `test_japan_universe_parse` and `test_europe_universe_parse`: synthetic HTML tables via monkeypatched `requests.get`; correct suffixes and home indices; cross-list duplicate removed.
  - `test_glitch_filter`: a +200% one-day spike that reverts is removed; a real +70% move that holds is kept.
  - `test_fetch_fng`: a fake JSON with unix timestamps gives the right dates and values.
  - `test_download_market_drops_today`.
- [ ] **Step 2:** Run the tests; expect them to fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `pytest -q`; expect it to pass.
- [ ] **Step 5:** Commit with message `feat(data): Japan/Europe universes, FGI, glitch filter, holdout scope` and the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

### Task 2: Strategy builders and the rebalance engine

**Files:**
- create `src/swingbot/backtest/rebalance.py` and `src/swingbot/strategies/__init__.py`, `crypto.py`, `stocks.py`;
- test `tests/test_strategies.py`.

**Interfaces (produces):**
- `rebalance.run_weights(opens, closes, weights, cost_per_side, cash_cols=("CASH",), short_funding_annual=0.0) -> pd.Series`:
  - same timing and drift semantics as `backtest.trend.run_portfolio` (a weight row dated at a signal close trades at the next bar's open). Reuse or generalise that code; do not copy-paste it.
  - Turnover cost excludes `cash_cols`.
  - Negative weights are shorts.
  - Funding is charged daily: `short_funding_annual / 365 × |current short notional|`.
  - Cash columns are constant 1.0 prices (no return).
  - Missing prices are forward-filled.
  - `equity.attrs` holds scalars only: `worst_short_loss` (worst return of any single short position between rebalances; NaN if none).
- `crypto.weekly_dates(index) -> DatetimeIndex`: Sundays in the index.
- `crypto.c1_weights(closes, eligible, btc="BTCUSDT", top=5, lookback=30, sma=200) -> pd.DataFrame`:
  - rows are weekly signal dates; columns are assets plus `"CASH"`;
  - 0.2 each for the top 5 by 30-day return among eligible coins with 30 prior bars, when BTC close > its 200-day SMA; otherwise `CASH = 1.0`;
  - fewer than 5 candidates → equal weight among those available (cash if none).
- `crypto.c2_weights(closes, eligible, btc="BTCUSDT", bottom=5, lookback=30) -> pd.DataFrame`: BTC +0.5, bottom 5 eligible non-BTC coins −0.1 each, `CASH` +1.0. Fewer than 5 candidates → −0.5/n each.
- `crypto.c3_weights(closes, eligible, fng: pd.Series, mode: str, ...)`: `mode` is "contrarian" (gate open when fng[t−1] ≤ 25) or "confirming" (fng[t−1] ≥ 55); C1 weights where the gate is open, else `CASH = 1.0`. A missing FGI value means the gate is closed.
- `stocks.s1_frame(bars: dict[str, DataFrame], index_close: dict[str, pd.Series], home: dict[str, str], cost: float, horizon=10) -> pd.DataFrame`:
  - the long frame for `portfolio.simulate`, with columns `date, asset, s1, ret_long_10, split`;
  - signal: RSI(14) < 30, close > SMA200, and home index close > its SMA200 (forward-filled to the stock's dates);
  - `ret_long_10`: next open to close t+10, minus `cost`;
  - split is filled by the runner.
- `stocks.s2_weights(closes, home_regime: pd.DataFrame, min_valid=50, top_frac=0.10, min_top=5) -> pd.DataFrame`:
  - monthly union-calendar month-end rows;
  - 12-1 momentum `close[t−21]/close[t−252] − 1`;
  - holds the top 10% (at least 5) of stocks with a valid value **and** whose home regime is on (each stock is gated by its own home index); the remainder goes to `CASH`;
  - fewer than 50 valid stocks → `CASH = 1.0`;
  - `home_regime` is a bool frame (dates × stocks) giving "home index above its 200-day SMA", forward-filled.
- `stocks.equal_weight_hold(closes, min_valid)`: the monthly-rebalanced equal-weight benchmark over the same universe and dates (S2 criteria 2 and 3).

**Steps:**
- [ ] **Step 1:** Write failing tests:
  - truncation/no-lookahead tests for c1, c2, c3, s1_frame and s2_weights;
  - `test_c3_fng_lag` (an FGI change on day t does not affect the decision at t);
  - `test_c1_btc_regime_cash`;
  - `test_c2_short_loses_when_price_rises_and_funding` (one short; price +10% over 7 days; equity change ≈ −0.1×10% − funding − costs);
  - `test_cash_turnover_free` (moving fully to CASH costs only the sold legs);
  - `test_delisted_coin_ffilled_no_nan`;
  - `test_s2_min_valid_cash`;
  - `test_s2_home_regime_per_stock`;
  - `test_s1_signal_and_returns_net_cost` (a small hand example).
- [ ] **Step 2:** Run the tests; expect them to fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `pytest -q`; expect it to pass.
- [ ] **Step 5:** Commit with message `feat(strategies): C1-C3, S1, S2 builders and rebalance engine` and the trailer.

---

### Task 3: Runner with gates and the one-time final test

**Files:** create `scripts/originals.py` and `src/swingbot/strategies/run.py` (the logic, so it can be tested); test `tests/test_originals_run.py`.

**Interfaces:**
- `run.crypto_inputs(panel: pd.DataFrame, allow_holdout: bool) -> (opens, closes, eligible)`:
  - loads each panel asset's bars via `store.load_bars` and re-applies `panel.split_relistings`, so `"SYM@date"` segment names match the panel;
  - `eligible` is pivoted from the panel's `eligible` column (dates × assets, False where missing).
- `run.us_stock_inputs(stock_panel)` and `run.foreign_inputs(market)`: bars dicts, index closes and home maps. Foreign bars go through `foreign.glitch_filter` first.
- `run.evaluate(strategy: str, equity: pd.Series, period: tuple, benchmarks: dict) -> dict`:
  - computes the spec's criteria for that strategy on the slice from the first valid-signal date in the period, re-based to 1;
  - returns metrics, `pass: bool`, and the failing reason.
  - Benchmarks per spec:
    - C1/C3: BTC buy-and-hold;
    - C2: absolute;
    - S1: median of the `__all__` baseline over seeds 0–4 (S1 itself also takes the median over seeds 0–4);
    - S2: same-universe equal-weight hold.
  - Also returns an information-only HAC (lag 3) t-stat of monthly returns against the benchmark, plus the home-index figures.
- **Dev mode** (`scripts/originals.py`, no flag):
  - Runs the crypto strategies on the existing crypto panel (holdout locked) and S1/S2 on US data. The US universe is the single stocks from `universe/stocks.csv` with ^GSPC as home index; download ^GSPC into `data/raw/stock` if missing.
  - Prints and saves `reports/originals_dev_{stamp}.txt|csv`: dev and validation tables and pass flags.
  - **C3:** computes both modes on discovery only, picks the higher Sharpe, and writes `{"c3_mode": ..., "discovery_sharpes": {...}}` to `reports/originals_c3_choice.json` **before** computing C3's validation. If that file already exists, the stored choice is reused.
  - Writes `reports/originals_gate.json`: strategy → validation pass.
- **Final mode** (`--final`):
  - reads `originals_gate.json` and runs only the strategies that passed validation.
  - **Crypto** (if any crypto strategy passed):
    - pre-flight checks that the files exist;
    - writes the record `data/holdout_use_crypto_originals.json` (it refuses if the record already exists);
    - builds the panel with `build_panel("crypto", allow_holdout=True)`;
    - evaluates the period 2024-01-01 to the end;
    - saves results, or the error, into the record.
  - **Stocks** (if S1 and/or S2 passed): runs Japan and Europe (full history; the period starts at each strategy's first valid date) and applies the verdict rules (confirmed, mixed, not confirmed).
  - Saves `reports/originals_final_{stamp}.txt|csv`.
  - Never runs a final test for a strategy that failed validation.

**Steps:**
- [ ] **Step 1:** Write failing tests:
  - `evaluate` criteria for each strategy type on hand-built equity curves;
  - the C3 choice is persisted and reused, and validation is computed only after the choice file exists;
  - final mode skips failed strategies;
  - the crypto record is written before `build_panel` is called (monkeypatched), and a second run is refused;
  - the stock verdict logic (both, one, neither).

  Use tmp `DATA_DIR` and `reports` dirs; never touch real data.
- [ ] **Step 2:** Run the tests; expect them to fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `pytest -q`; expect it to pass.
- [ ] **Step 5:** Commit with message `feat(originals): gated runner with one-time final tests` and the trailer.

---

### Task 4: Real runs (controller)

- [ ] Download the data: `python scripts/download.py --market fng`, then `--market japan`, then `--market europe`, plus ^GSPC. Do not inspect the Japan or Europe data.
- [ ] Run `python scripts/originals.py`. Record the dev and validation results and the C3 choice.
- [ ] Run `python scripts/originals.py --final`, only for strategies that passed. It runs once.
- [ ] Final review, push, and report to the user.
