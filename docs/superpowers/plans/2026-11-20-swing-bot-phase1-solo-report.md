# Swing Bot Phase 1: Data → Signals → Outcomes → Solo Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download stock/ETF and crypto daily data, compute ~35 swing signals per asset-day, attach forward outcomes and regime labels, and produce an HTML report of each signal's solo edge broken down by market and regime.

**Architecture:** Three layers kept separate: signals (bool matrix, no trading), outcomes (forward returns from next open), analysis (statistics on the discovery split). Everything ends up in one long-format Parquet panel per market (`date, asset, <signals>, <outcomes>, <regimes>, <meta>, split`). Holdout dates are cut off at load time.

**Tech Stack:** Python (3.15 if wheels install, else 3.12 via `uv`), pandas, numpy, pyarrow, yfinance, requests, statsmodels, plotly, jinja2, lxml, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-swing-research-bot-design.md`

**Out of this plan (later plans):** pair lift / conditional edge / triples / FDR across pairs / redundancy map (Phase 4), LightGBM + SHAP (Phase 5), portfolio backtest + holdout run (Phase 6), delisted crypto archives from data.binance.vision, validation-split confirmation.

## Global Constraints

- Daily bars only. Signal at close of t; entry at **open of t+1**.
- Costs per round trip: stocks `0.0010`, crypto `0.0030`.
- Horizons: `(5, 10, 20)` trading bars.
- Splits — stocks: discovery 2000-01-01..2017-12-31, validation 2018-01-01..2023-12-31; crypto: discovery 2018-01-01..2021-12-31, validation 2022-01-01..2023-12-31; **holdout 2024-01-01 onward for both**.
- Holdout rows never loaded unless `allow_holdout=True`; each such load appends a line to `data/holdout_log.txt`.
- Min occurrences: 100 per row, 30 per regime slice; below that the row is flagged `insufficient`.
- Benchmarks: `SPY` (stocks), `BTCUSDT` (crypto).
- Indicators hand-written in pandas; no TA-Lib.
- Free data only; no API keys.
- Tests never hit the network (monkeypatch fetchers).

## Review Focus

1. Asset with short history (< 250 bars, e.g. new coin): signals all False, outcomes NaN, regimes "unknown"; pipeline must not crash — test in Task 3 and Task 5.
2. Gaps/NaN bars (trading halts, missing crypto days): indicators must not raise; quality check reports gaps — test in Task 2 and Task 3.
3. Calendar mismatch between asset and benchmark (crypto trades weekends, stocks don't; asset listed later than benchmark): market regime aligned by date with forward-fill, never back-fill — test in Task 6.
4. Signal that never fires or fires a handful of times: solo table shows it with `insufficient=True`, no division-by-zero in profit factor or t-stat — test in Task 7.
5. Network failure / empty response / delisted ticker during download: that asset is skipped and logged, download continues — test in Task 2.

---

### Task 1: Project setup + config + store with holdout guard

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/swingbot/__init__.py`, `src/swingbot/config.py`, `src/swingbot/data/__init__.py`, `src/swingbot/data/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces:
  - `config.COSTS: dict[str, float]` = `{"stock": 0.0010, "crypto": 0.0030}`
  - `config.HORIZONS: tuple[int, ...]` = `(5, 10, 20)`
  - `config.SPLITS: dict[str, dict[str, tuple[str, str]]]` keyed by market then `"discovery"|"validation"`
  - `config.HOLDOUT_START = "2024-01-01"`, `config.MIN_N = 100`, `config.MIN_N_SLICE = 30`
  - `config.BENCHMARK = {"stock": "SPY", "crypto": "BTCUSDT"}`
  - `config.DATA_DIR: Path` (= repo `data/`, overridable by env `SWINGBOT_DATA`)
  - `store.save_bars(symbol: str, market: str, df: pd.DataFrame) -> Path` → `DATA_DIR/raw/{market}/{symbol}.parquet`
  - `store.load_bars(symbol: str, market: str, allow_holdout: bool = False) -> pd.DataFrame`
  - `store.split_of(dates: pd.DatetimeIndex, market: str) -> pd.Series[str]` values `"discovery"|"validation"|"holdout"|"pre"`
  - Bars schema everywhere: `DatetimeIndex` named `date` (tz-naive, sorted, unique), float columns `open, high, low, close, volume`.

- [ ] **Step 1:** `git init`; write `pyproject.toml` (src layout, package `swingbot`, deps above, `[tool.pytest.ini_options] pythonpath=["src"]`), `.gitignore` (`data/`, `.venv/`, `__pycache__/`, `reports/`).
- [ ] **Step 2:** Create venv and install: `python -m venv .venv; .venv\Scripts\pip install -e .[dev]`. If any wheel fails on 3.15: `uv venv --python 3.12 .venv; uv pip install -e .[dev]`. Expected: `python -c "import pandas, statsmodels, plotly"` succeeds.
- [ ] **Step 3: Write failing tests** in `tests/test_store.py` (use `tmp_path` + monkeypatched `DATA_DIR`):
  - `test_roundtrip`: save then `load_bars(..., allow_holdout=True)` equals original.
  - `test_holdout_cut`: bars 2023-12-29..2024-01-03; default load max date == 2023-12-29.
  - `test_holdout_logged`: `allow_holdout=True` appends one line containing symbol to `holdout_log.txt`.
  - `test_split_of`: stock dates 2017-12-29→"discovery", 2018-01-02→"validation", 2024-01-02→"holdout", 1999-12-31→"pre"; crypto 2021-12-31→"discovery".
- [ ] **Step 4:** Run `pytest tests/test_store.py -v` → FAIL (import error).
- [ ] **Step 5:** Implement `config.py` and `store.py`.
- [ ] **Step 6:** Run `pytest tests/test_store.py -v` → PASS.
- [ ] **Step 7:** Commit `feat: project setup, config, bar store with holdout guard`.

---

### Task 2: Data fetchers, universe, quality checks, download script

**Files:**
- Create: `src/swingbot/data/yahoo.py`, `src/swingbot/data/binance.py`, `src/swingbot/data/universe.py`, `src/swingbot/data/quality.py`, `scripts/download.py`, `universe/etfs.csv`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `store.save_bars`, bars schema.
- Produces:
  - `@dataclass(frozen=True) AssetMeta(symbol: str, market: str, sector: str, cap_bucket: str, is_etf: bool, is_major: bool)` in `universe.py`
  - `fetch_yahoo(ticker: str, start: str = "2000-01-01") -> pd.DataFrame` (`auto_adjust=True`; bars schema; empty DataFrame on no data)
  - `fetch_binance(symbol: str, start: str = "2017-01-01") -> pd.DataFrame` — GET `https://api.binance.com/api/v3/klines`, `interval=1d`, `limit=1000`, page by `startTime` = last open time + 1 day until fewer than 1000 rows returned; drop the still-open current-day candle.
  - `stock_universe() -> list[AssetMeta]` — S&P 500 from the Wikipedia constituents table (`pd.read_html`), sector = GICS sector, `cap_bucket="large"`; plus S&P MidCap 400 table with `cap_bucket="mid"`; plus `universe/etfs.csv` rows (`is_etf=True`, sector = ETF's sector or `"broad"`). Cached to `universe/stocks.csv` after first build; Yahoo ticker form (`BRK.B`→`BRK-B`).
  - `crypto_universe(n: int = 60) -> list[AssetMeta]` — `/api/v3/ticker/24hr`, USDT quote pairs, exclude stablecoin bases (`USDC, FDUSD, TUSD, DAI, USDP, BUSD, EUR, USDE`) and leveraged tokens (base ending `UP, DOWN, BULL, BEAR`), top n by `quoteVolume`; `is_major` for BTC, ETH; sector `"crypto"`, cap_bucket `"major"|"alt"`. Cached to `universe/crypto.csv`.
  - `check_bars(df: pd.DataFrame, market: str) -> list[str]` — issue strings for: duplicate dates, gaps (> 5 calendar days stock, > 1 crypto), non-positive prices, `high < low`, > 5% zero-volume bars, stock single-bar move > 50% (split suspicion).
  - `scripts/download.py [--market stock|crypto|all] [--limit N]` — fetch, `check_bars`, save; writes `data/universe_meta.parquet` (all AssetMeta rows) and `data/quality_report.csv` (symbol, issues); a fetch exception or empty frame logs `SKIP <symbol>: <reason>` and continues.
- `universe/etfs.csv` columns `symbol,sector`: SPY broad, QQQ broad, IWM broad, DIA broad, XLK tech, XLF financials, XLE energy, XLV health, XLI industrials, XLY discretionary, XLP staples, XLU utilities, XLB materials, XLRE realestate, XLC communication, GLD gold, TLT bonds, HYG bonds, EEM international, EFA international.

- [ ] **Step 1: Write failing tests** (all network monkeypatched):
  - `test_binance_paging`: fake `requests.get` returns 1000 rows then 3 rows; result has 1003 rows minus current-day candle, schema correct, index sorted.
  - `test_yahoo_normalizes`: fake `yf.download` frame with MultiIndex columns → bars schema, lowercase columns.
  - `test_crypto_universe_filters`: fake ticker list containing USDCUSDT, BTCUPUSDT, ETHBTC, BTCUSDT → only BTCUSDT kept, `is_major=True`.
  - `test_check_bars_flags`: frame with duplicate date, a 10-day gap (stock), `high<low` row → three corresponding issue strings; clean frame → `[]`.
  - `test_download_skips_failures`: fetcher raising for one symbol and returning empty for another; script's `run(symbols)` saves the good one and returns two SKIP messages.
- [ ] **Step 2:** `pytest tests/test_data.py -v` → FAIL.
- [ ] **Step 3:** Implement the five modules and `etfs.csv`. `download.run(assets: list[AssetMeta], fetch: Callable) -> list[str]` is the testable core; `main()` parses args.
- [ ] **Step 4:** `pytest tests/test_data.py -v` → PASS.
- [ ] **Step 5:** Smoke run: `python scripts/download.py --market all --limit 5` → 5 stock + 5 crypto parquet files exist under `data/raw/`.
- [ ] **Step 6:** Commit `feat: stock and crypto data download with quality checks`.

---

### Task 3: Indicators

**Files:**
- Create: `src/swingbot/indicators.py`
- Test: `tests/test_indicators.py`

**Interfaces:**
- Produces (all pure, input Series/bars DataFrame, output aligned to input index, NaN during warm-up, never raise on NaN input):
  - `sma(s, n)`, `ema(s, n)` (`adjust=False`)
  - `rsi(close, n=14)` — Wilder smoothing (`ewm(alpha=1/n, adjust=False)`)
  - `atr(df, n=14)` — Wilder, true range uses prior close
  - `bollinger(close, n=20, k=2.0) -> tuple[mid, upper, lower]` (population std, `ddof=0`)
  - `keltner(df, n=20, k=1.5) -> tuple[mid, upper, lower]` (EMA mid, ATR width)
  - `macd(close, fast=12, slow=26, signal=9) -> tuple[line, signal]`
  - `adx(df, n=14) -> tuple[adx, plus_di, minus_di]` (Wilder)
  - `stochastic(df, n=14, d=3) -> tuple[k, d]`
  - `obv(df)`, `ad_line(df)`
  - `zscore(close, n=20)`
  - `donchian_high(high, n)` / `donchian_low(low, n)` — max/min of the **previous** n bars (`rolling(n).max().shift(1)`)
  - `supertrend_dir(df, n=10, mult=3.0) -> Series[int]` (+1/−1)
  - `psar_dir(df, step=0.02, max_step=0.2) -> Series[int]` (+1/−1)
  - `swing_low(low, k=2) -> Series[float]` / `swing_high(high, k=2)` — fractal pivot: bar t−k is lower than the k bars each side; value is placed at bar **t** (when confirmed), forward-filled = "last confirmed swing low". Never uses bars after t.

- [ ] **Step 1: Write failing tests** with small hand-computed fixtures:
  - `test_sma_ema_values` on `[1..10]`: `sma(n=3).iloc[-1]==9`, `ema(n=3)` matches manual recursion.
  - `test_rsi_all_up_is_100` and `test_rsi_known_value` (Wilder example series; assert to 2 decimals).
  - `test_atr_uses_prior_close` (gap-up bar: TR = high − prev close).
  - `test_bollinger_ddof0`, `test_macd_shapes`, `test_adx_range_0_100`, `test_stoch_range_0_100`.
  - `test_donchian_excludes_today`: today's high equal to new max is not included in today's value.
  - `test_swing_low_no_future`: `swing_low(low)` computed on `low[:t+1]` equals full-series value at t for every t.
  - `test_nan_and_short_input`: 10-bar frame with a NaN close → every function returns same-length output, no exception.
- [ ] **Step 2:** `pytest tests/test_indicators.py -v` → FAIL.
- [ ] **Step 3:** Implement `indicators.py`.
- [ ] **Step 4:** `pytest tests/test_indicators.py -v` → PASS.
- [ ] **Step 5:** Commit `feat: hand-written indicator library`.

---

### Task 4: Signal registry + signal library + lookahead test

**Files:**
- Create: `src/swingbot/signals/__init__.py` (imports all family modules so they register), `registry.py`, `trend.py`, `pullback.py`, `breakout.py`, `meanrev.py`, `momentum.py`, `divergence.py`, `volume.py`, `sr.py`, `candles.py`
- Test: `tests/test_signals.py`

**Interfaces:**
- Consumes: `indicators.*`.
- Produces:
  - `@dataclass(frozen=True) Signal(name: str, family: str, direction: Literal["long","short"], fn: Callable[[pd.DataFrame], pd.Series])`
  - `REGISTRY: dict[str, Signal]`; decorator `register(name, family, direction="long")`
  - `compute_signals(df: pd.DataFrame) -> pd.DataFrame` — one bool column per registered signal, NaN → False.
  - `cross_sectional_signals(closes: pd.DataFrame) -> dict[str, pd.DataFrame]` — for `mom_12_1_top_decile` (12-month return skipping last month, top 10% across assets present that day, rebalanced monthly on first trading day, held through month). Columns = assets.
  - Signal names (column names, used by every later phase), family in brackets:
    - [trend] `ma20_50_cross_up`, `ma20_50_cross_down`(short), `ma50_200_cross_up`, `ma50_200_cross_down`(short), `close_above_ma200_cross`, `macd_cross_up`, `macd_cross_down`(short), `adx_trend_up` (ADX>25 and +DI crosses above −DI), `supertrend_flip_up`, `supertrend_flip_down`(short), `psar_flip_up`
    - [pullback] `pullback_ma20_uptrend` (low ≤ SMA20, close > SMA20, SMA20 > SMA50 > SMA200), `rsi14_pullback_uptrend` (RSI14 crosses down into 40–50 while close > SMA200), `rsi2_oversold` (RSI2 < 10, close > SMA200), `fib_retrace_50_618` (close within 50–61.8% retrace between last confirmed swing low and swing high, swing high after swing low, close > SMA200)
    - [breakout] `donchian20_breakout`, `donchian20_breakdown`(short), `donchian55_breakout`, `squeeze_release_up` (BB inside Keltner previous bar, not today, close > BB mid), `volume_breakout` (close > donchian_high(20) and volume > 1.5× SMA20 volume), `high_52w` (close > donchian_high(252))
    - [meanrev] `bb_lower_close`, `bb_upper_close`(short), `rsi14_oversold` (cross below 30), `rsi14_overbought`(short, cross above 70), `zscore_below_2`, `stoch_oversold_cross` (%K crosses above %D with both < 20), `gap_down_fill_setup` (open < prior low by > 1 ATR×0.5 and close > open)
    - [momentum] `roc20_accel` (ROC20 > 0 and ROC20 − ROC20.shift(5) > 0 crossing up), `mom_12_1_top_decile` (cross-sectional)
    - [divergence] `rsi_bull_divergence` (new confirmed swing low below prior swing low while RSI14 at the new swing is above RSI14 at prior swing), `macd_bull_divergence` (same with MACD line)
    - [volume] `obv_leads_price` (OBV at 20-bar high while close below its 20-bar high), `close_above_avwap_swinglow` (close crosses above VWAP anchored at last confirmed swing low), `ad_rising` (A/D line crosses above its SMA20 while close > SMA50)
    - [sr] `swing_low_bounce` (low within 1% of last confirmed swing low, close > open), `pivot_support_hold` (low ≤ classic daily pivot S1 from prior bar's HLC, close > S1)
    - [candles] `bullish_engulfing`, `hammer` (lower wick ≥ 2× body, upper wick ≤ body, after 3 down closes), `inside_bar_breakout` (prior bar inside its predecessor; today closes above the mother bar high), `morning_star`
  - "Cross" signals fire only on the bar the condition becomes true (event, not state).

- [ ] **Step 1: Write failing tests:**
  - `test_registry_complete`: set of registered names equals the list above (minus `mom_12_1_top_decile`), every family in the 9 families, every direction valid.
  - `test_no_lookahead` (parametrized over all registry names): random-walk fixture of 600 bars (seed 0); for 20 random t ≥ 300, `compute_signals(df.iloc[:t+1]).iloc[-1] == compute_signals(df).iloc[t]`.
  - `test_short_history_all_false`: 50-bar frame → all columns False, no exception.
  - Hand-built fixtures for one signal per family asserting it fires on exactly the expected bar(s): `ma20_50_cross_up`, `rsi2_oversold`, `donchian20_breakout`, `bb_lower_close`, `roc20_accel`, `rsi_bull_divergence`, `obv_leads_price`, `swing_low_bounce`, `bullish_engulfing`.
  - `test_cross_sectional_top_decile`: 20 synthetic assets with known 12-1 returns → top 2 flagged for the whole month, none flagged before 12 months of history.
- [ ] **Step 2:** `pytest tests/test_signals.py -v` → FAIL.
- [ ] **Step 3:** Implement registry and family modules.
- [ ] **Step 4:** `pytest tests/test_signals.py -v` → PASS.
- [ ] **Step 5:** Commit `feat: swing signal library with lookahead guard`.

---

### Task 5: Outcomes

**Files:**
- Create: `src/swingbot/outcomes.py`
- Test: `tests/test_outcomes.py`

**Interfaces:**
- Consumes: `indicators.atr`, `config.COSTS`, `config.HORIZONS`.
- Produces: `compute_outcomes(df: pd.DataFrame, bench: pd.DataFrame, market: str) -> pd.DataFrame` indexed like `df`, columns (h ∈ HORIZONS):
  - `gross_{h}` = `close[t+h] / open[t+1] − 1`
  - `ret_long_{h}` = `gross_{h} − cost`; `ret_short_{h}` = `−gross_{h} − cost`
  - `bench_gross_{h}` = same formula on benchmark, aligned on asset dates (benchmark reindexed to asset dates; if benchmark lacks t+1 or t+h dates use next available benchmark bar; NaN if none)
  - `excess_long_{h}` = `ret_long_{h} − bench_gross_{h}`; `excess_short_{h}` = `ret_short_{h} + bench_gross_{h}`
  - `mfe_{h}` = `max(high[t+1..t+h]) / open[t+1] − 1`; `mae_{h}` = `min(low[t+1..t+h]) / open[t+1] − 1`
  - `atr_exit_long`, `atr_exit_short` — entry open[t+1], stop 2×ATR14(t), target 3×ATR14(t), walk bars t+1..t+20; first touch wins; if stop and target both inside the same bar → stop; else exit close[t+20]; minus cost.
  - Last rows lacking t+h are NaN; rows where ATR is NaN have NaN ATR exits.

- [ ] **Step 1: Write failing tests:**
  - `test_entry_is_next_open`: hand series where `open[t+1]` differs from `close[t]`; `gross_5` uses `open[t+1]`.
  - `test_costs_applied`: crypto `ret_long_5 == gross_5 − 0.003`.
  - `test_excess_vs_bench`: asset = benchmark → `excess_long_h == −cost` everywhere defined.
  - `test_atr_exit_stop_first_same_bar`: bar hitting both stop and target → result equals stop return minus cost.
  - `test_atr_exit_time_exit`: flat series → exits at close[t+20].
  - `test_tail_nan` and `test_short_history_no_crash` (30 bars → `*_20` all NaN, no exception).
  - `test_calendar_mismatch`: crypto asset with weekend bars vs stock-calendar benchmark → no exception, no back-filled values (bench value at t uses dates ≥ t only).
- [ ] **Step 2:** `pytest tests/test_outcomes.py -v` → FAIL.
- [ ] **Step 3:** Implement `outcomes.py` (vectorized for gross/mfe/mae via `shift`/rolling on reversed series; ATR exit may loop per row with numpy — acceptable at daily frequency).
- [ ] **Step 4:** `pytest tests/test_outcomes.py -v` → PASS.
- [ ] **Step 5:** Commit `feat: forward outcomes with costs, excess and ATR exits`.

---

### Task 6: Regimes + panel builder

**Files:**
- Create: `src/swingbot/regimes.py`, `src/swingbot/panel.py`, `scripts/build_panel.py`
- Test: `tests/test_regimes_panel.py`

**Interfaces:**
- Consumes: `compute_signals`, `cross_sectional_signals`, `compute_outcomes`, `store.load_bars`, `store.split_of`, `AssetMeta` rows from `data/universe_meta.parquet`.
- Produces:
  - `label_regimes(df: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame` columns:
    - `trend_regime`: `"up"` if close > SMA200 and SMA200 > SMA200.shift(20); `"down"` if close < SMA200 and SMA200 < SMA200.shift(20); else `"sideways"`; `"unknown"` while SMA200 NaN.
    - `vol_regime`: percentile rank of ATR14/close within trailing 504 bars (min 252): `< 1/3` `"low"`, `> 2/3` `"high"`, else `"mid"`; `"unknown"` before.
    - `market_regime`: `"bull"` if benchmark close > benchmark SMA200 else `"bear"`; benchmark aligned to asset dates with **forward-fill only**; `"unknown"` before benchmark SMA200 exists.
    - `adx_regime`: ADX14 > 25 `"trending"`, < 20 `"ranging"`, else `"neutral"`.
  - `build_panel(market: str, allow_holdout: bool = False) -> pd.DataFrame` — long format: `date, asset` + all signal columns + outcome columns + regime columns + `sector, cap_bucket, is_etf, is_major, market` + `split`. Rows with split `"pre"` dropped.
  - `scripts/build_panel.py --market stock|crypto|all` → `data/panel/{market}.parquet`.

- [ ] **Step 1: Write failing tests:**
  - `test_trend_regime_labels` on synthetic rising / falling / flat series.
  - `test_market_regime_ffill_only`: benchmark missing weekends → asset weekend rows take Friday's value; asset dates before benchmark start → `"unknown"`.
  - `test_short_history_unknown`: 100-bar asset → all regimes `"unknown"` except `adx_regime` where defined.
  - `test_build_panel_shape` with 3 synthetic assets in tmp `DATA_DIR`: row count = sum of non-"pre", non-holdout bars; required columns present; no holdout dates.
- [ ] **Step 2:** `pytest tests/test_regimes_panel.py -v` → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `pytest tests/test_regimes_panel.py -v` → PASS.
- [ ] **Step 5:** Commit `feat: regime labels and per-market panel builder`.

---

### Task 7: Statistics + solo analysis

**Files:**
- Create: `src/swingbot/analysis/__init__.py`, `stats.py`, `solo.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: panel DataFrame from Task 6, `REGISTRY`, `config.MIN_N`, `config.MIN_N_SLICE`.
- Produces:
  - `@dataclass TestResult(mean: float, se: float, t: float, p: float, n: int, n_dates: int)`
  - `clustered_mean_test(values: pd.Series, dates: pd.Series, horizon: int) -> TestResult` — average values per date (removes cross-sectional correlation), then OLS on a constant with HAC (Newey–West) `maxlags=horizon` (removes overlap); if `n_dates < 3` return NaN se/t/p, never raise.
  - `bh_qvalues(p: np.ndarray) -> np.ndarray` — Benjamini–Hochberg, NaN-aware (NaN in → NaN out).
  - `solo_mask(panel: pd.DataFrame, signal: str) -> pd.Series[bool]` — signal fires and no signal from a **different family** fires on the same asset-day.
  - `solo_table(panel: pd.DataFrame, horizon: int = 10, metric: str = "excess", group_by: list[str] | None = None, split: str = "discovery") -> pd.DataFrame` — one row per (signal[, groups]); picks `{metric}_{direction}_{horizon}` by signal direction; columns: `signal, family, direction, *group_by, n_all, n_solo, mean_all, mean_solo, edge_vs_all_days` (mean_all minus the unconditional mean of that metric over all rows in the same group/split), `median_all, hit_rate, profit_factor, t, p, q, occ_per_year, insufficient`. `insufficient = n_all < MIN_N` (no groups) or `< MIN_N_SLICE` (with groups). `profit_factor` = sum gains / |sum losses|, `inf` if no losses, NaN if no rows. `q` from `bh_qvalues` across all rows of the table.
  - `mom_12_1_top_decile` is included via its panel column (family `momentum`, long).

- [ ] **Step 1: Write failing tests:**
  - `test_clustered_mean_known`: iid normal values mean 0.01 over 2000 dates → mean ≈ 0.01, p < 0.05; values all 0 → p > 0.5 or NaN.
  - `test_clustered_collapses_same_day`: 100 identical values on one date → `n_dates == 1`, se NaN, no exception.
  - `test_bh_qvalues_known`: `[0.01, 0.04, 0.03, 0.2]` → `[0.04, 0.0533, 0.0533, 0.2]` (4 dp); NaN preserved.
  - `test_solo_mask_other_family`: A (trend) + B (trend) same day → still solo for A; A + C (meanrev) → not solo.
  - `test_solo_table_direction_metric`: short signal uses `excess_short_10`.
  - `test_solo_table_rare_signal`: signal firing 3 times → row present, `insufficient=True`, finite or NaN values, no warnings raised as errors; signal never firing → `n_all == 0`, `profit_factor` NaN.
  - `test_solo_table_groups`: `group_by=["market","trend_regime"]` → one row per signal per present group.
  - `test_split_filter`: validation rows excluded when `split="discovery"`.
- [ ] **Step 2:** `pytest tests/test_analysis.py -v` → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `pytest tests/test_analysis.py -v` → PASS.
- [ ] **Step 5:** Commit `feat: clustered stats and solo signal analysis`.

---

### Task 8: HTML report + analyze script + end-to-end run

**Files:**
- Create: `src/swingbot/report/__init__.py`, `report/build.py`, `report/templates/report.html.j2`, `scripts/analyze.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `solo_table`, panels from `data/panel/`, `data/quality_report.csv`.
- Produces:
  - `build_report(panels: dict[str, pd.DataFrame], out_path: Path, horizon: int = 10) -> Path` — single self-contained HTML (plotly via CDN `https://cdn.plot.ly/plotly-2.35.2.min.js`). Sections in order:
    1. **Caveats header** (fixed copy): "Discovery split only. Survivorship bias: delisted stocks and some delisted coins are missing. Costs: stocks 0.10%, crypto 0.30% per round trip. Results with q ≥ 0.10 or flagged insufficient are not evidence of an edge."
    2. Data coverage: assets per market, date range per market, count of quality issues.
    3. Solo leaderboard per market (`solo_table` no groups), sorted by `q` then `mean_all` desc; insufficient rows greyed.
    4. Heatmap: signal × (market, trend_regime) and signal × (market, vol_regime), cell = `mean_all` of excess in bps; insufficient cells blank; hover shows n, t, q.
    5. Segment table: `group_by=["market","sector"]` for stocks, `["market","cap_bucket"]` for crypto.
  - Every number column shown alongside `n`.
  - `scripts/analyze.py [--horizon 10]` → `reports/solo_report_{YYYY-MM-DD}.html` and CSVs of each table next to it.

- [ ] **Step 1: Write failing tests:**
  - `test_report_renders`: synthetic panels (2 markets × 3 assets) → file exists, contains caveat sentence verbatim, contains every registered signal name, contains `"plotly"`.
  - `test_report_insufficient_blank`: a slice with n < 30 renders as empty cell (no number).
- [ ] **Step 2:** `pytest tests/test_report.py -v` → FAIL.
- [ ] **Step 3:** Implement template, `build_report`, `analyze.py`.
- [ ] **Step 4:** `pytest -v` (full suite) → all PASS.
- [ ] **Step 5: End-to-end:** `python scripts/download.py --market all` (full universe; run in background, expect tens of minutes), then `python scripts/build_panel.py --market all`, then `python scripts/analyze.py`. Expected: report opens; `data/holdout_log.txt` absent or unchanged; leaderboard shows all 41 signals per market.
- [ ] **Step 6:** Commit `feat: solo signal HTML report and end-to-end pipeline`.
