# Swing Trading Research Bot — Design

Date: 2026-11-20

## Intent

Build a Python research system that tests every codable swing trading method on
large amounts of historical data (US stocks/ETFs and crypto) and reports, honestly:

1. Which methods have an edge **on their own**.
2. Which methods **work best together**, and which one actually drives the result
   when several fire on the same day.
3. **Where** each method or combination works: which market, segment and regime.

The user wants to work like a quant: treat ideas as hypotheses, try to disprove
them, and only trust results that hold up on unseen data after costs.

## Goals

- Signal library covering the main swing families: trend, pullback, breakout,
  mean reversion, momentum, divergence, volume, support/resistance, candlesticks.
- Clean attribution of overlapping signals (solo, conditional, pair lift, model-based).
- Every result broken down by market, segment and regime.
- Statistical honesty: costs, minimum samples, multiple-testing correction,
  date-clustered uncertainty, walk-forward validation, a locked holdout.
- Finalists turned into real portfolio backtests and compared with buy-and-hold.

## Non-goals (v1)

- Live trading or broker connections (paper trading comes after v1).
- Intraday data/strategies. Daily bars only.
- An LLM making trading decisions. Claude builds and reviews; rules/models decide.
- Paid data.
- Parameter optimization. Each method uses fixed textbook defaults in v1.
- Subjective chart patterns (head & shoulders, flags, cup & handle). Their
  definitions are ambiguous; deferred to v2.

## Anti-goals (failure even if it "works")

- A ranked list of winners that are just multiple-testing luck.
- Any lookahead: a signal using data not available at the close of day t.
- Results reported without costs, or without comparison to the baseline drift.
- Combos "winning" only because they occur 12 times.

## Constraints

- Windows 10, Python 3.15 installed. Some scientific wheels (lightgbm, shap,
  numba) may not exist for 3.15 yet; if so, use a Python 3.12 venv (via `uv`).
- Free data only: Yahoo Finance (stocks/ETFs), Binance public API (crypto).
- Must run on a single desktop: signal tables stored as Parquet, vectorized pandas/numpy.

## Approaches considered

| Approach | Summary | Trade-off |
|---|---|---|
| **A. Strategy-per-backtest** | Write each method as a full strategy, run a backtester per method and per combo. | Simple, but combos explode, overlapping signals can't be separated, very slow. |
| **B. Signal matrix + event study + model attribution (chosen)** | Compute every signal as a boolean column per asset-day, attach forward outcomes, analyze solo/overlap/pairs/regimes statistically, then backtest only finalists as portfolios. | More upfront design, but attribution is clean, combos are cheap to evaluate, and breakdowns by market/regime fall out naturally. |
| **C. Pure ML** | Feed all indicators into one model, trade its predictions. | Black box, overfits fast, doesn't answer "which method worked". Kept only as one attribution tool inside B. |

Chosen: **B**, because the user's core questions (which method, which combo, which
market) are attribution questions, and B is the only one built to answer them.

## Decisions made with the user

- Markets: **US stocks/ETFs + crypto**.
- Style: swing trading (days to weeks), daily bars.
- Claude is the engineer and skeptical reviewer, not the trader.
- Baseline rules first; ML compared fairly afterwards.
- Attribution of overlapping signals and market-specific results are first-class.

## Architecture

Three layers, then a portfolio stage.

```
data  ->  signals (bool matrix)  ->  outcomes (forward returns)  ->  analysis  ->  report
                         \-> regimes/segments (labels) -----------/       \-> finalists -> portfolio backtest -> holdout
```

### 1. Data

- **Stocks/ETFs:** yfinance daily OHLCV, adjusted. Universe: current S&P 500
  constituents + S&P MidCap sample + major ETFs (SPY, QQQ, IWM, DIA, 11 sector
  SPDRs, GLD, TLT, etc.). From 2000 where available.
  Known limitation: survivorship bias (delisted stocks missing). Reported in every
  report header. Mitigation later: historical constituent lists.
- **Crypto:** Binance spot daily klines for top ~60 USDT pairs by volume, including
  delisted pairs where `data.binance.vision` archives exist. From 2017.
- Storage: one Parquet per asset under `data/raw/`, plus a metadata table
  (asset class, sector, cap bucket, ETF flag, BTC/ETH vs alt).
- Data quality checks: missing days, zero volume, split jumps, duplicate dates.

### 2. Signals

- Each method is a pure function `df -> pd.Series[bool]` (plus direction: long or
  short), registered by name and family. Computed from data up to and including
  the close of day t only.
- Indicators implemented in-house with pandas (no TA-Lib install pain), each unit-tested
  against hand-computed values.
- Initial library (~35, fixed defaults):
  - Trend: MA 20/50 cross, MA 50/200 cross, close>200MA, MACD cross, ADX>25 + DI,
    Supertrend flip, Parabolic SAR flip.
  - Pullback: touch of 20MA in uptrend, RSI(14) 40–50 in uptrend, RSI(2)<10 above 200MA,
    Fibonacci 50–61.8% retrace of last swing.
  - Breakout: Donchian 20 high, Donchian 55 high, Bollinger/Keltner squeeze release,
    breakout with volume > 1.5x avg, 52-week high.
  - Mean reversion: close < lower Bollinger, RSI(14)<30, z-score(20) < -2,
    Stochastic oversold cross, gap-down fill setup.
  - Momentum: 12-1 month relative strength top decile, ROC(20) acceleration.
  - Divergence: RSI bullish divergence, MACD bullish divergence.
  - Volume: OBV new high before price, close > anchored VWAP from swing low,
    accumulation/distribution rising.
  - S/R: bounce off prior swing low, pivot-point support hold.
  - Candles: bullish engulfing, hammer, inside-bar breakout, morning star.
  - Mirror short versions where the method defines one.
- Output: `signals/{asset}.parquet`, one bool column per signal.

### 3. Outcomes

- Entry at **open of t+1** (never the close that generated the signal).
- For horizons h = 5, 10, 20 trading days: net forward return, max favorable and
  max adverse excursion, and a standardized exit result (2x ATR stop / 3x ATR target /
  time exit at h).
- Costs per round trip: stocks 0.10% (fees + slippage), crypto 0.30%. Configurable.
- **Excess return** = asset forward return minus the benchmark's (SPY for stocks,
  BTC for crypto) forward return over the same dates, net of costs, so a signal
  doesn't look good just because the market went up. Reports also show each
  signal's edge vs the unconditional all-days mean of the same universe and split.

### 4. Regimes and segments (labels per asset-day, known at close t)

- Trend regime: price vs 200MA and its slope (up / down / sideways).
- Volatility regime: ATR% percentile vs own history (low / mid / high).
- Market regime: SPY (stocks) or BTC (crypto) above/below 200MA.
- Trending vs ranging: ADX > 25 vs < 20.
- Segments: asset class, sector, cap bucket, ETF vs single stock, BTC/ETH vs altcoin.

### 5. Analysis (the "smart" part)

All statistics computed on the discovery period only, then confirmed on validation.

- **Solo edge:** stats when signal A fires and no other signal from a different
  family fires the same day.
- **Conditional edge:** A+B days vs B-only days → A's marginal contribution.
- **Pair lift:** perf(A∧B) − max(perf(A only), perf(B only)). Positive = synergy,
  ~0 = redundant, negative = conflict. All pairs evaluated.
- **Triples:** only when both constituent pairs show positive lift and the triple
  has enough samples.
- **Model attribution:** gradient boosting (LightGBM) on the signal matrix + regime
  labels predicting forward excess return; SHAP values for per-signal contribution
  and SHAP interaction values for pairs. Walk-forward training only.
- **Redundancy map:** signal co-occurrence correlation, so near-duplicate methods are
  grouped instead of being counted as independent confirmation.
- Every stat cut by market × segment × regime.
- **Honesty rules:**
  - Minimum 100 occurrences (30 for a regime slice) or flagged "insufficient data".
  - Uncertainty via block bootstrap clustered by date (signals on the same day
    across many assets are not independent).
  - Benjamini–Hochberg false discovery rate across all tests; deflated Sharpe for
    finalists.
  - Metrics: mean net excess return, median, hit rate, profit factor, t-stat/CI,
    occurrences per year.

### 6. Splits

- Stocks: discovery 2000–2017, validation 2018–2023, **holdout 2024–present**.
- Crypto: discovery 2018–2021, validation 2022–2023, **holdout 2024–present**.
- Holdout is read only by the final step, once. Code enforces this: the loader
  refuses holdout dates unless an explicit `--final-holdout` flag is passed, and
  each use is logged.

### 7. Portfolio backtest of finalists

- Finalists (single signals and combos that pass validation) become tradable rules:
  entry next open, ATR stop, exit rule, position sizing at 1% account risk per trade,
  max open positions, per-market capital.
- Metrics: CAGR, max drawdown, Sharpe, Sortino, turnover, exposure, vs buy-and-hold
  SPY / BTC.
- Robustness: ±20% parameter jitter, per-year results, other market.
- Then one run on the holdout.

### 8. Report

- Static HTML report (plus CSVs): data coverage and caveats, solo leaderboard,
  pair-lift heatmap, method × market × regime matrix, redundancy map, finalists'
  equity curves, holdout result. Every number shows sample size and confidence.

## Project layout

```
src/swingbot/
  config.py          costs, splits, paths
  data/              yahoo.py, binance.py, universe.py, quality.py, store.py
  indicators.py      pure indicator functions
  signals/           registry.py + one module per family
  outcomes.py        forward returns, excursions, standardized exits
  regimes.py         regime and segment labels
  analysis/          solo.py, pairs.py, attribution.py, stats.py
  backtest/          portfolio.py
  report/            build.py, templates/
scripts/             CLI entry points (download, build_signals, analyze, backtest, report)
tests/
data/                (gitignored) raw, signals, outcomes, results
```

## Stack

pandas, numpy, pyarrow, yfinance, requests, scipy, statsmodels, lightgbm, shap,
plotly (report charts), jinja2, pytest. Indicators hand-written.

## Phases

1. Data pipeline + quality checks.
2. Indicators + signal library + lookahead tests.
3. Outcomes + regimes + solo analysis → first report.
4. Pairs, conditional edge, triples, redundancy map, FDR.
5. Model attribution (LightGBM + SHAP).
6. Portfolio backtest of finalists + robustness + holdout.
7. (After v1) paper trading via Alpaca / Binance testnet.

## Left to the builder

- Exact universe lists and ticker sourcing.
- Swing high/low detection rule (fractal-based, documented).
- Report styling.
- Performance tuning (vectorization, caching).

## Calls Claude made (user may override)

- Daily bars only, fixed method defaults, long + short signals both evaluated
  (short results flagged "requires shorting").
- Costs: 0.10% stocks, 0.30% crypto per round trip.
- Holdout = 2024 onward.
- Subjective chart patterns deferred to v2.
- Indicators hand-written instead of TA-Lib.
