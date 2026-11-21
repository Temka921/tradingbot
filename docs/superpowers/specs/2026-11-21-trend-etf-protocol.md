# Trend-following ETF test: pre-registered protocol

Date: 2026-11-21. Written and committed **before** any result was computed. Nothing below may change after results are seen.

## Hypothesis

A textbook trend filter improves an ETF portfolio's risk-adjusted return, and above all cuts large drawdowns, versus simply holding the same ETFs. The filter: hold an asset only while it is above its 10-month moving average (Faber, 2007, "A Quantitative Approach to Tactical Asset Allocation").

## Rules (fixed, textbook, no tuning)

**Rule A (primary): Faber GTAA-5**
- Assets: SPY (US stocks), EFA (international stocks), IEF (US 7-10y Treasuries), VNQ (US REITs), DBC (commodities).
- Signal: at each month-end close, compute each asset's 10-month simple average of month-end closes. Uses only data up to that close.
- Position: each asset gets a 20% slot.
  - Close > its 10-month average → hold the asset in its slot.
  - Otherwise → hold SHY (1-3y Treasuries, the cash proxy) in that slot.
- Trading: trade at the **next trading day's open**, the project's no-lookahead convention.
- Costs: 0.10% per round trip (`config.COSTS["stock"]`), charged as 0.05% on each unit of weight bought or sold.
- Start: the first month-end at which every asset has 10 month-end closes.

**Rule B (secondary, breadth check):** the same rule over all ETFs in `universe/etfs.csv` that have 10 month-end closes at that date. Equal slots among them; SHY is excluded from the risk set and is the cash proxy.

**Benchmarks:**
- (1) Buy-and-hold of the same assets, equal weight, rebalanced monthly at the same times with the same costs.
- (2) SPY buy-and-hold.

## Periods

These follow the project's stock/ETF splits:
- Discovery: start to 2017-12-31.
- Validation: 2018-01-01 to 2023-12-31.
- Holdout: 2024-01-01 onward. Used once, and only if Rule A passes validation.

## Pass criteria for Rule A on validation (all must hold)

1. Total return > 0.
2. Sharpe ratio (monthly returns, annualised) ≥ the Sharpe of benchmark (1).
3. Max drawdown smaller in magnitude than benchmark (1).

The same three are reported for discovery (in-sample, for context) and, if run, for the holdout. Holdout verdict: "confirmed" if all three hold on the holdout, otherwise "not confirmed".

Honesty notes:
- Six years of monthly data cannot give statistical significance. The test asks whether the rule *behaved as documented* out of sample, not whether it is proven.
- A t-test of the monthly return difference (rule − benchmark 1, HAC lag 3) is reported for information only.

## Holdout handling

- The ETF holdout has never been examined. If Rule A passes validation, ETF bars from 2024-01-01 onward are loaded **only** via `store.load_bars(..., allow_holdout=True)`, which logs every access.
- The use is recorded in `data/holdout_use_etf_trend.json`.
- Rule B is reported on the holdout only as a secondary line.
- Crypto holdout data is not touched.
