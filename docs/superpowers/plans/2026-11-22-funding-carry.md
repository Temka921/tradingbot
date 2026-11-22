# Funding-Rate Carry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Implement the pre-registered funding-carry test (F1, F2) end to end: data, backtest and a gated runner with a one-time final.

**Spec (binding: rules, return model, costs, periods, pass criteria, holdout handling):** `docs/superpowers/specs/2026-11-22-funding-carry-design.md`. Obey `CLAUDE.md`.

**Tech:** Python 3.15, pandas, numpy, requests, pytest. Reuse:
- `swingbot.data.binance` archive helpers (S3 listing pagination, zip parsing; generalise the prefix rather than copying);
- `swingbot.panel` (split_relistings, build_panel);
- `swingbot.backtest.portfolio.metrics`;
- `swingbot.strategies.crypto.weekly_dates`;
- `swingbot.strategies.run`: patterns for the record, refusal and `_spend` (reuse where it fits).

## Global Constraints

- No lookahead:
  - F2 ranks on data up to Sunday's close and trades at Monday's open.
  - Daily funding for day d is used only for returns on day d, never for that day's selection.
- 2024+ data (funding, perp, spot) never enters dev/validation code paths. It is loaded only in `--final`, after the record `data/holdout_use_crypto_funding.json` is written; a second run is refused.
- Costs: spot 0.10% per side and perp 0.05% per side, on each leg's traded notional. Cash moves are free.
- Tests never hit the network.

### Task 1: Futures data

- Create `src/swingbot/data/futures.py`:
  - `list_um_symbols()`: perpetual USDT symbols from the archive listing `data/futures/um/monthly/fundingRate/`, paginated.
  - `fetch_funding(symbol)`: monthly archive zips, plus the API `fapi/v1/fundingRate` for the months after the archive (paginate by startTime). Returns a DataFrame with `funding_time` (tz-naive UTC) and `rate`, sorted and deduped. Handle optional header rows and ms/µs timestamps.
  - `fetch_perp_klines(symbol)`: `data/futures/um/monthly/klines/{S}/1d/` zips plus the API `fapi/v1/klines` for recent months. Bars schema; the still-open day is dropped.
  - `save_funding`/`load_funding` (`data/raw/funding/`), and perp bars via `store.save_bars(sym, "perp", df)`.
  - `daily_funding(df) -> pd.Series`: the sum of rates per UTC day.
- `config.HOLDOUT_MARKETS` gains `"perp"`. A loader for funding (`load_funding(symbol, allow_holdout=False)`) cuts at HOLDOUT_START unless allowed, and logs when allowed, the same way as `store.load_bars`.
- Add `scripts/download.py --market futures`: universe = USDT perps whose base is in `universe/crypto.csv`; downloads funding and perp klines.
- Tests:
  - archive listing pagination;
  - zip parse with and without a header;
  - ms vs µs timestamps;
  - API tail pagination;
  - `daily_funding` sums;
  - the holdout cut and log for funding and perp.

### Task 2: Carry backtest

- Create `src/swingbot/strategies/carry.py`:
  - `f1_weights(dates) -> pd.DataFrame`: BTC 0.5 and ETH 0.5 on the first date, i.e. constant targets (rebalanced weekly back to 50/50).
  - `f2_weights(funding_daily: pd.DataFrame, eligible: pd.DataFrame, top=5, lookback=7) -> pd.DataFrame`: weekly Sunday rows; weights over coins plus `CASH`; the rule exactly as in the spec.
  - `run_carry(spot_open, spot_close, perp_open, perp_close, funding_daily, weights, spot_cost=0.001, perp_cost=0.0005) -> pd.Series`: daily equity starting at 1.
    - A weight row dated at a signal close trades at the next day's open.
    - Carry position w in coin X means a spot long of w/2 and a perp short of w/2, with w/2 held as margin.
    - Each day: `equity_change = Σ_X (w_X/2) × (spot_ret − perp_ret + funding_day)`. On trade days the returns run from the open.
    - Positions drift between rebalances: track spot and perp leg values separately.
    - Costs are charged on traded notional per leg.
    - A missing price on either leg after a position exists closes the position at the last closes; its weight goes to cash.
    - `attrs` holds scalars only: `negative_funding_share`, `avg_gross_funding_yield`, `turnover_per_year`.
- Tests:
  - a hand-computed one-coin case: price +10% on both legs, funding 0.01%/day for 3 days → equity = 1 + 0.5 × 3 × 0.0001 minus costs;
  - negative funding is paid;
  - cash moves are free;
  - no-lookahead truncation for `f2_weights`;
  - F2 holds cash when no coin has positive 7-day funding;
  - a delisting closes the position without NaN.

### Task 3: Runner

- Create `scripts/funding_carry.py` (logic in `src/swingbot/strategies/carry_run.py`):
  - **Dev mode:**
    - load spot from the existing crypto panel (holdout locked, with relisting split names; eligible from the panel), plus perp and funding with `allow_holdout=False`;
    - run F1 and F2 continuously;
    - evaluate dev (2020) and validation (2022–2023) with the spec's three criteria, plus information columns;
    - write `reports/funding_dev_{stamp}.txt|csv` and `reports/funding_gate.json`.
  - **Final mode (`--final`):**
    - only strategies that passed validation;
    - pre-flight file checks, then the record (exclusive create) before any 2024+ load;
    - `build_panel("crypto", allow_holdout=True)`, then perp and funding with `allow_holdout=True`;
    - evaluate 2024-01-01 to the end;
    - results or error into the record;
    - `reports/funding_final_{stamp}.txt|csv`;
    - if nothing passed, print that and spend nothing.
- Tests:
  - criteria evaluation;
  - gate respected;
  - record written before any holdout load (monkeypatched);
  - second run refused;
  - nothing spent when nothing passed.
