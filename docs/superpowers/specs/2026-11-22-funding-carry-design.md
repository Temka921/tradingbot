# Funding-Rate Carry: Design (pre-registered)

Date: 2026-11-22. Written and committed before any result was computed. Rules and pass bars are fixed.

## Intent

Test whether a market-neutral "get paid for risk" strategy works where price-prediction strategies failed. The strategy is long spot crypto plus an equal short in the USDT-margined perpetual future, collecting the funding that leveraged longs pay.

**Economic reason:** demand for leveraged long exposure makes perpetual funding positive most of the time. The carry trade supplies the other side and is paid for it. No price forecast is needed.

## Data (new downloads)

**Funding history:**
- Source: Binance USDⓈ-M funding rates. Monthly archive `data.binance.vision/data/futures/um/monthly/fundingRate/{SYMBOL}/`, which runs from 2020-01 and includes delisted contracts. The latest partial month comes from the API (`fapi/v1/fundingRate`).
- Payments come every 8h (some contracts use 4h or 1h; use each row's own timestamp).
- Stored in `data/raw/funding/{SYMBOL}.parquet`: `funding_time` (UTC), `rate`.

**Perpetual prices:** daily klines from `data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/1d/`, stored in `data/raw/perp/{SYMBOL}.parquet` in the bars schema.

**Spot prices:** the existing crypto spot data, including delisted and relisted splits. Point-in-time liquidity eligibility comes from the existing panel logic.

**Universe:** USDT perpetuals whose spot pair exists in `universe/crypto.csv` (the same exclusions: no stablecoins, wrapped, tokenized or leveraged tokens).

**Daily funding:** the sum of that UTC day's funding rates, received by the short perp leg.

## Strategies (exactly two)

**F1: Static BTC+ETH carry**
- Always hold carry in BTC and ETH, 50% of capital each.

**F2: Rotating carry**
- **When:** every Sunday close (UTC); trade at Monday's open.
- **Candidates:** coins that are `eligible` that day (spot top-100 by liquidity) and have a perp with at least 7 days of funding history.
- **Selection:** rank by the mean daily funding over the last 7 days. Hold carry in the top 5 with a positive 7-day mean, at equal weight.
- **Empty slots:** fewer than 5 qualifying coins leaves the remaining slots in cash. No qualifying coins at all means 100% cash.

## Return model (per unit of capital)

**Position:** a "carry position" of weight w in coin X is spot long X of notional w/2, plus perp short X of notional w/2. The other w/2 of the capital is margin for the short (1× hedge, no leverage).

**Daily return of a position:** `(w/2) × (spot_ret − perp_ret + daily_funding_rate)`.
- spot_ret and perp_ret are close-to-close returns; on trade days they run from that day's open.
- Funding is received when positive and paid when negative.

**Costs:** spot 0.10% per side plus perp 0.05% per side, charged on each leg's traded notional, so 0.30% per round trip of a full carry position. Moving to or from cash is free; cash earns 0%.

**Delisting:** a spot or perp delisting or data end exits the position at the last available closes. Its weight goes to cash until the next rebalance.

**Not modelled:**
- exchange failure risk (FTX-type);
- liquidation of the short in extreme spikes (1× margin makes it unlikely but not impossible);
- interest on idle cash.

These are stated in the report.

## Periods

- Dev: 2020-01-01 to 2021-12-31.
- Validation: 2022-01-01 to 2023-12-31.
- Final (one-time crypto holdout): 2024-01-01 to the latest complete day.

Rules for the 2024+ data:
- All 2024+ data (funding, perp and spot) is excluded from dev/validation code paths.
- It is loaded only by `--final`, which:
  - writes `data/holdout_use_crypto_funding.json` before loading;
  - refuses a second run;
  - logs every spot access through `allow_holdout=True`;
  - saves results or the error into the record.

**Run structure:**
- Each strategy is one continuous run per mode.
- Periods are slices re-based to 1, starting at the first valid rebalance in the period.

## Pass criteria (each period; all must hold)

1. CAGR > 4% per year. This roughly beats holding cash in short-term Treasuries, which the strategy must beat to be worth its risks.
2. Sharpe (monthly returns, annualised, 0% risk-free) ≥ 1.0. A carry trade should be smooth.
3. Max drawdown smaller than 15% in magnitude.

**Gating:**
- A strategy reaches the final test only if it passes validation.
- Final verdict: "confirmed" if all three criteria hold on 2024+, otherwise "not confirmed".

**Information only:**
- the share of days with negative funding;
- the worst single week;
- the average gross funding yield;
- turnover;
- a BTC buy-and-hold line.

## Outputs

- `reports/funding_dev_{stamp}.txt|csv`
- `reports/funding_final_{stamp}.txt|csv`
- a plain-English summary for the user.

## Left to the builder

Module layout (suggested: `src/swingbot/data/futures.py`, `src/swingbot/strategies/carry.py`, `scripts/funding_carry.py`), archive pagination, and tests. Required tests:
- a hand-computed return check;
- funding sign;
- costs;
- no-lookahead truncation for F2 selection;
- a holdout guard test;
- network fully mocked.
