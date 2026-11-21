# Original Strategies (C1, C2, C3, S1, S2): Design

Date: 2026-11-21. Repo: this research repo (`swingbot`). The design is fixed before any result is computed.

## Intent

None of the 41 textbook methods, nor the trend-following ETF rule, passed a final out-of-sample test. The user wants to invent strategies from patterns this project actually found, then prove them only on data that has never been searched:

- the crypto 2024–2026 holdout (still unused), and
- markets never downloaded: Japanese and European stocks.

The goal is an honest answer to "does any original idea work?", not a strategy that looks good.

## Goals

- Five pre-registered strategies, each built on an observed pattern:
  - C1, C2, C3 for crypto;
  - S1, S2 for stocks.
- Develop and sanity-check on already-studied data:
  - crypto 2018–2021 (discovery), confirmed on crypto 2022–2023 (validation);
  - US stocks 2000–2017 (discovery), confirmed on US stocks 2018–2023 (validation).
- One final test per strategy on fresh data:
  - crypto: 2024-01-01 to the latest complete bar;
  - stocks: Japan and Europe, full available history.
- A short plain-English report plus CSVs, comparable to the earlier reports.

## Non-goals

- Live trading or broker integration (FinalBot stays paused).
- Parameter searches. The only data-chosen choice in the whole design is C3's direction (see below).
- AI-scored news. It is excluded because a model trained on 2024–2026 text leaks hindsight into a backtest.

## Anti-goals (failure even if the code works)

- Changing any rule, threshold or pass bar after seeing a final-test result.
- Adding strategy variants beyond these five, or re-running a final test.
- Reporting a "pass" that only beats a weak benchmark. Every strategy is compared with a fair baseline that shares its biases.

## Constraints

Existing project rules apply (CLAUDE.md):
- no lookahead: decide at close t, trade at open t+1;
- locked holdouts loaded only with `allow_holdout=True` (logged);
- costs always on;
- tests never hit the network.

Further constraints for this work:
- **Costs:**
  - crypto 0.30% per round trip (0.15% per side on turnover);
  - Japan and Europe stocks 0.20% per round trip (0.10% per side): foreign fees and wider spreads;
  - US stocks keep 0.10%.
- **Crypto universe:**
  - the existing panel: all Binance USDT pairs ever listed, relisted tickers split, delisted coins exit at last close;
  - point-in-time top-100 liquidity `eligible` flag.
- **Survivorship:**
  - US and foreign stock universes are today's index members, which biases them.
  - Mitigation: stock strategies must beat a same-universe, same-rules random baseline, which shares the bias.
  - The report says plainly that a stock pass is weaker evidence than a crypto pass.
- **Machine:** Windows, 7.9 GB RAM, Python 3.15 venv. Process one market at a time.

## Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| Search many patterns on 2000–2023 and keep the best | Finds noise; nothing left to test it on | Rejected |
| Test each idea in every market | Patterns here were market-specific (dip-buying helped US stocks, hurt crypto), so cross-asset tests mean little | Rejected |
| **Few hypothesis-driven strategies, developed on studied data, final-tested once on fresh data of the same asset class** | Fewer shots, but each result is honest | **Chosen** |

Within the chosen approach, the user picked the five strategies below. News is included only through the crypto Fear & Greed Index (C3).

## The strategies

Shared portfolio rules:
- Rebalance-based strategies (C1, C2, C3, S2) produce target weights at a signal close, executed at the next bar's open. They reuse `backtest.trend.run_portfolio`.
- Holdings drift between rebalances.
- A delisted coin's last close is carried forward, so it is effectively exited at its last close. It is sold at the next rebalance.

### C1: Crypto momentum rotation (pattern: in crypto, strength persists; bear markets hurt everything)

- **Rebalance:** weekly, at the close of each Sunday (UTC daily bars). Trade at Monday's open.
- **Ranking:** coins `eligible` that day (point-in-time top-100 by liquidity, from the panel), ranked by 30-day return `close_t / close_{t-30} - 1`. A coin needs 30 prior bars.
- **Holdings:** the top 5, equal weight.
- **Regime switch:** hold coins only when BTC's close is above its 200-day simple average. Otherwise hold 100% cash (USDT, 0% return).
- **Costs:** 0.15% per unit of weight traded.

### C2: Crypto weakness hedge (pattern: weak coins kept trailing BTC in both studied periods)

- **Rebalance:** weekly, same schedule as C1.
- **Short leg:** the bottom 5 `eligible` coins by 30-day return, excluding BTC, 10% of capital each, 50% total.
- **Long leg:** 50% BTC (spot).
- **Cash:** the rest is held as USDT, so weights are BTC +0.5, five shorts at −0.1 each, USDT +1.0 (sums to 1).
- **Costs:** 0.15% per unit traded on both legs.
- **Funding/borrow:** a conservative assumed cost of 10% per year on short notional, charged daily (0.0274%/day × 0.5). This is an assumption; real perpetual funding varies and has often paid shorts.
- **Caveat:** C2 needs a futures/margin account; it cannot be done with spot crypto.

### C3: Crypto mood + momentum (news factor: Crypto Fear & Greed Index)

- **Data:** alternative.me Fear & Greed Index, daily since 2018-02-01 (`https://api.alternative.me/fng/?limit=0&format=json`), cached to `data/raw/sentiment/fng.parquet`.
- **Timing:** use the value dated the day **before** the signal close (lag 1 day), so publication timing can never leak.
- **Rule:** C1, plus a mood gate. If the gate is closed at a rebalance, hold cash instead of coins.
- **The one data-chosen choice in this design:** the gate is either
  - (a) contrarian: open only when the lagged index ≤ 25 ("extreme fear"), or
  - (b) confirming: open only when the lagged index ≥ 55 ("greed").

  Pick the option with the higher Sharpe on crypto **discovery (2018-02 to 2021-12) only**. Record the choice in the results file before validation is computed.
- Everything else is identical to C1.

### S1: Dip in a healthy uptrend (pattern: dips that bounced were mostly inside uptrends; regime mattered most in the ML model)

- **Signal at close t:**
  - RSI(14) < 30, using `indicators.rsi`;
  - the stock's close is above its own 200-day simple average;
  - its home market index's close is above that index's 200-day simple average, forward-filled to the stock's dates.
- **Execution:** fixed horizon. Enter at the open of t+1, exit at the close of t+10 bars, using `backtest.portfolio.simulate` with 10 equal slots and random daily fill (seeds 0–4).
- **Trade return:** forward 10-day return from the next open, net of that market's cost.
- **Home indices:** US ^GSPC, Japan ^N225, Germany ^GDAXI, France ^FCHI, UK ^FTSE, all via Yahoo.

### S2: Momentum with a crash switch (pattern: trend filters cut crashes; momentum is the best-documented effect)

- **Rebalance:** monthly, at each month-end close, within one market's universe.
- **Ranking:** 12-1 momentum, `close_{t-21} / close_{t-252} - 1`. A stock needs 252 prior bars.
- **Holdings:** the top 10% (at least 5 stocks), equal weight.
- **Regime switch:** if the home index is below its 200-day average at the signal close, hold 100% cash (0% return).
- **Europe:** treated as one universe. Each stock's crash switch uses its own home index, so a German stock is held only while the DAX is above its average.
- **Costs:** per market, applied on turnover.

## Markets and data (new)

**Japan:** Nikkei 225 constituents (Wikipedia "Nikkei 225"), Yahoo tickers `<code>.T`.

**Europe:**
- DAX 40 (`.DE`), CAC 40 (`.PA`) and FTSE 100 (`.L`), constituents from Wikipedia;
- de-duplicate cross-listings (keep the home listing);
- FTSE prices in pence are fine, because only returns are used.

**Shared rules:**
- Universe CSVs are cached in `universe/` (`japan.csv`, `europe.csv`, with columns symbol, market, home_index) and committed.
- Bars are downloaded from 2000-01-01 with the existing Yahoo fetcher.
- New market names are `japan` and `europe`, stored under `data/raw/{market}/`. `config.COSTS` gains `japan: 0.0020` and `europe: 0.0020`.
- These markets are entirely fresh, so **no** holdout lock is needed for them. Their full history is the final test.
- Discipline: the controller downloads them and runs only the final-test script on them. No exploratory analysis, plots or ad-hoc queries on Japan or Europe data before the final run. That keeps them unseen.

## Phases and pass criteria (fixed now)

Each strategy goes through three gates.

| Gate | Crypto (C1, C2, C3) | Stocks (S1, S2) |
|---|---|---|
| Dev check (in-sample) | crypto 2018-02 to 2021-12 | US 2000–2017 |
| Validation gate | crypto 2022–2023 | US 2018–2023 |
| Final test (once) | crypto 2024-01-01 to latest complete bar (holdout, logged) | Japan full history and Europe full history |

**Pass criteria** (the same at the validation gate and the final test; computed on the period's equity, re-based to 1 at the period start):
- **C1, C3:**
  - (1) Sharpe (monthly returns, annualised) ≥ BTC buy-and-hold Sharpe over the same period;
  - (2) |max drawdown| < BTC's;
  - (3) at the final test only, total return > 0.
- **C2 (market-neutral):**
  - (1) total return > 0;
  - (2) Sharpe ≥ 0.5.
- **S1:**
  - (1) total return > 0;
  - (2) median total return over seeds 0–4 > the median of the same-model random baseline (rule = every row of that universe, the `"__all__"` rule, same slots and seeds).
  - Sharpe is reported but is not a criterion. S1's equity counts profit and loss only when a trade closes (positions are not valued daily), which understates volatility.
- **S2:**
  - (1) total return > 0;
  - (2) total return > the equal-weight buy-and-hold of the same universe (monthly rebalanced, same costs, same data basis);
  - (3) Sharpe ≥ the Sharpe of that same equal-weight buy-and-hold.
- Home-index buy-and-hold figures (^GSPC, ^N225, ^GDAXI, ^FCHI, ^FTSE) are reported for information only. They are price indices, except the DAX, which includes dividends, while stock data is dividend-adjusted. Comparing against them would be biased, so they are not pass criteria.

**Rules for using the gates:**
- A strategy reaches its final test only if it passes its validation gate. It is never tuned in response to a validation failure.
- Stock final verdict:
  - "confirmed" if it passes in **both** Japan and Europe;
  - "mixed" if it passes in one;
  - "not confirmed" if it passes in neither.
- Crypto final verdict: "confirmed" or "not confirmed".
- With five strategies tested, one pass could be luck. The report says so, and also reports a monthly-return t-statistic (HAC, lag 3) against each strategy's benchmark, for information only.

## Holdout handling (crypto 2024+)

- The crypto final test is one script run with `--final`.
- It writes `data/holdout_use_crypto_originals.json` **before** loading; it refuses if that file exists.
- It checks every required file exists before writing the record.
- It builds the crypto panel with `build_panel("crypto", allow_holdout=True)`, so every load is logged.
- It drops the newest bar, which may be partial.
- It saves results into the record, and writes an error into the record if it crashes after writing it.
- It runs only strategies that passed validation.

## Outputs

- `reports/originals_dev_{stamp}.txt|csv`: dev and validation tables, plus C3's recorded direction choice.
- `reports/originals_final_{stamp}.txt|csv`: final-test tables and verdicts.
- A short plain-English summary for the user, delivered in chat.

## Builder-check resolutions (Claude's calls, fixed before results)

1. **Last bar:**
   - Crypto uses every bar the fetchers return. The Binance API fetcher already drops the still-open daily candle, and archive files hold closed months only, so the crypto run does not drop a bar.
   - Japan and Europe drop any bar dated on or after the download date (UTC).
   - The exact end date used is written into the results.
2. **C2 mechanics:**
   - Shorts drift with price between weekly rebalances. There is no cap and no simulated liquidation.
   - The report shows the worst single-short loss, so blow-up risk is visible.
   - Funding/borrow is 10%/yr on the **current** short notional, accrued daily inside the portfolio loop.
   - No margin or collateral is modelled beyond the USDT cash line.
3. **Cash is free to move:**
   - Turnover costs apply only to non-cash assets.
   - USDT (crypto) and cash (S2) carry no trading cost and earn 0%.
4. **Benchmarks:** see the S1 and S2 pass criteria above. Pass bars compare against same-universe benchmarks built from the same adjusted data; index figures are information only.
5. **Calendars and data glitches:**
   - Each stock's crash switch uses its home index, forward-filled to the stock's dates.
   - S2 rebalances on the last date of each month present in the universe's union calendar.
   - Prices are forward-filled.
   - Glitch filter, applied to Japan and Europe before any strategy runs: a stock-day is a glitch if |daily return| > 60% **and** the price reverses by more than half that move within 2 trading days. Glitch closes are set to NaN and forward-filled.
   - S1 trade returns are clipped at the pooled 0.1/99.9 percentiles, as in all earlier tables.
6. **Warm-up:**
   - Each period's statistics start at the first date in that period on which the strategy has a valid signal.
   - C3's direction choice therefore uses discovery from its first valid rebalance, about 2018-09, to 2021-12.
7. **S1 equity:**
   - The `simulate` realized-P&L equity curve is used. Idle slots earn 0%.
   - The random baseline uses the identical model, so the comparison is fair.
8. **One continuous run:**
   - Each strategy runs once over all available data for its market. Periods are slices of that one equity curve, re-based at the slice start.
   - The crypto final run's panel includes 2018 onward, so warm-up and state carry over naturally.
9. **S2 minimum universe:**
   - At a rebalance date, S2 needs at least 50 stocks with a valid 12-1 momentum value. Otherwise it holds cash.
   - The final-test period for each market starts at the first month-end with 50 or more valid stocks.

## Left to the builder

- Module layout. Suggested: `src/swingbot/strategies/` with one function per strategy that returns target weights, or masks for S1.
- How C2's negative weights and the funding drag are applied inside or beside `run_portfolio`.
- Wikipedia table parsing details and ticker cleaning.
- Test design. Required: a no-lookahead truncation test for every weight and mask builder, and a test that the FGI lag is applied.
- Report formatting.
