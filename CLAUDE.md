# Swing Trading Research Bot

Python research system that tests swing trading methods on US stocks/ETFs and
crypto daily data, and reports which methods work alone, which work best
together, and in which markets and regimes. It is a research tool, not a live
trading bot.

- Design: `docs/superpowers/specs/2026-11-20-swing-research-bot-design.md`
- Current plan: `docs/superpowers/plans/2026-11-20-swing-bot-phase1-solo-report.md`

## Architecture (keep these layers separate)

1. **Data** (`src/swingbot/data/`): Yahoo Finance (stocks/ETFs), Binance public API (crypto) → `data/raw/{market}/{symbol}.parquet`.
2. **Signals** (`src/swingbot/signals/`): each method is a registered pure function `bars -> bool Series`. Signals never trade.
3. **Outcomes** (`src/swingbot/outcomes.py`): forward returns from the **next bar's open**, net of costs, plus excess vs benchmark (SPY / BTCUSDT).
4. **Regimes** (`src/swingbot/regimes.py`): trend, volatility, market and ADX labels known at close of day t.
5. **Analysis** (`src/swingbot/analysis/`): solo edge, then pairs/conditional edge, then model attribution. Everything is broken down by market × segment × regime.
6. **Report** (`src/swingbot/report/`): static HTML in `reports/`.

## Non-negotiable rules

- **No lookahead.** A signal or regime label at day t uses only data up to the close of t. Entry is the open of t+1. Every new signal must pass `test_no_lookahead`.
- **Holdout is locked.** Data from 2024-01-01 onward loads only with `allow_holdout=True`, which is logged to `data/holdout_log.txt`. Never use it while exploring or tuning.
- **Costs always on:** stocks 0.10%, crypto 0.30% per round trip (`config.COSTS`).
- **Statistics:** date-clustered, overlap-aware (HAC) tests; Benjamini–Hochberg q-values; rows under 100 occurrences (30 per slice) are flagged `insufficient`, never ranked as winners.
- **Fixed textbook parameters** for every method until a later phase explicitly adds tuning.
- Claude builds and reviews; Claude never makes trading decisions or gives investment advice.
- Tests never hit the network.

## Commands

```powershell
.venv\Scripts\activate
pytest -v
python scripts/download.py --market all       # stock | crypto | all, --limit N for a quick run
python scripts/build_panel.py --market all
python scripts/analyze.py --horizon 10
```

## Conventions

- Bars schema: `DatetimeIndex` named `date` (tz-naive, sorted, unique), float columns `open, high, low, close, volume`.
- Signal column names are the registry names; they are stable IDs used by every later phase, so never rename them.
- Indicators are hand-written in `src/swingbot/indicators.py` (no TA-Lib).
- `data/` and `reports/` are gitignored.
- Windows: if a dependency has no Python 3.15 wheel, use a Python 3.12 venv via `uv`.

## Working with the user

- Ask as many clarifying questions as useful; don't ask for approval to proceed.
- Be blunt about overfitting and weak results. A method that fails is a valid finding.
