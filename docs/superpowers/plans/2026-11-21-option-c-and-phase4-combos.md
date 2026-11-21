# Option C Data + Phase 4 (Combos) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ETFs and crypto trustworthy primary results (≈100 ETFs; crypto with delisted coins, point-in-time liquidity filter, delisting exits; stocks labelled suspect) and add Phase 4: pair lift, conditional edge, triples, a redundancy map and a combo report.

**Architecture:** The data layer gains a Binance archive fetcher and an all-pairs crypto universe. The panel gains `delisted` and `eligible` columns. A new `analysis/views.py` splits panels into the `etf`, `crypto` and `stock` views that every analysis and report iterates over. Phase 4 lives in `analysis/pairs.py` (pairs, triples, regime slices) and `analysis/redundancy.py`, rendered by `report/combos.py` into a second HTML report.

**Tech Stack:** Python 3.15, pandas 3, numpy, statsmodels, requests, plotly (CDN), jinja2, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-swing-research-bot-design.md` (read the "Decisions after Phase 1 (2026-11-21)" section first). Also obey `CLAUDE.md`.

## Global Constraints

- No lookahead: any label at day t (eligibility included) uses data up to the close of t only; entry is the open of t+1.
- Holdout locked: `store.load_bars(..., allow_holdout=False)` everywhere in this plan.
- Costs: `config.COSTS` (stock 0.0010, crypto 0.0030) per round trip, always on.
- Stats: date-clustered HAC tests (`stats.hac_test`), BH q-values per table, `MIN_N = 100`, `MIN_N_SLICE = 30`, `n_dates >= max(min_n, 5*horizon)` to be sufficient. Insufficient rows are never winners.
- Signal names are stable IDs; never rename them.
- Tests never hit the network: monkeypatch `requests.get`.
- Fixed textbook parameters; no tuning.
- Constants this plan pins (put them in `config.py`): `DELIST_GAP_DAYS = 7`, `ELIGIBLE_TOP_N = {"crypto": 100}`, `LIQUIDITY_WINDOW = 30`, `REDUNDANT_JACCARD = 0.5`, `MAX_TRIPLES = 500`.
- View names and order: `("etf", "crypto", "stock")`; primary views `etf`, `crypto`; `stock` is suspect.

## Review Focus

1. **A delisted coin whose last bar falls inside the horizon:** its forward return must exit at the last close, not be NaN. NaN silently re-creates survivorship (Task 3).
2. **The archive returns microsecond timestamps** (Binance spot files from 2025 on): dates must still parse to the right day (Task 2).
3. **A coin that is never in the top 100:** its rows must never reach any statistic. A view with zero eligible rows for a signal gives an insufficient row, not a crash (Tasks 4 and 6).
4. **A pair where A never fires without B** (A-only is empty): lift is NaN and the row is insufficient or redundant, never a synergy (Task 6).
5. **The old 20-ETF rows still in `universe/stocks.csv`:** ETFs must come only from `etfs.csv`, and no symbol may appear twice in the stock universe (Task 1).

---

### Task 1: ETF universe (~100 ETFs, asset-class segment)

**Files:**
- Modify: `universe/etfs.csv` (rewrite), `universe/stocks.csv` (drop the ETF rows), `src/swingbot/data/universe.py` (`stock_universe`, `_etfs`)
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `stock_universe() -> list[AssetMeta]` = non-ETF rows of `stocks.csv` + every row of `etfs.csv` (`is_etf=True`, `cap_bucket="etf"`, `sector` = the asset class from the csv), with no duplicate symbols. The `stocks.csv` cache never contains ETFs again.

- [ ] **Step 1: Write the failing tests**

```python
def test_stock_universe_etfs_only_from_etfs_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    pd.DataFrame([("AAA", "stock", "Tech", "large", False, False),
                  ("SPY", "stock", "broad", "etf", True, False)],   # stale ETF row in the cache
                 columns=["symbol", "market", "sector", "cap_bucket", "is_etf", "is_major"]
                 ).to_csv(tmp_path / "stocks.csv", index=False)
    pd.DataFrame({"symbol": ["SPY", "TLT"], "sector": ["us_equity", "bonds"]}).to_csv(tmp_path / "etfs.csv", index=False)
    out = universe.stock_universe()
    syms = [a.symbol for a in out]
    assert syms == ["AAA", "SPY", "TLT"]
    tlt = out[2]
    assert tlt.is_etf and tlt.cap_bucket == "etf" and tlt.sector == "bonds"

def test_committed_etf_list():
    df = pd.read_csv(universe.UNIVERSE_DIR / "etfs.csv")
    assert 95 <= len(df) <= 110 and df["symbol"].is_unique and "SPY" in set(df["symbol"])
    assert set(df["sector"]) == {"us_equity", "factor", "us_sector", "international", "bonds",
                                 "commodity", "currency", "realestate"}
    stocks = pd.read_csv(universe.UNIVERSE_DIR / "stocks.csv")
    assert not stocks["is_etf"].any() and not set(stocks["symbol"]) & set(df["symbol"])
```

- [ ] **Step 2: Run them, expect FAIL** — `pytest tests/test_data.py -k "etf" -v`

- [ ] **Step 3: Implement**
  - `stock_universe()`: on a cache hit, return the non-ETF rows of `stocks.csv` followed by `_etfs()`, dropping later duplicates. On a cache miss, write only the Wikipedia rows (no ETFs) to the cache, then append `_etfs()`.
  - Rewrite `universe/etfs.csv` with exactly these rows (symbol,sector):
    - us_equity: SPY QQQ IWM DIA MDY IJH IJR IVV VTI OEF RSP IWB IWV
    - factor: IWD IWF IWN IWO IVE IVW DVY SDY VIG VYM USMV MTUM QUAL SPLV
    - us_sector: XLK XLF XLE XLV XLI XLY XLP XLU XLB XLRE XLC SMH IBB IYT ITB XHB XRT XME XOP KRE KBE GDX OIH IGV ITA IYZ
    - international: EEM EFA EWJ EWZ EWG EWU EWC EWA EWH EWT EWY EWW FXI INDA EWS EWP EWQ EWI EWL VGK VWO ILF EZA
    - bonds: TLT IEF SHY LQD HYG AGG TIP BND EMB MUB IEI
    - commodity: GLD SLV USO DBC DBA UNG GDXJ
    - currency: UUP FXE FXY
    - realestate: VNQ IYR RWX
  - Drop the ETF rows from `universe/stocks.csv`.
  - Fix any existing test that assumed ETFs live in `stocks.csv`.

- [ ] **Step 4: Run the full suite, expect PASS** — `pytest -q`

- [ ] **Step 5: Commit** — `git commit -am "feat(universe): ~100 ETFs by asset class, ETFs only from etfs.csv"`

---

### Task 2: Binance archive fetcher and all-pairs crypto universe

**Files:**
- Modify: `src/swingbot/data/binance.py`, `src/swingbot/data/universe.py`, `src/swingbot/data/download.py` (`default_fetch`)
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `binance.ARCHIVE_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"`
  - `binance.ARCHIVE_FILE_URL = "https://data.binance.vision/"`
  - `binance.list_archive_symbols(quote: str = "USDT") -> list[str]`: every spot symbol under prefix `data/spot/monthly/klines/` that ends with `quote`, sorted. Paginates the S3 XML (`IsTruncated`, `NextMarker`; request params `delimiter="/"`, `prefix=...`, `marker=...`). Parse with `xml.etree.ElementTree`, namespace-agnostic.
  - `binance.fetch_binance_archive(symbol: str, start: str = "2017-01-01") -> pd.DataFrame`: lists `data/spot/monthly/klines/{symbol}/1d/` (keys ending in `.zip`), downloads each monthly zip from `ARCHIVE_FILE_URL + key`, parses the headerless CSV (columns 0–5 = open_time, open, high, low, close, volume) and returns the standard bars schema. open_time is in ms, or in µs when it is above 1e14. Rows before `start` are dropped.
  - `binance.fetch_crypto(symbol: str) -> pd.DataFrame`: `fetch_binance(symbol)`; on `requests.HTTPError` with status 400 (delisted or invalid symbol), returns `fetch_binance_archive(symbol)`.
  - `download.default_fetch` uses `fetch_crypto` for crypto.
  - `universe.crypto_universe(n: int | None = None) -> list[AssetMeta]`:
    - The cache is `universe/crypto.csv`. On a miss, it uses the union of `list_archive_symbols()` and the live `/api/v3/ticker/24hr` USDT symbols.
    - It applies `excluded_crypto_base(base, listed=all bases)`. That function now also excludes `FIAT_BASES = {"EUR","GBP","AUD","BRL","TRY","RUB","NGN","UAH","ZAR","JPY","BIDR","IDRT","BVND"}`, and `STABLE_BASES` gains `"PAX","USDSB","UST","SUSD","EURI","AEUR","VAI"`.
    - BTCUSDT and ETHUSDT come first, the rest sorted by symbol; there is no volume ranking (eligibility is point-in-time in the panel, see Task 4).
    - `n` truncates when given.

- [ ] **Step 1: Write the failing tests**

```python
def test_list_archive_symbols_paginates(monkeypatch):
    pages = [
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>true</IsTruncated>'
        '<NextMarker>data/spot/monthly/klines/ETHBTC/</NextMarker>'
        '<CommonPrefixes><Prefix>data/spot/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>'
        '<CommonPrefixes><Prefix>data/spot/monthly/klines/ETHBTC/</Prefix></CommonPrefixes></ListBucketResult>',
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>'
        '<CommonPrefixes><Prefix>data/spot/monthly/klines/LUNAUSDT/</Prefix></CommonPrefixes></ListBucketResult>']
    calls = []
    def get(url, params=None, timeout=None, **k):
        calls.append(params.get("marker"))
        return FakeText(pages[len(calls) - 1])
    monkeypatch.setattr(binance.requests, "get", get)
    assert binance.list_archive_symbols() == ["BTCUSDT", "LUNAUSDT"]
    assert calls == [None, "data/spot/monthly/klines/ETHBTC/"]

def test_fetch_binance_archive_ms_and_us(monkeypatch):
    # one zip in ms (2021-01-01, 2021-01-02), one in µs (2025-01-01)
    ...  # build zips in memory with zipfile; the S3 listing returns both keys plus a .CHECKSUM key
    df = binance.fetch_binance_archive("LUNAUSDT")
    assert list(df.index) == list(pd.to_datetime(["2021-01-01", "2021-01-02", "2025-01-01"]))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"] and df.index.name == "date"

def test_fetch_crypto_falls_back_on_400(monkeypatch):
    def api(*a, **k): raise requests.HTTPError(response=FakeStatus(400))
    monkeypatch.setattr(binance, "fetch_binance", api)
    monkeypatch.setattr(binance, "fetch_binance_archive", lambda s, start="2017-01-01": "ARCHIVE")
    assert binance.fetch_crypto("LUNAUSDT") == "ARCHIVE"
    # a 500 is re-raised, not swallowed
    def api500(*a, **k): raise requests.HTTPError(response=FakeStatus(500))
    monkeypatch.setattr(binance, "fetch_binance", api500)
    with pytest.raises(requests.HTTPError):
        binance.fetch_crypto("BTCUSDT")

def test_crypto_universe_includes_delisted_and_excludes_fiat(monkeypatch, tmp_path):
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    monkeypatch.setattr(universe.binance, "list_archive_symbols",
                        lambda quote="USDT": ["BTCUSDT", "LUNAUSDT", "EURUSDT", "USTUSDT", "BTCUPUSDT"])
    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: FakeResp(
        [{"symbol": "ETHUSDT", "quoteVolume": "1"}, {"symbol": "ARBUSDT", "quoteVolume": "1"}]))
    out = [a.symbol for a in universe.crypto_universe()]
    assert out == ["BTCUSDT", "ETHUSDT", "ARBUSDT", "LUNAUSDT"]
```

Write the `...` body: two zips built with `zipfile.ZipFile(io.BytesIO(), "w")`. `requests.get` is patched to return the S3 listing XML for the listing URL (keys: the 2 zips plus a `.zip.CHECKSUM`) and the zip bytes (`.content`) for file URLs. Add small `FakeText(text)` (`.text`, `.content`, `raise_for_status()`) and `FakeStatus(code)` (`.status_code`) helpers next to `FakeResp`. Replace the old volume-ranking tests (`test_crypto_universe_filters`, `test_crypto_universe_top_n_and_cache`, `test_crypto_universe_exclusions_and_order`) with versions that assert the new ordering. Keep `test_real_coins_ending_in_b_up_down_kept` and the `excluded_crypto_base` cases.

- [ ] **Step 2: Run them, expect FAIL** — `pytest tests/test_data.py -v`

- [ ] **Step 3: Implement** the interfaces above. `fetch_binance_archive` reuses the `fetch_binance` normalisation: drop duplicate dates (keep last), sort, float columns, tz-naive `date` index.

- [ ] **Step 4: Run `pytest -q`, expect PASS.** `test_committed_crypto_cache_is_clean` may still read the old cache; it must keep passing (the controller regenerates the cache in Task 9).

- [ ] **Step 5: Commit** — `git commit -am "feat(data): Binance archive fetch for delisted pairs, all-pairs crypto universe"`

---

### Task 3: Delisting-aware outcomes and the `delisted` panel column

**Files:**
- Modify: `src/swingbot/outcomes.py`, `src/swingbot/panel.py`, `src/swingbot/config.py`
- Test: `tests/test_outcomes.py`, `tests/test_regimes_panel.py`

**Interfaces:**
- Produces:
  - `compute_outcomes(df, bench, market, delisted: bool = False) -> pd.DataFrame`. With `delisted=True`, let `L` = index of the last bar. For each row t with `t+1 <= L` and `t+h > L`:
    - `gross_h = close[L] / open[t+1] - 1`;
    - `bench_gross_h` exits at the benchmark close on the first benchmark date ≥ `date[L]`;
    - `ret_*`, `excess_*` follow from those;
    - `atr_exit_*` time-exits at `close[L]` when the 20-bar window runs past `L` (stop/target checks over the bars that exist);
    - `mfe_h`/`mae_h` stay NaN.

    With `delisted=False`, output is unchanged.
  - `config.DELIST_GAP_DAYS = 7`, `config.RELIST_GAP_DAYS = 5`.
  - **Relisted symbols are different assets.** Binance reuses tickers: `LUNAUSDT` is Terra LUNA until 2022-05-13 (crashed to 0.00005), then a new coin from 2022-05-31 (~$5). Both the API and the archive return the two as one series. Add `panel.split_relistings(symbol: str, df: pd.DataFrame, max_gap_days: int = config.RELIST_GAP_DAYS) -> list[tuple[str, pd.DataFrame]]`:
    - split at every gap between consecutive dates that is longer than `max_gap_days`;
    - return the segments in time order;
    - the last segment keeps `symbol`; earlier segments are named `f"{symbol}@{segment_first_date:%Y-%m-%d}"`;
    - no gap returns `[(symbol, df)]`.

    `build_panel` applies it to crypto only, right after loading. Each segment is its own asset: same meta row, its own signals and outcomes, and its own `delisted` flag, so the dead first LUNA segment is delisted and exits at its last close. The benchmark is looked up by the last segment's name. Never split stocks: holidays make multi-day gaps normal.
  - Panel: an asset is delisted when its last date (after the holdout cut) is earlier than `market_last - DELIST_GAP_DAYS` days, where `market_last` = the latest last date over all loaded assets of that market. The panel gets a bool column `delisted` (per asset, constant), and `compute_outcomes` is called with it.

- [ ] **Step 1: Write the failing tests**

```python
def test_delisted_exit_at_last_close():
    df = bars_from_closes([10, 10, 10, 10, 5])          # last bar L = 4; open == close
    bench = bars_from_closes([100] * 5)
    out = compute_outcomes(df, bench, "crypto", delisted=True)
    # t=1: entry open[2]=10, horizon 5 would end at t=6 > L -> exits at close[4]=5
    assert out.loc[df.index[1], "gross_5"] == pytest.approx(-0.5)
    assert out.loc[df.index[1], "ret_long_5"] == pytest.approx(-0.5 - 0.003)
    assert out.loc[df.index[1], "excess_long_5"] == pytest.approx(-0.5 - 0.003)
    assert np.isnan(out.loc[df.index[4], "gross_5"])    # no t+1 bar: no entry
    assert out["atr_exit_long"].notna().sum() >= 0      # does not crash on a short series

def test_not_delisted_tail_still_nan():
    df = bars_from_closes([10, 10, 10, 10, 5])
    out = compute_outcomes(df, bars_from_closes([100] * 5), "crypto")
    assert out["gross_5"].isna().all()

def test_panel_marks_delisted(capsys):
    btc = rand_bars(900, seed=1, start="2020-01-01", freq="D")       # runs to the holdout cut
    dead = rand_bars(400, seed=2, start="2020-01-01", freq="D")      # ends early 2021
    for s, d in (("BTCUSDT", btc), ("LUNAUSDT", dead)):
        store.save_bars(s, "crypto", d)
    _meta([("BTCUSDT", "crypto", "crypto", "major", False, True),
           ("LUNAUSDT", "crypto", "crypto", "alt", False, False)])
    out = panel.build_panel("crypto")
    assert out.groupby("asset")["delisted"].first().to_dict() == {"BTCUSDT": False, "LUNAUSDT": True}
    last = out[out.asset == "LUNAUSDT"].iloc[-3]
    assert not np.isnan(last["gross_10"])               # exits at the last close instead of NaN

def test_split_relistings():
    d1 = pd.date_range("2021-01-01", periods=30, freq="D")
    d2 = pd.date_range(d1[-1] + pd.Timedelta(days=18), periods=20, freq="D")
    df = bars_from_closes([1.0] * 50); df.index = pd.DatetimeIndex(list(d1) + list(d2), name="date")
    segs = panel.split_relistings("LUNAUSDT", df)
    assert [s for s, _ in segs] == ["LUNAUSDT@2021-01-01", "LUNAUSDT"]
    assert [len(x) for _, x in segs] == [30, 20]
    assert panel.split_relistings("BTCUSDT", df.iloc[:30])[0][0] == "BTCUSDT"

def test_panel_relisted_symbol_becomes_two_assets(capsys):
    # BTCUSDT continuous; LUNAUSDT with an 18-day gap -> two assets, the first delisted
    ...
    d = out.groupby("asset")["delisted"].first().to_dict()
    assert d["LUNAUSDT@2020-01-01"] is True and d["LUNAUSDT"] is False
```

The relisting tests sit in `tests/test_regimes_panel.py` (`bars_from_closes`: import it or copy a minimal version there).

Use the `bars_from_closes` helper in `tests/test_outcomes.py`, or add one there (open = close = given, high/low ±1%).

- [ ] **Step 2: Run them, expect FAIL.**
- [ ] **Step 3: Implement.** Vectorise the delisted fill: `exit_idx = np.minimum(t + h, L)`. Use the same index for the benchmark lookup.
- [ ] **Step 4: Run `pytest -q`, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat(outcomes): delisted assets exit at last close; panel delisted flag"`

---

### Task 4: Point-in-time liquidity eligibility

**Files:**
- Modify: `src/swingbot/regimes.py`, `src/swingbot/panel.py`, `src/swingbot/config.py`
- Test: `tests/test_regimes_panel.py`

**Interfaces:**
- Produces:
  - `regimes.liquidity_eligible(closes: pd.DataFrame, volumes: pd.DataFrame, top_n: int, window: int = config.LIQUIDITY_WINDOW) -> pd.DataFrame` (bool, same shape: dates × assets):
    - dollar volume = close × volume;
    - trailing `window`-day median with `min_periods=window`, computed per asset on its own bars before aligning to the union of dates;
    - per date, rank descending with `method="first"`; eligible when rank ≤ `top_n`; NaN → False.
  - `config.ELIGIBLE_TOP_N = {"crypto": 100}`, `config.LIQUIDITY_WINDOW = 30`.
  - Panel: bool column `eligible`. For markets in `ELIGIBLE_TOP_N` it comes from `liquidity_eligible` over all loaded assets of the market (computed once, like `cross_sectional_signals`); otherwise it is True.

- [ ] **Step 1: Write the failing tests**

```python
def test_liquidity_eligible_ranks_and_warmup():
    d = pd.date_range("2020-01-01", periods=40, freq="D")
    closes = pd.DataFrame({"A": 1.0, "B": 1.0, "C": 1.0}, index=d)
    vols = pd.DataFrame({"A": 300.0, "B": 200.0, "C": 100.0}, index=d)
    e = regimes.liquidity_eligible(closes, vols, top_n=2, window=30)
    assert not e.iloc[:29].any().any()                 # warm-up: never eligible
    assert e.iloc[29:]["A"].all() and e.iloc[29:]["B"].all() and not e.iloc[29:]["C"].any()

def test_liquidity_eligible_no_lookahead():
    rng = np.random.default_rng(0)
    d = pd.date_range("2020-01-01", periods=120, freq="D")
    closes = pd.DataFrame(rng.uniform(1, 2, (120, 5)), index=d, columns=list("ABCDE"))
    vols = pd.DataFrame(rng.uniform(1, 1000, (120, 5)), index=d, columns=list("ABCDE"))
    full = regimes.liquidity_eligible(closes, vols, top_n=2, window=30)
    cut = regimes.liquidity_eligible(closes.iloc[:80], vols.iloc[:80], top_n=2, window=30)
    pd.testing.assert_frame_equal(full.iloc[:80], cut)

def test_panel_eligible_column():
    # 3 crypto assets with flat volumes 3e6/2e6/1e6, ELIGIBLE_TOP_N patched to {"crypto": 2}
    ...
    assert out.loc[out.asset == "CUSDT", "eligible"].sum() == 0
    assert out.loc[out.asset == "BTCUSDT", "eligible"].iloc[40:].all()
    # stock panels: eligible is all True
```

Write the `...` from the pattern of `test_build_panel_skips_stale_crypto`: `rand_bars` with a fixed `volume` column, `monkeypatch.setattr(config, "ELIGIBLE_TOP_N", {"crypto": 2})`.

- [ ] **Step 2: Run them, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `pytest -q`, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat(panel): point-in-time top-100 liquidity eligibility"`

---

### Task 5: Universe views and the option-C solo report

**Files:**
- Create: `src/swingbot/analysis/views.py`
- Modify: `src/swingbot/report/build.py`, `src/swingbot/report/templates/report.html.j2`
- Test: `tests/test_analysis.py`, `tests/test_report.py`

**Interfaces:**
- Produces:
  - `views.VIEWS = ("etf", "crypto", "stock")`, `views.PRIMARY = ("etf", "crypto")`, `views.SUSPECT = ("stock",)`
  - `views.SEGMENT_KEY = {"etf": "sector", "crypto": "cap_bucket", "stock": "sector"}`
  - `views.universe_views(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]`:
    - keys in `VIEWS` order, only the non-empty views;
    - `etf` = stock-panel rows with `is_etf`, `stock` = stock-panel rows without, `crypto` = the crypto panel;
    - rows with `eligible == False` are dropped (when the column exists);
    - adds a categorical column `view`;
    - raises `ValueError` if a panel's `market` column contains another market.
  - `build_report(panels, out_path, horizon=10, quality=None, as_of=None) -> Path` keeps its signature but iterates `universe_views(panels)`:
    - one leaderboard, heatmap columns (`view | regime`), segment table and coverage row per view;
    - the separate "ETF control" table is gone (ETF is a full view);
    - the stock view's leaderboard heading carries the text `SUSPECT_LABEL = "Suspect: survivorship-biased (today's index members only)"`;
    - `heatmap_matrix` groups by `view` instead of `market`.
  - CSV names become `solo_{kind}_{view}_{stamp}.csv`.
  - `SURVIVORSHIP` text is rewritten to say:
    - (a) single stocks are today's index members, so they are suspect; it keeps the `{stock_mean}` number;
    - (b) crypto includes delisted Binance pairs, uses a point-in-time top-100 liquidity filter, and delisted coins exit at their last close (an approximation: post-delisting value may be lower);
    - (c) ETFs: currently listed funds only, so closed funds are missing (a smaller bias).

    Drop the `ETF_NOTE` paragraph.

- [ ] **Step 1: Write the failing tests**

```python
def test_universe_views_split_and_eligible():
    stock = make_market_panel("stock", assets=("SPY", "AAA"))
    stock["is_etf"] = stock["asset"] == "SPY"
    crypto = make_market_panel("crypto", assets=("BTCUSDT", "XUSDT"))
    crypto["eligible"] = crypto["asset"] == "BTCUSDT"
    v = universe_views({"stock": stock, "crypto": crypto})
    assert list(v) == ["etf", "crypto", "stock"]
    assert set(v["etf"]["asset"]) == {"SPY"} and set(v["stock"]["asset"]) == {"AAA"}
    assert set(v["crypto"]["asset"]) == {"BTCUSDT"}
    assert (v["etf"]["view"] == "etf").all()

def test_report_views_and_suspect_label(tmp_path):
    stock = make_market_panel("stock", assets=("SPY", "QQQ", "AAA", "BBB"), seed=1)
    stock["is_etf"] = stock["asset"].isin(["SPY", "QQQ"])
    panels = {"stock": stock, "crypto": make_market_panel("crypto", seed=2)}
    html = build_report(panels, tmp_path / "r.html", horizon=10).read_text(encoding="utf-8")
    assert SUSPECT_LABEL in html
    assert html.index("etf") < html.index(SUSPECT_LABEL)
    assert "etf | " in html and "stock | " in html and "crypto | " in html     # heatmap columns
    assert list(tmp_path.glob("solo_leaderboard_etf_*.csv"))
```

Update existing report tests that reference the ETF control table, `ETF_NOTE` or the old CSV names so that they assert the new behaviour (the view-based CSV names; ETFs are a view, not a control).

- [ ] **Step 2: Run them, expect FAIL.**
- [ ] **Step 3: Implement.** `universe_mean_excess` is computed on the `stock` view.
- [ ] **Step 4: Run `pytest -q`, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat(report): etf/crypto primary views, stocks labelled suspect"`

---

### Task 6: Pair analysis (lift, conditional edge)

**Files:**
- Create: `src/swingbot/analysis/pairs.py`
- Modify: `src/swingbot/analysis/stats.py` (add `diff_test`), `src/swingbot/config.py`
- Test: `tests/test_pairs.py` (new)

**Interfaces:**
- Consumes: `solo.clip_bounds`, `solo.clip_values`, `stats.hac_test`, `stats.bh_qvalues`, `stats.effective_df`, `stats.min_dates_for_test`, `SIGNAL_INFO`.
- Produces:
  - `stats.diff_test(a: TestResult, b: TestResult, horizon: int) -> TestResult`: `mean = a.mean - b.mean`, `se = sqrt(a.se² + b.se²)`, `t = mean/se`, two-sided p with df = `min(effective_df(a.n_dates, h), effective_df(b.n_dates, h))`, `n = a.n + b.n`, `n_dates = min(a.n_dates, b.n_dates)`. se/t/p are NaN if either se is NaN. (Treating the two series as independent overstates the variance when they are positively correlated, so the test is conservative.)
  - `@dataclass class EdgeContext`: `horizon: int`, `date_code: np.ndarray` (int per row), `n_dates_total: int`, `vals: dict[str, np.ndarray]` (clipped metric per direction), `base: dict[str, np.ndarray]` (per-date all-rows mean per direction, indexed by date code), `fires: dict[str, np.ndarray]` (bool per signal), `info: dict[str, tuple[str, str]]`.
  - `build_context(panel, horizon=10, metric="excess", split="discovery", signal_info=None, rows_mask=None) -> EdgeContext`: rows = the split (and `rows_mask` when given). Clipping is pooled exactly as in `solo_table`.
  - `@dataclass class GroupStats`: `n: int`, `n_dates: int`, `mean: float` (date-weighted mean of the metric), `edge: TestResult` (date-matched edge vs `base`, via `hac_test` on the date-sorted d series).
  - `group_stats(ctx, mask: np.ndarray, direction: str) -> GroupStats`: per-date sums and counts with `np.bincount` on `date_code`; NaN metric values ignored.
  - `pair_table(panel, horizon=10, metric="excess", split="discovery", signal_info=None) -> pd.DataFrame`, one row per unordered pair (a < b in registry order) with the **same direction**. Columns, in order:
    - identity: `a, b, family_a, family_b, direction, jaccard`;
    - counts: `n_ab, n_dates_ab, n_a_only, n_b_only`;
    - combo: `mean_ab, edge_ab, t_ab, p_ab, q_ab`;
    - singles and conditional edges: `edge_a_only, edge_b_only, cond_a_given_b, cond_b_given_a`;
    - lift: `lift, t_lift, p_lift, q_lift`;
    - flags: `insufficient, verdict`.

    Definitions:
    - `jaccard` = |A∧B| / |A∨B| over the split rows;
    - `cond_a_given_b = edge_ab - edge_b_only`;
    - `lift = edge_ab - max(edge_a_only, edge_b_only)`;
    - `t_lift`/`p_lift` = `diff_test(AB, best_only)`, where best_only is the only-group with the larger edge;
    - `q_ab` and `q_lift` are BH over the table's rows separately.
  - `insufficient` = `n_ab < MIN_N` or `n_dates_ab < max(MIN_N, 5h)`, or either only-group is empty or has NaN edge.
  - `add_pair_verdicts(t) -> pd.DataFrame` (sets `verdict`):
    - `"insufficient"`;
    - `"synergy"`: q_lift < 0.10, lift > 0, mean_ab > 0;
    - `"conflict"`: q_lift < 0.10, lift < 0;
    - `"redundant"`: not significant and jaccard ≥ `config.REDUNDANT_JACCARD` (0.5); if insufficient only because an only-group is empty, also redundant;
    - otherwise `"not significant"`.

    The order of `PAIR_VERDICTS` is the same as this list, and `pair_table` calls `add_pair_verdicts`.

- [ ] **Step 1: Write the failing tests** — build a synthetic panel (pattern of `make_market_panel` in `tests/test_report.py`: 400 dates × 20 assets, `excess_long_10` ~ N(0, 0.02), discovery split) with a 3-signal `signal_info` `{"A": ("meanrev","long"), "B": ("trend","long"), "S": ("trend","short")}`:

```python
def test_diff_test_arithmetic():
    a = TestResult(0.03, 0.01, 3, 0.0, 100, 200); b = TestResult(0.01, 0.01, 1, 0.3, 100, 200)
    r = diff_test(a, b, 10)
    assert r.mean == pytest.approx(0.02) and r.se == pytest.approx(np.sqrt(2) * 0.01)
    assert r.t == pytest.approx(0.02 / (np.sqrt(2) * 0.01))
    assert np.isnan(diff_test(a, TestResult(0.0, np.nan, np.nan, np.nan, 1, 1), 10).p)

def test_synergy_detected():
    p = synth(plant_ab=+0.03)          # A, B fire independently ~15%; rows with both get +3%
    t = pair_table(p, horizon=10, signal_info=INFO)
    r = t[(t.a == "A") & (t.b == "B")].iloc[0]
    assert r.verdict == "synergy" and r.lift > 0.02 and r.q_lift < 0.10
    assert r.cond_a_given_b == pytest.approx(r.edge_ab - r.edge_b_only)

def test_no_lift_when_a_carries_it():
    p = synth(plant_a=+0.03)           # every A row +3%, B adds nothing
    r = pair_table(p, 10, signal_info=INFO).query("a == 'A' and b == 'B'").iloc[0]
    assert r.verdict != "synergy" and abs(r.lift) < 0.01

def test_conflict_detected():
    p = synth(plant_a=+0.03, plant_b=+0.03, plant_ab=-0.06)
    r = pair_table(p, 10, signal_info=INFO).query("a == 'A' and b == 'B'").iloc[0]
    assert r.verdict == "conflict"

def test_same_direction_only_and_columns():
    t = pair_table(synth(), 10, signal_info=INFO)
    assert set(zip(t.a, t.b)) == {("A", "B")}            # S (short) never pairs with longs
    assert list(t.columns) == PAIR_COLS

def test_a_never_alone_is_redundant_not_synergy():
    p = synth(); p["A"] = p["B"]                           # identical firings
    r = pair_table(p, 10, signal_info=INFO).iloc[0]
    assert r.verdict == "redundant" and r.jaccard == 1.0 and np.isnan(r.lift)

def test_ineligible_rows_never_counted():
    p = synth(plant_ab=+0.03); p["eligible"] = False
    from swingbot.analysis.views import universe_views
    v = universe_views({"crypto": p.assign(market="crypto")})
    assert "crypto" not in v                               # nothing eligible -> no view
```

`synth(**plants)` is a test helper in `tests/test_pairs.py` with a fixed seed. Planted effects are added to `excess_long_10` on the matching rows.

- [ ] **Step 2: Run them, expect FAIL.**
- [ ] **Step 3: Implement.** Per-date means via `np.bincount(date_code[m], weights=v[m], minlength=n_dates_total)` / counts. d = daily − base on dates with count > 0, in date order, then `hac_test(d_series, horizon, n)`.
- [ ] **Step 4: Run `pytest -q`, expect PASS.** Also time `pair_table` on a 4,000-date × 300-asset synthetic panel with 41 signals; it must finish in under 120 s. Report the time in the commit message body.
- [ ] **Step 5: Commit** — `git commit -am "feat(analysis): pair lift and conditional edge with conservative HAC diff test"`

---

### Task 7: Triples, regime slices and the redundancy map

**Files:**
- Modify: `src/swingbot/analysis/pairs.py`
- Create: `src/swingbot/analysis/redundancy.py`
- Test: `tests/test_pairs.py`

**Interfaces:**
- Consumes: `build_context`, `group_stats`, `diff_test`, `pair_table` output (Task 6).
- Produces:
  - `triple_table(panel, pairs: pd.DataFrame, horizon=10, metric="excess", split="discovery", signal_info=None) -> pd.DataFrame`.
    - Candidate triples: same direction, all 3 constituent pairs present in `pairs`, not insufficient, with `lift > 0`. Keep at most `config.MAX_TRIPLES` (500), ranked by the smallest constituent lift, descending.
    - Columns: `a, b, c, direction, n_abc, n_dates_abc, mean_abc, edge_abc, best_pair, edge_best_pair_only, lift, t_lift, p_lift, q_lift, insufficient, verdict`.
    - `edge_best_pair_only` is the largest edge among (A∧B∧¬C, A∧C∧¬B, B∧C∧¬A); `lift = edge_abc − edge_best_pair_only`, tested with `diff_test`.
    - Same insufficiency rule and verdicts as pairs (via `add_pair_verdicts`; triples have no jaccard, so redundant never applies).
    - An empty candidate set returns an empty frame with those columns.
  - `pair_regime_table(panel, pairs: pd.DataFrame, horizon=10, regime_col="trend_regime", split="discovery", signal_info=None) -> pd.DataFrame`.
    - For each pair with verdict `"synergy"` and each value of `regime_col` (excluding `"unknown"`): build a context restricted to that regime (`rows_mask`), so the baseline is same-regime and same-day.
    - Columns: `a, b, direction, regime, n_ab, n_dates_ab, mean_ab, edge_ab, t_ab, p_ab, q_ab, insufficient` (q is BH within the table; insufficient uses `MIN_N_SLICE`).
    - No synergy pairs → an empty frame with the columns.
  - `redundancy.jaccard_matrix(panel, split="discovery", signal_info=None) -> pd.DataFrame` (signals × signals, diagonal 1.0; NaN where neither fires).
  - `redundancy.redundancy_groups(jac: pd.DataFrame, threshold: float = config.REDUNDANT_JACCARD) -> list[list[str]]`: connected components of the graph with edges where jaccard ≥ threshold, size ≥ 2, each sorted in registry order, groups sorted by first member.

- [ ] **Step 1: Write the failing tests**

```python
def test_triple_synergy():
    p = synth3(plant_abc=+0.04)        # three long signals A, B, C; +4% only when all three fire
    pairs = pair_table(p, 10, signal_info=INFO3)
    t = triple_table(p, pairs, 10, signal_info=INFO3)
    # pairs carry part of the effect, so lifts > 0 and the triple is evaluated
    assert len(t) == 1 and t.iloc[0].lift > 0 and t.iloc[0].verdict in ("synergy", "not significant")

def test_triple_empty_when_no_positive_pairs():
    p = synth3()
    t = triple_table(p, pair_table(p, 10, signal_info=INFO3).assign(lift=-1.0), 10, signal_info=INFO3)
    assert t.empty and "lift" in t.columns

def test_pair_regime_table_same_regime_baseline():
    p = synth(plant_ab=+0.03); p["trend_regime"] = np.where(p.index % 2 == 0, "up", "down")
    pairs = pair_table(p, 10, signal_info=INFO)
    r = pair_regime_table(p, pairs, 10, signal_info=INFO)
    assert set(r.regime) == {"up", "down"} and (r.edge_ab > 0.02).all()

def test_redundancy_groups():
    jac = pd.DataFrame([[1, .9, 0], [.9, 1, .1], [0, .1, 1]], index=list("XYZ"), columns=list("XYZ"), dtype=float)
    assert redundancy_groups(jac, 0.5) == [["X", "Y"]]

def test_jaccard_matrix_duplicate_signals():
    p = synth(); p["B"] = p["A"]
    j = jaccard_matrix(p, signal_info=INFO)
    assert j.loc["A", "B"] == 1.0 and j.loc["A", "A"] == 1.0
```

Use the real-registry duplicate case as a sanity check in a comment: `bb_lower_close` and `zscore_below_2` must end up in one group on real data (checked in Task 9).

- [ ] **Step 2: Run them, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `pytest -q`, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat(analysis): triples, regime slices for synergy pairs, redundancy map"`

---

### Task 8: Combo report and `analyze.py`

**Files:**
- Create: `src/swingbot/report/combos.py`, `src/swingbot/report/templates/combos.html.j2`
- Modify: `scripts/analyze.py`
- Test: `tests/test_report.py` (or new `tests/test_combo_report.py`)

**Interfaces:**
- Consumes: `universe_views`, `PRIMARY`, `SUSPECT`, `SUSPECT_LABEL`, `pair_table`, `triple_table`, `pair_regime_table`, `jaccard_matrix`, `redundancy_groups`, plus `CDN_URL`, `_fmt`/`_bps` from `report.build`.
- Produces:
  - `build_combo_report(panels: dict[str, pd.DataFrame], out_path: Path, horizon: int = 10, as_of: str | None = None) -> Path`. For each view:
    - verdict counts for pairs and triples;
    - a pair table showing all synergy and conflict rows plus the 30 non-insufficient rows with the highest `edge_ab`, with columns a, b, direction, verdict, n_ab, dates, mean_ab, edge_ab, q_ab, edge_a_only, edge_b_only, lift, q_lift, jaccard (bps formatting for edges and means);
    - the triple table;
    - the regime table;
    - a Jaccard heatmap (plotly, signals × signals);
    - the redundancy group list.
  - Full CSVs: `combo_{pairs|triples|pair_regimes|jaccard}_{view}_{stamp}.csv`. The stock view carries `SUSPECT_LABEL`.
  - A caveat box at the top. It reuses the solo report's survivorship text, and adds:
    - lift is measured against the better single-only edge;
    - q is per table;
    - discovery split only;
    - a synergy here is a hypothesis for validation, not a result.
  - `scripts/analyze.py` builds both reports (`solo_report_{stamp}.html`, `combo_report_{stamp}.html`) and prints both paths.

- [ ] **Step 1: Write the failing test**

```python
def test_combo_report_renders(tmp_path):
    stock = make_market_panel("stock", assets=("SPY", "QQQ", "AAA", "BBB"), seed=1)
    stock["is_etf"] = stock["asset"].isin(["SPY", "QQQ"])
    panels = {"stock": stock, "crypto": make_market_panel("crypto", seed=2)}
    out = build_combo_report(panels, tmp_path / "combo_report_x.html", horizon=10, as_of="x")
    html = out.read_text(encoding="utf-8")
    assert SUSPECT_LABEL in html and CDN_URL in html
    for v in ("etf", "crypto", "stock"):
        assert (tmp_path / f"combo_pairs_{v}_x.csv").exists()
    assert "lift" in html.lower() and "hypothesis" in html.lower()
```

Also extend `test_analyze_script_missing_market` (or add a test) so that `analyze.main()` writes both reports from tiny panels in `tmp_path`.

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `pytest -q`, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat(report): combo report (pairs, triples, regimes, redundancy)"`

---

### Task 9: Data refresh and real run (controller, network)

Run by the controller, not a subagent. Nothing here is unit-tested; the outputs are checked by eye and by the final review.

- [ ] Regenerate `universe/crypto.csv`: delete it, run `python -c "from swingbot.data.universe import crypto_universe; print(len(crypto_universe()))"`, then make sure `test_committed_crypto_cache_is_clean` passes and commit the csv.
- [ ] `python scripts/download.py --market all` (background; crypto falls back to the archive for delisted pairs). Record the OK/SKIP counts.
- [ ] `python scripts/build_panel.py --market all`. Record assets, rows, delisted count and eligible share.
- [ ] `python scripts/analyze.py --horizon 10`. Check that:
  - `bb_lower_close` and `zscore_below_2` share a redundancy group;
  - the ETF view has about 100 assets;
  - crypto eligible rows ≈ 100 per date once enough coins exist.
- [ ] Whole-branch review, then a report to the user with the numbers.
