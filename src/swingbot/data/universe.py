import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from swingbot import config
from swingbot.data import binance

UNIVERSE_DIR: Path = config._REPO_ROOT / "universe"
UA = {"User-Agent": "Mozilla/5.0 (compatible; swingbot-research/0.1)"}
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
STABLE_BASES = {"USDC", "FDUSD", "TUSD", "DAI", "USDP", "BUSD", "EUR", "USDE",
                "USD1", "PYUSD", "RLUSD", "USDS", "XUSD", "RWUSD", "U"}
# Non-core exclusions: wrapped/duplicate BTC and tokenized gold (not independent crypto assets).
WRAPPED_OR_COMMODITY_BASES = {"WBTC", "PAXG", "XAUT"}
# Binance tokenized stocks/ETFs trade as <TICKER>B. A suffix heuristic would drop real coins
# (ARB, SHIB, BNB, TRB...), so this is an EXPLICIT list taken from Binance live USDT pairs
# (2026-11); new tokenized listings must be added here. Real B-ending coins are never listed.
TOKENIZED_EQUITY_BASES = {
    "SOXLB", "NVDAB", "SPCXB", "MUB", "GOOGLB", "QQQB", "CRCLB", "MSTRB", "INTCB", "ANTB",
    "SNDKB", "KORUB", "SNXXB", "TQQQB", "SOXSB", "SKHYB", "AMDB", "NBISB", "MRVLB", "SPYB",
    "TSLAB", "EWYB", "AVGOB", "DRAMB", "BNCB", "AMZNB", "COINB", "BEB", "TSMB", "BMNRB",
    "METAB", "MSFTB", "INTWB", "LITEB", "IRENB", "AAOIB", "AAPLB", "SQQQB", "HOODB", "MUUB",
    "CBRSB", "AXTIB", "PDDB", "CRWVB", "PLTRB", "ARMB", "QCOMB", "BABAB", "COHRB", "MVLLB",
    "MRNAB", "ALABB", "WDCB", "RKLBB", "AMATB", "ORCLB", "SHAZB", "KSTRB", "APPB", "SMHB",
    "GLWB", "NFLXB", "DELLB", "UVXYB", "ASTSB", "STXB", "AGPUB", "CRDOB", "GEVB", "SKDDB",
    "GMEB", "SMCIB", "ASMLB", "CYPHB", "BSPB", "USARB",
}


LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")
MAJORS = {"BTC", "ETH"}


@dataclass(frozen=True)
class AssetMeta:
    symbol: str
    market: str
    sector: str
    cap_bucket: str
    is_etf: bool
    is_major: bool


def excluded_crypto_base(base: str, listed: frozenset[str] | set[str] = frozenset()) -> bool:
    """`listed` = all USDT-quoted bases on the exchange. Binance leveraged tokens are
    <LISTED BASE> + UP/DOWN/BULL/BEAR (BTCUP), so a suffix only counts when the stem is listed;
    this keeps JUP, SYRUP etc."""
    if not (base.isascii() and base.isalnum()):
        return True
    if base in STABLE_BASES or base.endswith("USD") or base in WRAPPED_OR_COMMODITY_BASES:
        return True
    if base in TOKENIZED_EQUITY_BASES:
        return True
    for suf in LEVERAGED_SUFFIXES:
        if base.endswith(suf) and base[: -len(suf)] in listed:
            return True
    return False


def _yahoo_ticker(t: str) -> str:
    return str(t).strip().replace(".", "-")


def _wiki_table(url: str, sym_cols: tuple[str, ...]) -> pd.DataFrame:
    resp = requests.get(url, headers=UA, timeout=30)
    resp.raise_for_status()
    for t in pd.read_html(io.StringIO(resp.text)):
        if any(c in t.columns for c in sym_cols) and "GICS Sector" in t.columns:
            return t
    raise ValueError(f"constituents table not found at {url}")


def _from_wiki(url: str, cap: str) -> list[AssetMeta]:
    t = _wiki_table(url, ("Symbol", "Ticker symbol", "Ticker"))
    col = next(c for c in ("Symbol", "Ticker symbol", "Ticker") if c in t.columns)
    return [
        AssetMeta(_yahoo_ticker(s), "stock", str(sec), cap, False, False)
        for s, sec in zip(t[col], t["GICS Sector"])
    ]


def _etfs() -> list[AssetMeta]:
    df = pd.read_csv(UNIVERSE_DIR / "etfs.csv")
    return [AssetMeta(r.symbol, "stock", r.sector or "broad", "etf", True, False) for r in df.itertuples()]


def _read_cache(path: Path) -> list[AssetMeta]:
    df = pd.read_csv(path, keep_default_na=False)
    return [
        AssetMeta(r.symbol, r.market, r.sector, r.cap_bucket, bool(r.is_etf), bool(r.is_major))
        for r in df.itertuples()
    ]


def _write_cache(path: Path, assets: list[AssetMeta]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([a.__dict__ for a in assets]).to_csv(path, index=False)


def stock_universe() -> list[AssetMeta]:
    cache = UNIVERSE_DIR / "stocks.csv"
    if cache.exists():
        return _read_cache(cache)
    assets = _from_wiki(SP500_URL, "large") + _from_wiki(SP400_URL, "mid") + _etfs()
    seen: set[str] = set()
    uniq = [a for a in assets if not (a.symbol in seen or seen.add(a.symbol))]
    _write_cache(cache, uniq)
    return uniq


def crypto_universe(n: int = 60) -> list[AssetMeta]:
    cache = UNIVERSE_DIR / "crypto.csv"
    if cache.exists():
        return _read_cache(cache)[:n]
    resp = requests.get(f"{binance.BASE_URL}/api/v3/ticker/24hr", timeout=30)
    resp.raise_for_status()
    rows = []
    tickers = resp.json()
    listed = {t["symbol"][:-4] for t in tickers if t["symbol"].endswith("USDT")}
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        base = sym[: -len("USDT")]
        if excluded_crypto_base(base, listed):
            continue
        rows.append((sym, base, float(t["quoteVolume"])))
    rows.sort(key=lambda r: r[2], reverse=True)
    assets = [
        AssetMeta(sym, "crypto", "crypto", "major" if base in MAJORS else "alt", False, base in MAJORS)
        for sym, base, _ in rows[:n]
    ]
    _write_cache(cache, assets)
    return assets


def with_benchmark(assets: list[AssetMeta], market: str, limit: int | None) -> list[AssetMeta]:
    """Benchmark first, then up to `limit` other assets (all if limit is None)."""
    bench = config.BENCHMARK[market]
    b = [a for a in assets if a.symbol == bench]
    if not b:
        raise ValueError(f"benchmark {bench} missing from {market} universe")
    rest = [a for a in assets if a.symbol != bench]
    return b + (rest[:limit] if limit else rest)
