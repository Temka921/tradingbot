"""Japan/Europe universes and downloads, glitch filter, Crypto Fear & Greed index."""
import io
import re
from datetime import datetime, timezone

import pandas as pd
import requests

from swingbot import config
from swingbot.data import download, store, universe
from swingbot.data.universe import UA, AssetMeta
from swingbot.data.yahoo import fetch_yahoo

NIKKEI_URL = "https://en.wikipedia.org/wiki/Nikkei_225"
DAX_URL = "https://en.wikipedia.org/wiki/DAX"
CAC_URL = "https://en.wikipedia.org/wiki/CAC_40"
FTSE_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"

_CODE_COLS = ("code", "ticker", "ticker symbol", "symbol", "epic", "tidm", "stock code", "securities code")
_NAME_COLS = ("company", "name", "company name", "constituent", "security")


def _today() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).date())


def _find_col(columns, candidates) -> str | None:
    norm = {re.sub(r"\[.*?\]", "", str(c)).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    return None


def _constituents(url: str) -> pd.DataFrame:
    """Return (code, name) rows from the first table on the page with a ticker/code column."""
    resp = requests.get(url, headers=UA, timeout=30)
    resp.raise_for_status()
    for t in pd.read_html(io.StringIO(resp.text)):
        code = _find_col(t.columns, _CODE_COLS)
        if code is None:
            continue
        name = _find_col(t.columns, _NAME_COLS)
        out = pd.DataFrame({"code": t[code].astype(str).str.strip(),
                            "name": t[name].astype(str).str.strip() if name else t[code].astype(str)})
        return out[out["code"].notna() & (out["code"] != "") & (out["code"].str.lower() != "nan")]
    raise ValueError(f"constituents table not found at {url}")


def _norm_name(name: str) -> str:
    n = re.sub(r"[^a-z0-9 ]", "", name.lower())
    n = re.sub(r"\b(se|ag|sa|plc|nv|group|holding|holdings|co|corp|inc|ltd|the)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def _suffix(code: str, suffix: str) -> str:
    code = code.upper()
    if suffix == ".T":
        digits = re.sub(r"\D", "", code.split(".")[0])
        return digits.zfill(4) + ".T"
    if code.endswith(suffix):
        code = code[: -len(suffix)]
    if suffix == ".L":
        code = code.rstrip(".").replace(".", "-")
    return code + suffix


def japan_universe() -> pd.DataFrame:
    cache = universe.UNIVERSE_DIR / "japan.csv"
    if cache.exists():
        return pd.read_csv(cache, keep_default_na=False)
    t = _constituents(NIKKEI_URL)
    syms = list(dict.fromkeys(_suffix(c, ".T") for c in t["code"]))
    df = pd.DataFrame({"symbol": syms, "market": "japan", "home_index": config.HOME_INDEX["japan"]})
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def europe_universe() -> pd.DataFrame:
    cache = universe.UNIVERSE_DIR / "europe.csv"
    if cache.exists():
        return pd.read_csv(cache, keep_default_na=False)
    rows, seen_names, seen_syms = [], set(), set()
    for url, suffix in ((DAX_URL, ".DE"), (CAC_URL, ".PA"), (FTSE_URL, ".L")):
        for code, name in _constituents(url).itertuples(index=False):
            key, sym = _norm_name(name), _suffix(code, suffix)
            if key in seen_names or sym in seen_syms:
                continue
            seen_names.add(key)
            seen_syms.add(sym)
            rows.append((sym, "europe", config.HOME_INDEX[suffix]))
    df = pd.DataFrame(rows, columns=["symbol", "market", "home_index"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def download_market(market: str) -> list[str]:
    """Download every universe symbol plus its home indices. Bars dated today (UTC) or later are dropped."""
    uni = {"japan": japan_universe, "europe": europe_universe}[market]()
    symbols = list(dict.fromkeys(list(uni["symbol"]) + list(uni["home_index"])))
    today = _today()
    msgs: list[str] = []
    for sym in symbols:
        asset = AssetMeta(sym, market, "", "", False, False)
        try:
            df = download.fetch_with_retry(lambda a: fetch_yahoo(a.symbol), asset)
            if df is None or df.empty:
                msgs.append(f"SKIP {sym}: empty frame")
                continue
            df = df[df.index < today]
            if df.empty:
                msgs.append(f"SKIP {sym}: empty frame")
                continue
            store.save_bars(sym, market, df)
        except Exception as e:  # noqa: BLE001 - one bad symbol must not stop the run
            msgs.append(f"SKIP {sym}: {type(e).__name__}: {e}")
            continue
        msgs.append(f"OK {sym}: {len(df)} bars")
    return msgs


def _glitch_rows(close) -> list[int]:
    bad = []
    for i in range(1, len(close)):
        prev = close[i - 1]
        if not prev or prev != prev or close[i] != close[i]:
            continue
        if abs(close[i] / prev - 1) <= 0.60:
            continue
        move = close[i] - prev
        sign = 1.0 if move > 0 else -1.0
        for k in (1, 2):
            if i + k < len(close) and (close[i] - close[i + k]) * sign > 0.5 * abs(move):
                bad.append(i)
                break
    return bad


def glitch_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Remove data glitches: |close/prev - 1| > 0.60 and a reversal of more than half the move
    within the next 2 bars. Glitch OHLC values are replaced by the previous (cleaned) bar's values;
    detection repeats (max 3 passes) so multi-day spikes are fully cleaned. Returns a copy."""
    out = df.copy()
    cols = [out.columns.get_loc(c) for c in ("open", "high", "low", "close") if c in out.columns]
    for _ in range(3):
        bad = _glitch_rows(out["close"].to_numpy(dtype=float))
        if not bad:
            break
        for i in bad:  # ascending, so a run of glitches propagates the last good bar
            out.iloc[i, cols] = out.iloc[i - 1, cols].to_numpy()
    return out


def fetch_fng() -> pd.Series:
    resp = requests.get(FNG_URL, headers=UA, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    idx = pd.to_datetime([int(d["timestamp"]) for d in data], unit="s", utc=True).tz_localize(None).normalize()
    s = pd.Series([float(d["value"]) for d in data], index=idx.rename("date"), name="fng", dtype=float)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _fng_path():
    return config.DATA_DIR / "raw" / "sentiment" / "fng.parquet"


def save_fng(s: pd.Series) -> None:
    p = _fng_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    s.rename("fng").to_frame().to_parquet(p)


def load_fng() -> pd.Series:
    return pd.read_parquet(_fng_path())["fng"]
