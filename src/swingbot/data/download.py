import time
from pathlib import Path
from typing import Callable

import pandas as pd

from swingbot import config
from swingbot.data import store
from swingbot.data.binance import fetch_binance
from swingbot.data.quality import check_bars
from swingbot.data.universe import AssetMeta
from swingbot.data.yahoo import fetch_yahoo


def default_fetch(asset: AssetMeta) -> pd.DataFrame:
    if asset.market == "stock":
        return fetch_yahoo(asset.symbol)
    return fetch_binance(asset.symbol)


_sleep = time.sleep  # patchable in tests
RETRY_ATTEMPTS = 3


def fetch_with_retry(fetch: Callable[[AssetMeta], pd.DataFrame], asset: AssetMeta) -> pd.DataFrame:
    """Call fetch up to 3 times, sleeping 1s then 2s between attempts."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return fetch(asset)
        except Exception:  # noqa: BLE001
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            _sleep(2**attempt)


def _upsert(path: Path, new: pd.DataFrame, read, write) -> None:
    if path.exists() and not new.empty:
        old = read(path)
        new = pd.concat([old, new], ignore_index=True)
    elif path.exists():
        return
    write(new.drop_duplicates(["symbol", "market"], keep="last").reset_index(drop=True))


def run(assets: list[AssetMeta], fetch: Callable[[AssetMeta], pd.DataFrame] = default_fetch) -> list[str]:
    """Fetch, check and save each asset. Returns log messages (SKIP ... for failures).

    Reports are upserted by (symbol, market) so partial runs accumulate.
    """
    msgs: list[str] = []
    report: list[dict] = []
    saved: list[AssetMeta] = []
    try:
        for a in assets:
            try:
                df = fetch_with_retry(fetch, a)
                if df is None or df.empty:
                    msgs.append(f"SKIP {a.symbol}: empty frame")
                    continue
                issues = check_bars(df, a.market)
                store.save_bars(a.symbol, a.market, df)
            except Exception as e:  # noqa: BLE001 - one bad symbol must not stop the run
                msgs.append(f"SKIP {a.symbol}: {type(e).__name__}: {e}")
                continue
            saved.append(a)
            report.append({"symbol": a.symbol, "market": a.market, "issues": "; ".join(issues)})
            msgs.append(f"OK {a.symbol}: {len(df)} bars" + (f" ({len(issues)} issues)" if issues else ""))
    finally:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        rep = pd.DataFrame(report, columns=["symbol", "market", "issues"])
        _upsert(config.DATA_DIR / "quality_report.csv", rep,
                lambda p: pd.read_csv(p, keep_default_na=False), lambda d: d.to_csv(config.DATA_DIR / "quality_report.csv", index=False))
        meta = pd.DataFrame([a.__dict__ for a in saved], columns=list(AssetMeta.__dataclass_fields__))
        _upsert(config.DATA_DIR / "universe_meta.parquet", meta, pd.read_parquet,
                lambda d: d.to_parquet(config.DATA_DIR / "universe_meta.parquet"))
    return msgs
