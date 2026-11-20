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


def run(assets: list[AssetMeta], fetch: Callable[[AssetMeta], pd.DataFrame] = default_fetch) -> list[str]:
    """Fetch, check and save each asset. Returns log messages (SKIP ... for failures)."""
    msgs: list[str] = []
    report: list[dict] = []
    for a in assets:
        try:
            df = fetch(a)
        except Exception as e:  # noqa: BLE001 - one bad symbol must not stop the run
            msgs.append(f"SKIP {a.symbol}: {type(e).__name__}: {e}")
            continue
        if df is None or df.empty:
            msgs.append(f"SKIP {a.symbol}: empty frame")
            continue
        issues = check_bars(df, a.market)
        store.save_bars(a.symbol, a.market, df)
        report.append({"symbol": a.symbol, "issues": "; ".join(issues)})
        msgs.append(f"OK {a.symbol}: {len(df)} bars" + (f" ({len(issues)} issues)" if issues else ""))
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report, columns=["symbol", "issues"]).to_csv(config.DATA_DIR / "quality_report.csv", index=False)
    pd.DataFrame([a.__dict__ for a in assets]).to_parquet(config.DATA_DIR / "universe_meta.parquet")
    return msgs
