"""CLI: build data/panel/{market}.parquet. Core logic lives in swingbot.panel."""
import argparse

from swingbot import config
from swingbot.panel import build_panel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--market", choices=["stock", "crypto", "all"], default="all")
    args = p.parse_args()
    markets = ["stock", "crypto"] if args.market == "all" else [args.market]
    for m in markets:
        df = build_panel(m)
        path = config.DATA_DIR / "panel" / f"{m}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"{m}: {len(df)} rows, {df['asset'].nunique()} assets, {df.shape[1]} columns -> {path}")
        del df


if __name__ == "__main__":
    main()
