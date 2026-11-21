"""CLI: build the finalists HTML report and gate CSV from data/panel/*.parquet."""
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from swingbot import config
from swingbot.report.finalists import build_finalists_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=10, choices=list(config.HORIZONS))
    args = p.parse_args()
    panels = {}
    for m in ("stock", "crypto"):
        path = config.DATA_DIR / "panel" / f"{m}.parquet"
        if path.exists():
            panels[m] = pd.read_parquet(path)
        else:
            print(f"SKIP {m}: {path} missing (run scripts/build_panel.py)")
    if not panels:
        raise SystemExit("no panels found")
    stamp = dt.date.today().isoformat()
    reports = Path(config.DATA_DIR).parent / "reports"
    html, _ = build_finalists_report(panels, reports / f"finalists_report_{stamp}.html",
                                     horizon=args.horizon, as_of=stamp)
    print(html)
    print(reports / f"finalists_{stamp}.csv")


if __name__ == "__main__":
    main()
