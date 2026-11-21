"""CLI: the one-time holdout run for the eligible finalists of a gate CSV.

Runs at most once (data/holdout_used.json); see swingbot.holdout.
"""
import argparse
from pathlib import Path

from swingbot import config
from swingbot import holdout


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--finalists", type=Path, required=True, help="reports/finalists_<stamp>.csv")
    p.add_argument("--horizon", type=int, default=10, choices=list(config.HORIZONS))
    args = p.parse_args()
    out = holdout.run_holdout(args.finalists, horizon=args.horizon)
    if out is not None:
        print(out)


if __name__ == "__main__":
    main()
