"""CLI: download stock/crypto bars. Core logic lives in swingbot.data.download."""
import argparse

from swingbot.data.download import run
from swingbot.data.universe import crypto_universe, stock_universe


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--market", choices=["stock", "crypto", "all"], default="all")
    p.add_argument("--limit", type=int, default=None, help="max assets per market")
    args = p.parse_args()
    assets = []
    if args.market in ("stock", "all"):
        s = stock_universe()
        assets += s[: args.limit] if args.limit else s
    if args.market in ("crypto", "all"):
        c = crypto_universe()
        assets += c[: args.limit] if args.limit else c
    for m in run(assets):
        print(m)


if __name__ == "__main__":
    main()
