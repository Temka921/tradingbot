"""CLI: download stock/crypto bars. Core logic lives in swingbot.data.download."""
import argparse

from swingbot.data.download import run
from swingbot.data.universe import crypto_universe, stock_universe, with_benchmark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--market", choices=["stock", "crypto", "all"], default="all")
    p.add_argument("--limit", type=int, default=None, help="max assets per market")
    args = p.parse_args()
    assets = []
    if args.market in ("stock", "all"):
        assets += with_benchmark(stock_universe(), "stock", args.limit)
    if args.market in ("crypto", "all"):
        assets += with_benchmark(crypto_universe(), "crypto", args.limit)
    for m in run(assets):
        print(m)


if __name__ == "__main__":
    main()
