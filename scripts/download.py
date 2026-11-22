"""CLI: download stock/crypto bars. Core logic lives in swingbot.data.download."""
import argparse

from swingbot.data import foreign, futures
from swingbot.data.download import run
from swingbot.data.universe import crypto_universe, stock_universe, with_benchmark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--market", choices=["stock", "crypto", "all", "japan", "europe", "fng", "futures"], default="all")
    p.add_argument("--limit", type=int, default=None, help="max assets per market")
    args = p.parse_args()
    if args.market in ("japan", "europe"):
        for m in foreign.download_market(args.market):
            print(m)
        return
    if args.market == "fng":
        s = foreign.fetch_fng()
        foreign.save_fng(s)
        print(f"OK fng: {len(s)} days")
        return
    if args.market == "futures":  # USDT perps of the crypto universe: funding + perp klines (not part of "all")
        syms = futures.futures_universe()
        for s in syms[: args.limit] if args.limit else syms:
            try:
                print(futures.download_symbol(s))
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {s}: {e}")
        return
    assets = []
    if args.market in ("stock", "all"):
        assets += with_benchmark(stock_universe(), "stock", args.limit)
    if args.market in ("crypto", "all"):
        assets += with_benchmark(crypto_universe(), "crypto", args.limit)
    for m in run(assets):
        print(m)


if __name__ == "__main__":
    main()
