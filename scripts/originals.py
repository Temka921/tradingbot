"""CLI: original strategies C1-C3, S1, S2 (docs/superpowers/specs/2026-11-21-original-strategies-design.md).

Default (dev mode): crypto discovery/validation on the holdout-locked panel and US S1/S2
discovery/validation; writes reports/originals_dev_<stamp>.txt|csv, the C3 direction choice
and the validation gate reports/originals_gate.json.

--final: the one-time final tests, only for strategies that passed validation. Crypto uses the
2024+ holdout (record data/holdout_use_crypto_originals.json, written before loading; a second
run is refused). Stocks use Japan and Europe full history.
"""
import argparse
from pathlib import Path

from swingbot import config
from swingbot.strategies import run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true", help="one-time final tests (crypto holdout, Japan, Europe)")
    p.add_argument("--reports", type=Path, default=config._REPO_ROOT / "reports")
    args = p.parse_args()
    if args.final:
        run.run_final(args.reports)
    else:
        run.run_dev(args.reports)


if __name__ == "__main__":
    main()
