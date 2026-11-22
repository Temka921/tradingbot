"""CLI: funding-rate carry F1/F2 (docs/superpowers/specs/2026-11-22-funding-carry-design.md).

Default (dev mode): dev (2020-2021) and validation (2022-2023) on holdout-locked spot, perp and
funding data; writes reports/funding_dev_<stamp>.txt|csv and the gate reports/funding_gate.json.

--final: the one-time 2024+ crypto holdout test, only for strategies that passed validation
(record data/holdout_use_crypto_funding.json, written before loading; a second run is refused).
"""
import argparse
from pathlib import Path

from swingbot import config
from swingbot.strategies import carry_run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true", help="one-time 2024+ crypto holdout test")
    p.add_argument("--reports", type=Path, default=config._REPO_ROOT / "reports")
    args = p.parse_args()
    if args.final:
        carry_run.run_final(args.reports)
    else:
        carry_run.run_dev(args.reports)


if __name__ == "__main__":
    main()
