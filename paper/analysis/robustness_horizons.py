"""Robustness of the single-signal results to the holding period (5, 10, 20 trading days).

Uses only pre-2024 panels (holdout locked). Writes paper/data/robustness_horizons.csv and
paper/data/robustness_rsi14.csv.
"""
from pathlib import Path

import pandas as pd

from swingbot import config
from swingbot.analysis.solo import solo_table
from swingbot.analysis.views import universe_views
from swingbot.report.build import add_verdicts

OUT = Path(__file__).resolve().parents[1] / "data"


def main():
    panels = {m: pd.read_parquet(config.DATA_DIR / "panel" / f"{m}.parquet") for m in ("stock", "crypto")}
    views = universe_views(panels)
    del panels
    counts, rsi = [], []
    for view, panel in views.items():
        for h in (5, 10, 20):
            for split in ("discovery", "validation"):
                t = add_verdicts(solo_table(panel, horizon=h, split=split))
                vc = t["verdict"].value_counts()
                counts.append({"view": view, "horizon": h, "split": split,
                               "candidates": int(vc.get("candidate", 0)),
                               "significant_negative": int(vc.get("significant negative edge (avoid)", 0)),
                               "candidate_signals": ";".join(t.loc[t.verdict == "candidate", "signal"])})
                r = t.set_index("signal").loc["rsi14_oversold"]
                rsi.append({"view": view, "horizon": h, "split": split, "edge_bps": r.edge_vs_all_days * 1e4,
                            "q": r.q, "mean_ret_bps": r.mean_ret * 1e4, "verdict": r.verdict})
                print(view, h, split, counts[-1]["candidates"], counts[-1]["significant_negative"], flush=True)
    pd.DataFrame(counts).to_csv(OUT / "robustness_horizons.csv", index=False)
    pd.DataFrame(rsi).to_csv(OUT / "robustness_rsi14.csv", index=False)


if __name__ == "__main__":
    main()
