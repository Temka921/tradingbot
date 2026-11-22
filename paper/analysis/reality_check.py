"""White's (2000) Reality Check for the best of the 41 signals, per universe and split.

For each signal k a daily series d_k(t) is built on every date of the split: the signal's
same-day mean excess return minus the same-day mean of all assets (the edge used in the paper),
and 0 on dates when the signal does not fire. The statistic is V = max_k sqrt(T) * mean(d_k).
Its null distribution comes from the stationary bootstrap (Politis and Romano, 1994) with mean
block length 2h = 20 days, B = 1000, seed 0, recentred at the sample means. Pre-2024 data only.
Writes paper/data/reality_check.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from swingbot import config
from swingbot.analysis.pairs import build_context
from swingbot.analysis.views import universe_views

OUT = Path(__file__).resolve().parents[1] / "data"
H, B, SEED = 10, 1000, 0


def edge_matrix(panel: pd.DataFrame, split: str) -> tuple[np.ndarray, list[str]]:
    ctx = build_context(panel, horizon=H, split=split)
    names, cols = [], []
    for s, (_, direction) in ctx.info.items():
        v, m = ctx.vals[direction], ctx.fires[s] & ~np.isnan(ctx.vals[direction])
        cnt = np.bincount(ctx.date_code[m], minlength=ctx.n_dates_total)
        tot = np.bincount(ctx.date_code[m], weights=v[m], minlength=ctx.n_dates_total)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(cnt > 0, tot / np.maximum(cnt, 1) - ctx.base[direction], 0.0)
        names.append(s)
        cols.append(np.nan_to_num(d))
    return np.column_stack(cols), names


def stationary_bootstrap_index(T: int, mean_block: float, rng) -> np.ndarray:
    idx = np.empty(T, dtype=np.int64)
    p = 1.0 / mean_block
    idx[0] = rng.integers(T)
    for t in range(1, T):
        idx[t] = rng.integers(T) if rng.random() < p else (idx[t - 1] + 1) % T
    return idx


def reality_check(D: np.ndarray) -> tuple[float, float, int]:
    T = D.shape[0]
    mean = D.mean(axis=0)
    v = np.sqrt(T) * mean.max()
    rng = np.random.default_rng(SEED)
    exceed = 0
    for _ in range(B):
        idx = stationary_bootstrap_index(T, 2 * H, rng)
        vb = np.sqrt(T) * (D[idx].mean(axis=0) - mean).max()
        exceed += vb >= v
    return float(v), (exceed + 1) / (B + 1), int(mean.argmax())


def main():
    panels = {m: pd.read_parquet(config.DATA_DIR / "panel" / f"{m}.parquet") for m in ("stock", "crypto")}
    views = universe_views(panels)
    del panels
    rows = []
    for view, panel in views.items():
        for split in ("discovery", "validation"):
            D, names = edge_matrix(panel, split)
            v, p, k = reality_check(D)
            rows.append({"view": view, "split": split, "n_signals": len(names), "n_dates": D.shape[0],
                         "best_signal": names[k], "best_mean_daily_edge_bps": float(D[:, k].mean() * 1e4),
                         "rc_stat": v, "rc_p": p})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "reality_check.csv", index=False)


if __name__ == "__main__":
    main()
