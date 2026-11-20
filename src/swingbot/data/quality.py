import pandas as pd

MAX_GAP_DAYS = {"stock": 5, "crypto": 1}


def check_bars(df: pd.DataFrame, market: str) -> list[str]:
    issues: list[str] = []
    if df.empty:
        return ["empty frame"]
    idx = pd.DatetimeIndex(df.index)
    n_dup = int(idx.duplicated().sum())
    if n_dup:
        issues.append(f"duplicate dates: {n_dup}")
    uniq = idx[~idx.duplicated()].sort_values()
    gaps = pd.Series(uniq[1:] - uniq[:-1]).dt.days if len(uniq) > 1 else pd.Series([], dtype=int)
    n_gap = int((gaps > MAX_GAP_DAYS[market]).sum())
    if n_gap:
        issues.append(f"gap > {MAX_GAP_DAYS[market]} days: {n_gap} (max {int(gaps.max())})")
    prices = df[["open", "high", "low", "close"]]
    n_nonpos = int((prices <= 0).any(axis=1).sum())
    if n_nonpos:
        issues.append(f"non-positive prices: {n_nonpos} bars")
    n_hl = int((df["high"] < df["low"]).sum())
    if n_hl:
        issues.append(f"high < low: {n_hl} bars")
    zero_frac = float((df["volume"] == 0).mean())
    if zero_frac > 0.05:
        issues.append(f"zero-volume bars: {zero_frac:.1%}")
    if market == "stock":
        moves = df["close"].pct_change().abs()
        n_big = int((moves > 0.5).sum())
        if n_big:
            issues.append(f"split suspicion: {n_big} bars with >50% move")
    return issues
