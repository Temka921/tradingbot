"""Redundancy map: which signals fire on (nearly) the same rows.

``jaccard_matrix`` gives the pairwise firing overlap |A and B| / |A or B| over a
split's rows; ``redundancy_groups`` joins signals whose overlap reaches a
threshold into connected components (on real data ``bb_lower_close`` and
``zscore_below_2`` are expected to land in one group).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingbot import config
from swingbot.signals import SIGNAL_INFO


def jaccard_matrix(panel: pd.DataFrame, split: str | None = "discovery",
                   signal_info: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Signals x signals Jaccard overlap of firing rows within ``split``, in registry order.

    The diagonal is 1.0; off-diagonal cells are NaN where neither signal fires.
    """
    info = SIGNAL_INFO if signal_info is None else signal_info
    names = list(info)
    missing = [s for s in names if s not in panel.columns]
    if missing:
        raise KeyError(f"signal columns missing from panel: {missing}")
    sel = np.ones(len(panel), dtype=bool) if split is None else \
        (panel["split"] == split).to_numpy(dtype=bool)
    x = np.column_stack([panel[s].to_numpy(dtype=bool)[sel] for s in names]).astype(np.float64) \
        if names else np.zeros((int(sel.sum()), 0))
    inter = x.T @ x                                    # exact integer counts in float64
    cnt = np.diag(inter)
    union = cnt[:, None] + cnt[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(union > 0, inter / np.where(union > 0, union, 1.0), np.nan)
    np.fill_diagonal(jac, 1.0)
    return pd.DataFrame(jac, index=names, columns=names)


def redundancy_groups(jac: pd.DataFrame, threshold: float = config.REDUNDANT_JACCARD
                      ) -> list[list[str]]:
    """Connected components (size >= 2) of the graph with an edge where jaccard >= threshold.

    Members keep the matrix (registry) order; groups are ordered by their first member.
    NaN cells and the diagonal are never edges.
    """
    names = list(jac.index)
    m = jac.reindex(index=names, columns=names).to_numpy(dtype="float64")
    k = len(names)
    parent = list(range(k))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edge = (m >= threshold) | (m.T >= threshold)       # NaN compares False
    for i, j in zip(*np.nonzero(np.triu(edge, k=1))):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(i)
    out = [[names[i] for i in g] for g in groups.values() if len(g) >= 2]
    out.sort(key=lambda g: names.index(g[0]))
    return out
