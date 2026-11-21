"""Validation gate: re-test discovery hypotheses on the untouched validation split.

Hypotheses of a view are its discovery solo "candidate" rows and discovery pair
"synergy" rows. Each is re-measured on the validation split of the same view and
marked a finalist only if it still holds there. Only finalists of a primary view
(``views.PRIMARY``) may go to the holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingbot.analysis.pairs import combo_sufficient, pair_table
from swingbot.analysis.solo import solo_table
from swingbot.analysis.stats import bh_qvalues
from swingbot.analysis.views import PRIMARY
from swingbot.report.build import add_verdicts

GATE_COLS = ["view", "kind", "rule", "direction", "disc_edge", "disc_q", "val_n", "val_n_dates",
             "val_mean", "val_mean_ret", "val_edge", "val_lift", "val_p", "val_q", "finalist",
             "holdout_eligible", "reason"]
Q_GATE = 0.10
NAN = float("nan")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=GATE_COLS)


def _reason(r: dict) -> str:
    """First failing condition, in the fixed order; a NaN compared quantity fails its condition."""
    if not r["_sufficient"]:
        return "insufficient on validation"
    if not r["val_edge"] > 0:
        return "edge <= 0"
    if not r["val_q"] < Q_GATE:
        return "q >= 0.10"
    if not r["val_mean"] > 0:
        return "mean excess <= 0"
    if not r["val_mean_ret"] > 0:
        return "mean return <= 0"
    if r["kind"] == "pair" and not r["val_lift"] > 0:
        return "lift <= 0"
    return ""


def validation_gate(view_panel: pd.DataFrame, view: str, horizon: int = 10,
                    signal_info: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Gate one view's discovery hypotheses on its validation split (columns ``GATE_COLS``).

    Hypotheses: ``add_verdicts(solo_table(split="discovery"))`` rows with verdict
    "candidate" (``kind="solo"``, ``rule`` = signal) and ``pair_table(split="discovery")``
    rows with verdict "synergy" (``kind="pair"``, ``rule`` = "a+b").

    - ``disc_edge``/``disc_q``: from the discovery row, ``edge_vs_all_days``/``q`` for
      solo, ``edge_ab``/``q_lift`` for pairs (q_lift is what made the pair a synergy).
    - ``val_*``: the same signal's / pair's row in the validation table. Solo:
      n_all, n_dates, mean_all (excess), mean_ret, edge_vs_all_days, p; ``val_lift``
      is NaN. Pair: n_ab, n_dates_ab, mean_ab, mean_ret_ab, edge_ab, lift, p_ab.
    - ``val_q``: Benjamini-Hochberg over ``val_p`` of this view's gated rows only.
    - Sufficient on validation: the solo table's ``insufficient`` flag; for pairs
      ``combo_sufficient(n_ab, n_dates_ab)`` (an empty only-group alone does not make
      the combo insufficient; its NaN lift then fails "lift <= 0").
    - ``finalist``: sufficient, val_edge > 0, val_q < 0.10, val_mean > 0,
      val_mean_ret > 0 and, for pairs, val_lift > 0. NaN fails a condition.
      ``reason`` is "" for finalists, else the first failing condition in that order.
    - ``holdout_eligible`` = finalist and ``view in PRIMARY``.

    No hypotheses gives an empty frame with ``GATE_COLS``. ``pair_table`` (the
    expensive call) is computed once per split, the validation one only when
    there is a pair hypothesis.
    """

    disc_solo = add_verdicts(solo_table(view_panel, horizon, split="discovery", signal_info=signal_info))
    cands = disc_solo[disc_solo["verdict"] == "candidate"]
    disc_pairs = pair_table(view_panel, horizon, split="discovery", signal_info=signal_info)
    syn = disc_pairs[disc_pairs["verdict"] == "synergy"]
    if cands.empty and syn.empty:
        return _empty()

    rows: list[dict] = []
    if not cands.empty:
        val_solo = solo_table(view_panel, horizon, split="validation",
                              signal_info=signal_info).set_index("signal")
        for _, d in cands.iterrows():
            v = val_solo.loc[d["signal"]]
            rows.append({
                "kind": "solo", "rule": d["signal"], "direction": d["direction"],
                "disc_edge": d["edge_vs_all_days"], "disc_q": d["q"],
                "val_n": int(v["n_all"]), "val_n_dates": int(v["n_dates"]),
                "val_mean": v["mean_all"], "val_mean_ret": v["mean_ret"],
                "val_edge": v["edge_vs_all_days"], "val_lift": NAN, "val_p": v["p"],
                "_sufficient": not bool(v["insufficient"]),
            })
    if not syn.empty:
        val_pairs = pair_table(view_panel, horizon, split="validation",
                               signal_info=signal_info).set_index(["a", "b"])
        for _, d in syn.iterrows():
            v = val_pairs.loc[(d["a"], d["b"])]
            rows.append({
                "kind": "pair", "rule": f"{d['a']}+{d['b']}", "direction": d["direction"],
                "disc_edge": d["edge_ab"], "disc_q": d["q_lift"],
                "val_n": int(v["n_ab"]), "val_n_dates": int(v["n_dates_ab"]),
                "val_mean": v["mean_ab"], "val_mean_ret": v["mean_ret_ab"],
                "val_edge": v["edge_ab"], "val_lift": v["lift"], "val_p": v["p_ab"],
                "_sufficient": bool(combo_sufficient(int(v["n_ab"]), int(v["n_dates_ab"]), horizon)),
            })

    q = bh_qvalues(np.array([r["val_p"] for r in rows], dtype="float64"))
    for r, qi in zip(rows, q):
        r["val_q"] = float(qi)
        r["reason"] = _reason(r)
        r["finalist"] = r["reason"] == ""
        r["holdout_eligible"] = r["finalist"] and view in PRIMARY
        r["view"] = view
    t = pd.DataFrame(rows)[GATE_COLS]
    for c in ("val_n", "val_n_dates"):
        t[c] = t[c].astype(np.int64)
    for c in ("disc_edge", "disc_q", "val_mean", "val_mean_ret", "val_edge", "val_lift", "val_p", "val_q"):
        t[c] = t[c].astype("float64")
    for c in ("finalist", "holdout_eligible"):
        t[c] = t[c].astype(bool)
    return t
