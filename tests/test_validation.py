import numpy as np
import pandas as pd

from swingbot.analysis.stats import bh_qvalues
from swingbot.analysis.validation import GATE_COLS, validation_gate

INFO = {"A": ("meanrev", "long"), "B": ("trend", "long"), "S": ("trend", "short")}

DISC_START, VAL_START = "2015-01-01", "2019-01-01"


def synth(a_disc: float = 0.0, a_val: float = 0.0, a_val_ret: float | None = None,
          ab_disc: float = 0.0, ab_val: float = 0.0, fire: float = 0.15,
          n_dates: int = 300, n_assets: int = 20, seed: int = 7) -> pd.DataFrame:
    """A view panel with ``n_dates`` discovery dates then ``n_dates`` validation dates.

    ``a_*`` is added to excess_long_10 and ret_long_10 where A fires, ``ab_*`` where A and B
    both fire, in the discovery / validation split. ``a_val_ret`` (if given) replaces ``a_val``
    for ret_long_10 on validation, so excess and absolute return can disagree.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(DISC_START, periods=n_dates).append(pd.bdate_range(VAL_START, periods=n_dates))
    assets = [f"X{i}" for i in range(n_assets)]
    p = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"]).to_frame(index=False)
    n = len(p)
    for name in INFO:
        p[name] = rng.random(n) < fire
    for side in ("long", "short"):
        for h in (5, 10, 20):
            p[f"excess_{side}_{h}"] = rng.normal(0, 0.02, n)
            p[f"ret_{side}_{h}"] = rng.normal(0, 0.02, n)
    val = (p["date"] >= pd.Timestamp(VAL_START)).to_numpy()
    a, b = p["A"].to_numpy(), p["B"].to_numpy()
    for col in ("excess_long_10", "ret_long_10"):
        av = a_val if (col == "excess_long_10" or a_val_ret is None) else a_val_ret
        y = p[col].to_numpy().copy()
        y[a & ~val] += a_disc
        y[a & val] += av
        y[a & b & ~val] += ab_disc
        y[a & b & val] += ab_val
        p[col] = y
    p["market"] = "crypto"
    p["split"] = pd.Categorical(np.where(val, "validation", "discovery"))
    return p


def row(g: pd.DataFrame, rule: str) -> pd.Series:
    r = g[g["rule"] == rule]
    assert len(r) == 1, f"expected one {rule!r} row, got {len(r)}"
    return r.iloc[0]


def test_persistent_signal_is_finalist():
    p = synth(a_disc=0.01, a_val=0.01)
    g = validation_gate(p, "crypto", signal_info=INFO)
    assert list(g.columns) == GATE_COLS
    r = row(g, "A")
    assert r["kind"] == "solo" and r["direction"] == "long"
    assert bool(r["finalist"]) and r["reason"] == ""
    assert bool(r["holdout_eligible"])
    assert r["val_edge"] > 0 and r["val_q"] < 0.10 and r["val_mean"] > 0 and r["val_mean_ret"] > 0
    assert r["disc_edge"] > 0 and r["disc_q"] < 0.10
    assert np.isnan(r["val_lift"])
    assert r["val_n"] >= 100 and r["val_n_dates"] >= 100

    gs = validation_gate(p, "stock", signal_info=INFO)
    rs = row(gs, "A")
    assert bool(rs["finalist"]) and not bool(rs["holdout_eligible"])
    assert (gs["view"] == "stock").all()


def test_discovery_only_effect_fails_on_validation():
    g = validation_gate(synth(a_disc=0.01, a_val=0.0), "crypto", signal_info=INFO)
    r = row(g, "A")
    assert not bool(r["finalist"]) and not bool(r["holdout_eligible"])
    assert r["reason"] in {"edge <= 0", "q >= 0.10"}


def test_negative_absolute_return_not_finalist():
    g = validation_gate(synth(a_disc=0.01, a_val=0.01, a_val_ret=-0.01), "crypto", signal_info=INFO)
    r = row(g, "A")
    assert r["val_mean"] > 0 and r["val_mean_ret"] < 0
    assert not bool(r["finalist"]) and r["reason"] == "mean return <= 0"


def test_pair_hypothesis_gated():
    g = validation_gate(synth(ab_disc=0.02, ab_val=0.02, fire=0.35), "etf", signal_info=INFO)
    r = row(g, "A+B")
    assert r["kind"] == "pair" and r["direction"] == "long"
    assert r["val_lift"] > 0 and r["val_edge"] > 0
    assert bool(r["finalist"]) and r["reason"] == "" and bool(r["holdout_eligible"])
    # val_q is BH over this view's gated rows (solo and pair together)
    assert len(g) > 1
    np.testing.assert_allclose(g["val_q"], bh_qvalues(g["val_p"].to_numpy(dtype="float64")))


def test_pair_insufficient_on_validation_not_finalist():
    # synergy planted in discovery; validation has too few dates for A and B together
    p = synth(ab_disc=0.02, ab_val=0.02, fire=0.35)
    val = (p["split"] == "validation").to_numpy()
    kill = val & (p["date"] >= pd.bdate_range(VAL_START, periods=40)[-1]).to_numpy()
    p.loc[kill, "B"] = False
    g = validation_gate(p, "etf", signal_info=INFO)
    r = row(g, "A+B")
    assert not bool(r["finalist"]) and r["reason"] == "insufficient on validation"


def test_no_hypotheses_empty():
    g = validation_gate(synth(), "crypto", signal_info=INFO)
    assert g.empty and list(g.columns) == GATE_COLS
