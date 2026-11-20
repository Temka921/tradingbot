"""Signal registry.

A signal is a pure function ``bars -> bool Series`` aligned to the bars index.
The value at day t uses only data up to the close of t; entry happens at the
open of t+1 (handled by the outcomes layer, never here). Registry names are
stable column IDs used by every later phase, so never rename them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd

Direction = Literal["long", "short"]
FAMILIES = (
    "trend", "pullback", "breakout", "meanrev", "momentum",
    "divergence", "volume", "sr", "candles",
)


@dataclass(frozen=True)
class Signal:
    name: str
    family: str
    direction: Direction
    fn: Callable[[pd.DataFrame], pd.Series]


REGISTRY: dict[str, Signal] = {}

# name -> (family, direction) for every signal, including cross-sectional ones
# that are not per-asset functions. Single source of truth for later analysis.
SIGNAL_INFO: dict[str, tuple[str, str]] = {
    "mom_12_1_top_decile": ("momentum", "long"),
}


def register(name: str, family: str, direction: Direction = "long"):
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if direction not in ("long", "short"):
        raise ValueError(f"bad direction {direction!r}")

    def deco(fn: Callable[[pd.DataFrame], pd.Series]):
        if name in SIGNAL_INFO:
            raise ValueError(f"duplicate signal {name!r}")
        REGISTRY[name] = Signal(name, family, direction, fn)
        SIGNAL_INFO[name] = (family, direction)
        return fn

    return deco


def _as_bool(s: pd.Series, index: pd.Index) -> pd.Series:
    s = pd.Series(s, index=index) if not isinstance(s, pd.Series) else s.reindex(index)
    return s.where(s.notna(), False).astype(bool)


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """One bool column per registered signal (registry order); NaN -> False."""
    cols = {name: _as_bool(sig.fn(df), df.index) for name, sig in REGISTRY.items()}
    return pd.DataFrame(cols, index=df.index)
