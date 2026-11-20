"""Signal library. Importing this package registers every per-asset signal."""
from .registry import REGISTRY, SIGNAL_INFO, Signal, compute_signals, register
from . import trend, pullback, breakout, meanrev, momentum, divergence, volume, sr, candles  # noqa: F401
from .momentum import cross_sectional_signals

__all__ = [
    "REGISTRY",
    "SIGNAL_INFO",
    "Signal",
    "compute_signals",
    "cross_sectional_signals",
    "register",
]
