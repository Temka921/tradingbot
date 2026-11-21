import os
from pathlib import Path

COSTS: dict[str, float] = {"stock": 0.0010, "crypto": 0.0030}
HORIZONS: tuple[int, ...] = (5, 10, 20)

SPLITS: dict[str, dict[str, tuple[str, str]]] = {
    "stock": {
        "discovery": ("2000-01-01", "2017-12-31"),
        "validation": ("2018-01-01", "2023-12-31"),
    },
    "crypto": {
        "discovery": ("2018-01-01", "2021-12-31"),
        "validation": ("2022-01-01", "2023-12-31"),
    },
}
HOLDOUT_START = "2024-01-01"
MIN_N = 100
MIN_N_SLICE = 30
DELIST_GAP_DAYS = 7          # last date this many days before the market's last date -> delisted
RELIST_GAP_DAYS = 5          # crypto: a gap longer than this splits a symbol into separate assets
MAX_SKIP_FRAC = 0.02      # build_panel raises if more assets than this fail with an error
ELIGIBLE_TOP_N: dict[str, int] = {"crypto": 100}  # market -> top-N by trailing dollar volume
LIQUIDITY_WINDOW = 30        # trailing days for the median dollar volume
BENCHMARK: dict[str, str] = {"stock": "SPY", "crypto": "BTCUSDT"}

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR: Path = Path(os.environ.get("SWINGBOT_DATA", _REPO_ROOT / "data"))
