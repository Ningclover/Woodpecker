"""FrameData dataclass — holds all arrays loaded from a gauss/wiener frame archive."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PlaneData:
    """Frame and channel array for one wire plane."""
    name: str          # "U", "V", or "W"
    frame: np.ndarray  # shape (nch, ntick)
    channels: np.ndarray  # shape (nch,)

    @property
    def ch_min(self) -> int:
        return int(self.channels[0])

    @property
    def ch_max(self) -> int:
        return int(self.channels[-1])


@dataclass
class FrameData:
    """All arrays from one frame archive, ready for GUI and processing."""

    anode_id: int
    filter_tag: str          # "gauss" or "wiener"
    frame: np.ndarray        # full (nch, ntick) — all planes concatenated
    channels: np.ndarray     # full channel array
    tickinfo: np.ndarray     # [start_tick, ntick, period_us]
    planes: List[PlaneData]  # split per-plane

    # Preserved raw dict for round-trip saving (key → ndarray, no .npy suffix)
    raw_data: Dict[str, np.ndarray] = field(default_factory=dict)

    # Source file path (for default output naming)
    source_path: str = ""

    @property
    def start_tick(self) -> int:
        return int(self.tickinfo[0])

    @property
    def nticks(self) -> int:
        return self.frame.shape[1]

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.nticks - 1
