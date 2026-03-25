"""Selection dataclass — the shared currency between GUI, processing, and pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PlaneChannelRange:
    plane: str          # "U", "V", or "W"
    ch_min: int
    ch_max: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlaneChannelRange":
        return cls(**d)


@dataclass
class Selection:
    """Holds a confirmed tick range and per-plane channel ranges."""

    tick_range: Optional[Tuple[int, int]] = None
    ch_ranges: List[Optional[PlaneChannelRange]] = field(
        default_factory=lambda: [None, None, None]
    )

    def is_complete(self) -> bool:
        return self.tick_range is not None

    def to_dict(self) -> dict:
        return {
            "tick_range": list(self.tick_range) if self.tick_range else None,
            "ch_ranges": [r.to_dict() if r else None for r in self.ch_ranges],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Selection":
        tick = tuple(d["tick_range"]) if d.get("tick_range") else None
        ch = [PlaneChannelRange.from_dict(r) if r else None for r in d.get("ch_ranges", [])]
        return cls(tick_range=tick, ch_ranges=ch)

    @classmethod
    def from_json(cls, s: str) -> "Selection":
        return cls.from_dict(json.loads(s))
