"""PipelineContext — carries all data between pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PipelineContext:
    """Shared carrier object passed through every processing step.

    Steps read from and write to this object. The ``outputs`` dict is an
    open bag for inter-step data so steps don't need to know each other's
    internal types directly.
    """

    frame_data: Optional[Any] = None      # FrameData
    cluster_data: Optional[Any] = None    # ClusterData (future)
    selection: Optional[Any] = None       # Selection
    outputs: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
