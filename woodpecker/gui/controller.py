"""Step-machine controller — pure Python, no matplotlib dependency.

Tracks the 4-step selection workflow:
  Step 0 — tick range   (drag vertical)
  Step 1 — U channels   (drag horizontal)
  Step 2 — V channels   (drag horizontal)
  Step 3 — W channels   (drag horizontal)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from woodpecker.core.selection import PlaneChannelRange, Selection

PLANE_LABELS = ["U", "V", "W"]

# (label, description, span_direction, active_plane_indices)
STEPS = [
    ("Step 1/4", "Select TICK range — drag UP/DOWN on any plot",        "vertical",   [0, 1, 2]),
    ("Step 2/4", "Select U channel range — drag LEFT/RIGHT on plane U", "horizontal", [0]),
    ("Step 3/4", "Select V channel range — drag LEFT/RIGHT on plane V", "horizontal", [1]),
    ("Step 4/4", "Select W channel range — drag LEFT/RIGHT on plane W", "horizontal", [2]),
]

STEP_COLORS = ["orange", "royalblue", "forestgreen", "crimson"]


class SelectionController:
    """Manages step state and the in-progress Selection.

    Callbacks (set by the GUI layer):
        on_step_changed(step_idx)    — called when the active step changes
        on_selection_complete(sel)   — called when all 4 steps are confirmed
        on_preview(step_idx, vmin, vmax) — called while dragging
    """

    def __init__(self) -> None:
        self._step: int = 0
        self._pending: Optional[Tuple[float, float]] = None
        self._selection = Selection()

        # GUI callbacks — set by app.py
        self.on_step_changed: Optional[Callable[[int], None]] = None
        self.on_selection_complete: Optional[Callable[[Selection], None]] = None
        self.on_preview: Optional[Callable[[int, float, float], None]] = None

    # ── public interface ───────────────────────────────────────────────────────

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def selection(self) -> Selection:
        return self._selection

    @property
    def steps(self):
        return STEPS

    @property
    def step_colors(self):
        return STEP_COLORS

    def span_selected(self, vmin: float, vmax: float) -> None:
        """Called by the GUI when the user drags a span on the active axes."""
        self._pending = (vmin, vmax)
        if self.on_preview:
            self.on_preview(self._step, vmin, vmax)

    def confirm_step(self) -> None:
        """Called when the user presses ENTER."""
        idx = self._step
        pending = self._pending

        if pending is not None:
            vlo, vhi = pending
            if idx == 0:
                self._selection.tick_range = (int(vlo), int(vhi))
            else:
                pi = STEPS[idx][3][0]
                self._selection.ch_ranges[pi] = PlaneChannelRange(
                    plane=PLANE_LABELS[pi],
                    ch_min=int(vlo),
                    ch_max=int(vhi),
                )
        else:
            print(f"  ({STEPS[idx][0]}: no drag made, step skipped)")

        self._pending = None
        next_idx = idx + 1

        if next_idx >= len(STEPS):
            if self.on_step_changed:
                self.on_step_changed(-1)  # signal "done"
            if self.on_selection_complete:
                self.on_selection_complete(self._selection)
        else:
            self._step = next_idx
            if self.on_step_changed:
                self.on_step_changed(self._step)

    def reset(self) -> None:
        """Restart from Step 0."""
        self._step = 0
        self._pending = None
        self._selection = Selection()
        if self.on_step_changed:
            self.on_step_changed(0)
        print("Selection reset — back to Step 1.")
