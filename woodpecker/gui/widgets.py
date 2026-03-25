"""Factory helpers for matplotlib widgets (SpanSelectors, buttons, text bars)."""

from __future__ import annotations

from typing import Callable, List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.widgets import Button, SpanSelector


def make_span_selectors(
    axes: List[Axes],
    active_plane_indices: List[int],
    direction: str,
    color: str,
    on_select: Callable[[float, float], None],
) -> List[SpanSelector]:
    spans = []
    for pi in active_plane_indices:
        sp = SpanSelector(
            axes[pi], on_select, direction=direction,
            useblit=True,
            props=dict(alpha=0.3, facecolor=color),
            interactive=True,
            drag_from_anywhere=False,
        )
        spans.append(sp)
    return spans


def make_save_button(fig: Figure) -> tuple:
    """Return (btn_ax, Button). btn_ax is initially invisible."""
    btn_ax = fig.add_axes([0.82, 0.10, 0.15, 0.06])
    btn = Button(btn_ax, "Save selection", color="0.85", hovercolor="lightgreen")
    btn.label.set_fontsize(11)
    btn_ax.set_visible(False)
    return btn_ax, btn


def make_instruction_text(fig: Figure):
    ax = fig.add_axes([0.01, 0.91, 0.98, 0.05])
    ax.axis("off")
    txt = ax.text(
        0.5, 0.5, "",
        transform=ax.transAxes,
        fontsize=11, ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )
    return txt


def make_summary_text(fig: Figure):
    ax = fig.add_axes([0.01, 0.10, 0.78, 0.06])
    ax.axis("off")
    txt = ax.text(
        0.01, 0.5, "— no selection yet —",
        transform=ax.transAxes,
        fontsize=9, va="center", family="monospace",
        bbox=dict(boxstyle="round", facecolor="#e8f4e8", alpha=0.8),
    )
    return txt
