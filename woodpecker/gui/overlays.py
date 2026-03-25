"""Matplotlib overlay helpers for selection rectangles."""

from __future__ import annotations

import matplotlib.patches as mpatches
from matplotlib.axes import Axes


def clear_overlays(ax: Axes, tag: str) -> None:
    for patch in list(ax.patches):
        if getattr(patch, "_overlay_tag", None) == tag:
            patch.remove()


def draw_hband(ax: Axes, ymin: float, ymax: float, color: str, tag: str, alpha: float = 0.25) -> None:
    clear_overlays(ax, tag)
    xlim = ax.get_xlim()
    p = mpatches.Rectangle(
        (xlim[0], ymin), xlim[1] - xlim[0], ymax - ymin,
        color=color, alpha=alpha, zorder=3,
    )
    p._overlay_tag = tag
    ax.add_patch(p)


def draw_vband(ax: Axes, xmin: float, xmax: float, color: str, tag: str, alpha: float = 0.25) -> None:
    clear_overlays(ax, tag)
    ylim = ax.get_ylim()
    p = mpatches.Rectangle(
        (xmin, ylim[0]), xmax - xmin, ylim[1] - ylim[0],
        color=color, alpha=alpha, zorder=3,
    )
    p._overlay_tag = tag
    ax.add_patch(p)
