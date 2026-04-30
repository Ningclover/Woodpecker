"""CLI subcommand: woodpecker extract-track-waveform

Extract a single 1D peak-aligned mean waveform from one wire plane of a
simulation frame archive produced by ``run-sim-check``.

Intended for sim archives whose track was generated to be parallel to a wire
plane and perpendicular to one specific plane's wires (the "target plane").
On the target plane the track illuminates many adjacent channels with a
clean, identical signal; on the other two planes it illuminates fewer.  This
tool computes the peak-aligned mean across the channels that actually carry
signal on the chosen plane.

Algorithm
---------
1. Load the frame archive and split into U/V/W planes.
2. Restrict to the target plane (parsed from filename or via ``--plane``).
3. Identify "signal channels" on that plane: those whose peak |ADC| exceeds
   ``--threshold`` × plane RMS.
4. For each signal channel, find the abs-peak in the full row; extract a
   ``2 * --half-window`` tick window centred on that peak; shift to align the
   peak at ``half_window``; accumulate.
5. Divide by the number of channels that contributed → 1D mean waveform.

Outputs
-------
- A ``.png`` plot of the 1D waveform.
- An ``.npy`` of the float64 array (length = ``2 * half_window``).

Usage
-----
  woodpecker extract-track-waveform sim.tar.bz2
  woodpecker extract-track-waveform sim.tar.bz2 --plane V
  woodpecker extract-track-waveform sim.tar.bz2 --threshold 7 --half-window 300
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

from woodpecker.cli.cmd_compare_waveforms import (
    _aligned_mean_waveform_full,
    _load_frames,
)


PLANE_LABELS = ["U", "V", "W"]
HD_PLANE_BOUNDARIES = [800, 1600]   # channel-count offsets within a single anode


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "extract-track-waveform",
        help="Extract a 1D peak-aligned mean waveform from a sim track on its target plane",
    )
    p.add_argument("frame_file", help="Sim frame archive (e.g. *-anode<N>.tar.bz2)")
    p.add_argument(
        "--plane", default=None, choices=PLANE_LABELS,
        help="Target plane (default: parse from filename, e.g. '...-anode0-U-anode0...')",
    )
    p.add_argument(
        "--detector", default=None, choices=["vd", "hd"],
        help="Detector flavour (default: 'hd' if filename starts with 'protodunehd', else 'vd')",
    )
    p.add_argument(
        "--tag", default=None,
        help="Frame tag (default: auto-detect raw<N>/raw)",
    )
    p.add_argument(
        "--threshold", type=float, default=5.0,
        help="Signal-channel cut: |peak ADC| > threshold * plane_RMS (default: 5)",
    )
    p.add_argument(
        "--half-window", type=int, default=200,
        help="Half-width of output waveform array in ticks (default: 200)",
    )
    p.add_argument(
        "--out", default=None,
        help="Output PNG path (default: <frame_file>.<plane>-waveform.png). "
             "The .npy is written next to the PNG with the same stem.",
    )
    p.add_argument(
        "--dpi", type=int, default=150,
        help="PNG DPI (default: 150)",
    )
    p.set_defaults(func=run)


# ── helpers ───────────────────────────────────────────────────────────────────

def _detect_plane_from_name(name: str) -> str | None:
    """Return 'U'/'V'/'W' if filename has '-anode<N>-<P>-' segment."""
    m = re.search(r"-anode\d+-([UVW])(?:-|\.)", name)
    return m.group(1) if m else None


def _detect_detector_from_name(name: str) -> str:
    return "hd" if "protodunehd" in name else "vd"


def _split_planes(frame: np.ndarray, channels: np.ndarray, detector: str):
    """Split (nch, ntick) into [(frame_U,ch_U), (frame_V,ch_V), (frame_W,ch_W)]."""
    if detector == "hd":
        starts = [0] + HD_PLANE_BOUNDARIES
        ends = HD_PLANE_BOUNDARIES + [len(channels)]
    else:
        diffs = np.diff(channels)
        gap_idx = list(np.where(diffs > 1)[0])
        starts = [0] + [i + 1 for i in gap_idx]
        ends = [i + 1 for i in gap_idx] + [len(channels)]
    return [(frame[s:e], channels[s:e]) for s, e in zip(starts, ends)]


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    path = args.frame_file
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    base = os.path.basename(path)
    detector = args.detector or _detect_detector_from_name(base)
    plane = args.plane or _detect_plane_from_name(base)
    if plane is None:
        print(f"ERROR: could not parse plane from filename '{base}'. Use --plane.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {path}")
    print(f"  detector={detector}  target plane={plane}")
    frame, channels, tickinfo, used_tag = _load_frames(path, args.tag)
    print(f"  tag={used_tag}  frame shape={frame.shape}  "
          f"channels {channels.min()}..{channels.max()}")

    plane_data = _split_planes(frame, channels, detector)
    if len(plane_data) != 3:
        print(f"ERROR: expected 3 planes after split, got {len(plane_data)}.",
              file=sys.stderr)
        sys.exit(1)

    plane_idx = PLANE_LABELS.index(plane)
    pframe, pchannels = plane_data[plane_idx]
    rms = float(pframe.std())
    peak_per_ch = np.abs(pframe).max(axis=1)
    sig_mask = peak_per_ch > args.threshold * rms
    n_sig = int(sig_mask.sum())
    if n_sig == 0:
        print(f"ERROR: no channels above {args.threshold}*RMS on plane {plane} "
              f"(RMS={rms:.2f}). Lower --threshold or check the input.",
              file=sys.stderr)
        sys.exit(1)

    sig_channels = pchannels[sig_mask]
    print(f"  Plane {plane}: {len(pchannels)} channels, RMS={rms:.2f}, "
          f"signal channels (peak > {args.threshold}*RMS): {n_sig} "
          f"({sig_channels.min()}..{sig_channels.max()})")

    nticks_window = 2 * args.half_window
    waveform = _aligned_mean_waveform_full(
        frame=pframe,
        channels=pchannels,
        ch_sel=sig_channels,
        nticks=nticks_window,
        half_window=args.half_window,
    )

    print(f"  Mean waveform: length={len(waveform)}, "
          f"peak={waveform[args.half_window]:.2f}, "
          f"abs-max={np.max(np.abs(waveform)):.2f}")

    # Resolve output paths
    if args.out:
        png_path = args.out
    else:
        png_path = f"{path}.{plane}-waveform.png"
    npy_path = os.path.splitext(png_path)[0] + ".npy"

    # Save .npy
    np.save(npy_path, waveform)
    print(f"  Saved {npy_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available — skipping PNG.", file=sys.stderr)
        return

    ticks_axis = np.arange(-args.half_window, args.half_window)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ticks_axis, waveform, lw=1.0)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.axvline(0, color="0.5", lw=0.5, ls="--")
    ax.set_xlabel("Tick (peak-aligned)")
    ax.set_ylabel("Mean ADC")
    ax.set_title(
        f"{base}\nplane {plane}, {n_sig} signal channels "
        f"(threshold = {args.threshold}*RMS = {args.threshold * rms:.1f})"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=args.dpi)
    plt.close(fig)
    print(f"  Saved {png_path}")
