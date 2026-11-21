"""CLI subcommand: woodpecker plot-frames

Draw U, V, W wire plane views from a FrameFileSink tar.bz2 archive.
Works with any tag (gauss, wiener, raw, orig, ...).

All chanmask_<type>_* keys in the archive are overlaid with distinct colors:
  bad       → blue   (chndb hard-bad channels)
  noisy     → orange (OmnibusNoiseFilter noisy tag)
  femb_noise→ red    (PDHDFEMBNoiseSub tag)
  others    → purple

  Full-waveform entries (t0=0, t1=nticks-1) → vertical line spanning full plot.
  Time-localised entries → segment spanning [t0, t1] only (clipped to view).

Usage
-----
  woodpecker plot-frames woodpecker_data/protodune-sp-frames-sim-anode2.tar.bz2
  woodpecker plot-frames data.tar.bz2 --tag raw2
  woodpecker plot-frames data.tar.bz2 --out frames.png
  woodpecker plot-frames data.tar.bz2 --tick-range 1000 3000
  woodpecker plot-frames data.tar.bz2 --zrange -50 50
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import tarfile

import numpy as np


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "plot-frames",
        help="Draw U/V/W wire plane views from a FrameFileSink tar.bz2",
    )
    p.add_argument("frame_file", help="Path to *-anode<N>.tar.bz2")
    p.add_argument(
        "--tag", default=None,
        help="Frame tag to load (default: auto-detect first available tag)",
    )
    p.add_argument(
        "--out", default=None,
        help="Output image path (default: <frame_file>.png)",
    )
    p.add_argument(
        "--tick-range", nargs=2, type=int, default=None, metavar=("T0", "T1"),
        help="Tick range to display (default: full range)",
    )
    p.add_argument(
        "--chan-range", nargs=2, type=int, default=None, metavar=("C0", "C1"),
        help="Channel range to display, inclusive, in absolute channel numbers "
             "(default: all channels)",
    )
    p.add_argument(
        "--planes", default=None,
        help="Comma-separated planes to draw, e.g. 'V' or 'U,W' (default: U,V,W)",
    )
    p.add_argument(
        "--pedestal", action="store_true",
        help="Subtract each channel's median before plotting. Needed for the "
             "'orig' tag, whose samples are raw ADC sitting on a large "
             "per-channel pedestal; NF output ('raw') is already baseline-subtracted.",
    )
    p.add_argument(
        "--zrange", nargs=2, type=float, default=None, metavar=("ZMIN", "ZMAX"),
        help="Color scale range (default: symmetric ±3*RMS)",
    )
    p.add_argument(
        "--dpi", type=int, default=150,
        help="Output image DPI (default: 150)",
    )
    p.add_argument(
        "--detector", default="vd", choices=["vd", "hd", "sbnd"],
        help=(
            "Detector type controlling U/V/W plane splitting. "
            "'vd' (default): split on channel number gaps (ProtoDUNE-VD global numbering). "
            "'hd': split at fixed offsets 800/1600 within local 0-based channel numbering "
            "(ProtoDUNE-HD, 800 U + 800 V + 960 W per APA). "
            "'sbnd': split at fixed offsets 1984/3968 within one APA's 5638-channel block "
            "(SBND, 1984 U + 1984 V + 1670 W per APA)."
        ),
    )
    p.set_defaults(func=run)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_archive(path: str) -> dict:
    """Return dict of name-without-.npy → ndarray for every .npy in archive."""
    data = {}
    with tarfile.open(path, "r:bz2") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".npy"):
                raw = tf.extractfile(member).read()
                data[member.name[:-4]] = np.load(io.BytesIO(raw))
    return data


def _find_tag(raw_data: dict, requested_tag: str | None, anode_id: int) -> str:
    """Return the best matching frame tag."""
    frame_keys = [k for k in raw_data if k.startswith("frame_")]
    if not frame_keys:
        raise ValueError(f"No frame_* keys found. Available: {list(raw_data)[:10]}")

    # Extract all available tags
    tag_re = re.compile(r"^frame_(.+)_\d+$")
    available = []
    for k in frame_keys:
        m = tag_re.match(k)
        if m:
            available.append(m.group(1))

    if requested_tag:
        if requested_tag in available:
            return requested_tag
        raise ValueError(f"Tag '{requested_tag}' not found. Available: {available}")

    # Auto-detect: prefer raw > gauss > wiener > * > first available
    for preferred in [f"raw{anode_id}", "raw", f"gauss{anode_id}", "gauss",
                      f"wiener{anode_id}", "wiener", "*"]:
        if preferred in available:
            return preferred

    return available[0]


def _split_planes(frame: np.ndarray, channels: np.ndarray,
                  boundaries: list[int] | None = None):
    """Split (nch, ntick) into [(frame_U,ch_U), (frame_V,ch_V), (frame_W,ch_W)].

    boundaries: sorted list of channel-index offsets where new planes begin
                (e.g. [800, 1600] for HD).  None → auto-detect from gaps.
    """
    if boundaries:
        # boundaries are channel-count offsets into the sorted channel array
        starts = [0] + boundaries
        ends = boundaries + [len(channels)]
    else:
        diffs = np.diff(channels)
        gap_idx = list(np.where(diffs > 1)[0])
        starts = [0] + [i + 1 for i in gap_idx]
        ends = [i + 1 for i in gap_idx] + [len(channels)]
    return [(frame[s:e], channels[s:e]) for s, e in zip(starts, ends)]


# ── main ──────────────────────────────────────────────────────────────────────

_ANOMALY_NSEG = 50
_ANOMALY_LOW = 0.3    # railed: segment RMS below this fraction of the median
_ANOMALY_HIGH = 3.0   # violent: segment RMS above this multiple of the median


def _mask_anomaly_ranges(raw_path: str, masks_by_key: dict) -> dict | None:
    """For every masked channel found in the sibling orig archive, return
    {channel: (t0, t1, frac)} where [t0, t1] spans the anomalous segments
    of the orig waveform and frac is the anomalous-segment fraction.
    (t0, t1, frac) = (0, nticks-1, 1.0) when no localised anomaly exists.
    Returns None when there is no orig archive or nothing is masked."""
    if not masks_by_key:
        return None
    base = os.path.basename(raw_path)
    if "sp-frames-raw" not in base:
        return None
    orig_path = os.path.join(os.path.dirname(raw_path) or ".",
                             base.replace("sp-frames-raw", "orig-frames"))
    if not os.path.exists(orig_path):
        print(f"  (no orig archive {os.path.basename(orig_path)}; "
              f"skipping anomaly ranges)")
        return None

    data = _load_archive(orig_path)
    frame_key = next((k for k in data if k.startswith("frame_")), None)
    ch_key = next((k for k in data if k.startswith("channels_")), None)
    if frame_key is None or ch_key is None:
        return None
    frame = data[frame_key].astype(np.float32)
    frame -= np.median(frame, axis=1, keepdims=True)
    chan_index = {int(c): i for i, c in enumerate(data[ch_key])}

    wanted = set()
    for info in masks_by_key.values():
        wanted.update(info["full"])
        wanted.update(ch for ch, _, _ in info["ranges"])

    nticks = frame.shape[1]
    seglen = nticks // _ANOMALY_NSEG
    out = {}
    for ch in wanted:
        i = chan_index.get(ch)
        if i is None:
            continue
        seg = frame[i, :_ANOMALY_NSEG * seglen].reshape(_ANOMALY_NSEG, seglen)
        srms = seg.std(axis=1)
        med = float(np.median(srms))
        anom = (srms < _ANOMALY_LOW * med) | (srms > _ANOMALY_HIGH * med)
        if med <= 0 or not anom.any():
            out[ch] = (0, nticks - 1, 1.0)
            continue
        idxs = np.flatnonzero(anom)
        out[ch] = (int(idxs[0]) * seglen,
                   (int(idxs[-1]) + 1) * seglen - 1,
                   float(anom.sum()) / _ANOMALY_NSEG)
    return out


def run(args: argparse.Namespace) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib",
              file=sys.stderr)
        sys.exit(1)

    path = args.frame_file
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {path} ...")
    raw_data = _load_archive(path)
    print(f"  Keys found: {sorted(raw_data)}")

    m = re.search(r"anode(\d+)", os.path.basename(path))
    anode_id = int(m.group(1)) if m else 0

    tag = _find_tag(raw_data, args.tag, anode_id)
    print(f"  Using tag: {tag}")

    frame_key = next((k for k in raw_data if k.startswith(f"frame_{tag}_")), None)
    ch_key    = next((k for k in raw_data if k.startswith(f"channels_{tag}_")), None)
    ti_key    = next((k for k in raw_data if k.startswith(f"tickinfo_{tag}_")), None)

    if frame_key is None:
        print(f"ERROR: frame_{tag}_* not found. Available: {sorted(raw_data)}",
              file=sys.stderr)
        sys.exit(1)

    frame    = raw_data[frame_key].astype(np.float32)
    channels = raw_data[ch_key]
    tickinfo = raw_data[ti_key] if ti_key else np.array([0, frame.shape[1], 0.5])

    # Pedestal removal.  Uses the full waveform (before any --tick-range cut)
    # so the baseline estimate is not biased by a zoom window that happens to
    # be signal-rich.
    if args.pedestal:
        frame -= np.median(frame, axis=1, keepdims=True)
        print("  Pedestal: subtracted per-channel median over full waveform")

    # Color assigned to each mask type for overlay lines.
    # Chosen to be visible on both hot_r (W plane) and RdBu_r (U/V planes).
    _MASK_COLORS = {
        "bad":        "gray",
        "noisy":      "blue",
        "femb_noise": "magenta",
    }
    _MASK_COLOR_DEFAULT = "yellow"

    # Draw order: non-bad masks first, bad last so it is always on top.
    _MASK_ZORDER = {"bad": 10, "noisy": 8, "femb_noise": 8}
    _MASK_ZORDER_DEFAULT = 7

    # Collect all chanmask_<type>_* keys present in the archive.
    # masks_by_key: raw key → {"mtype", "full": set[int], "ranges": list[(ch,t0,t1)]}
    # masks: mtype → {"full": set[int], "ranges": list[(ch,t0,t1)]}  (merged for plot)
    mask_re = re.compile(r"^chanmask_(.+)_\d+$")
    masks_by_key: dict[str, dict] = {}
    masks: dict[str, dict] = {}
    full_nticks = frame.shape[1]
    for key in sorted(raw_data):
        m = mask_re.match(key)
        if not m:
            continue
        mtype = m.group(1)
        key_info: dict = {"mtype": mtype, "full": set(), "ranges": []}
        for r in raw_data[key]:
            ch, t0, t1 = int(r[0]), int(r[1]), int(r[2])
            if t0 <= 0 and t1 >= full_nticks - 1:
                key_info["full"].add(ch)
            else:
                key_info["ranges"].append((ch, t0, t1))
        masks_by_key[key] = key_info
        # merge into per-type dict for plot overlay
        if mtype not in masks:
            masks[mtype] = {"full": set(), "ranges": []}
        masks[mtype]["full"].update(key_info["full"])
        masks[mtype]["ranges"].extend(key_info["ranges"])

    for mtype, info in sorted(masks.items()):
        color = _MASK_COLORS.get(mtype, _MASK_COLOR_DEFAULT)
        print(f"  Mask '{mtype}' ({color}): {len(info['full'])} full-waveform, "
              f"{len(info['ranges'])} time-localised")

    start_tick = int(tickinfo[0])
    nticks = frame.shape[1]
    # Use relative ticks (0-based index) for display — absolute start_tick is
    # a simulation clock offset and not meaningful for visual inspection.
    ticks = np.arange(nticks)

    # Tick range selection (relative indices)
    if args.tick_range:
        t0, t1 = args.tick_range
        i0 = max(0, t0)
        i1 = min(nticks, t1)
        frame = frame[:, i0:i1]
        ticks = ticks[i0:i1]

    if args.detector == "hd":
        plane_boundaries = [800, 1600]
    elif args.detector == "sbnd":
        plane_boundaries = [1984, 3968]
    else:
        plane_boundaries = None
    planes = _split_planes(frame, channels, plane_boundaries)
    plane_labels = ["U", "V", "W"]
    # pad to 3 if fewer splits
    while len(planes) < 3:
        planes.append((np.zeros((1, frame.shape[1])), np.array([0])))
    planes = planes[:3]

    # Plane selection (--planes U,V,W): keep only the requested plane views.
    if args.planes:
        want = [x.strip().upper() for x in args.planes.split(",") if x.strip()]
        bad = [w for w in want if w not in plane_labels]
        if bad:
            print(f"ERROR: unknown plane(s) {bad}; choose from U,V,W", file=sys.stderr)
            sys.exit(1)
        keep = [plane_labels.index(w) for w in want]
        planes = [planes[i] for i in keep]
        plane_labels = [plane_labels[i] for i in keep]

    # Channel range selection (--chan-range C0 C1, absolute channel numbers,
    # inclusive).  Applied per plane; planes with no surviving channel are
    # dropped so a zoom into one plane's channel span does not leave empty
    # panels behind.
    if args.chan_range:
        c0, c1 = args.chan_range
        sel_planes, sel_labels = [], []
        for (pframe, pch), label in zip(planes, plane_labels):
            keep = np.where((pch >= c0) & (pch <= c1))[0]
            if keep.size == 0:
                print(f"  Plane {label}: no channels in [{c0}, {c1}]; dropped")
                continue
            sel_planes.append((pframe[keep[0]:keep[-1] + 1],
                               pch[keep[0]:keep[-1] + 1]))
            sel_labels.append(label)
        if not sel_planes:
            print(f"ERROR: no channels in range [{c0}, {c1}]", file=sys.stderr)
            sys.exit(1)
        planes, plane_labels = sel_planes, sel_labels

    # Global RMS from all non-zero samples (used when --zrange not given)
    global_rms = float(np.std(frame[frame != 0])) if np.any(frame != 0) else 1.0

    print(f"  Anode {anode_id}, tag={tag}, "
          f"relative ticks={ticks[0]}..{ticks[-1]} (abs start={start_tick}), "
          f"channels={channels[0]}..{channels[-1]}, "
          f"global RMS={global_rms:.1f}")

    nrow = len(planes)
    fig, axes = plt.subplots(nrow, 1, figsize=(10, 6 * nrow), sharex=False,
                             squeeze=False)
    axes = axes[:, 0]
    fig.suptitle(f"{os.path.basename(path)}  |  tag={tag}  |  anode{anode_id}",
                 fontsize=11)

    for ax, (pframe, pch), label in zip(axes, planes, plane_labels):
        if pframe.shape[0] == 0:
            ax.set_title(f"Plane {label} — no channels")
            continue

        # Per-plane color scale
        if "gauss" in tag or "dnnsp" in tag:
            # Deconvolved-signal tags: SP 'gauss', or the DNN-ROI output 'dnnsp'
            # when L1SP is off (HD's L1SP-off path keeps the dnnsp tag rather
            # than retagging to gauss). Both are non-negative deconvolved charge,
            # so plot identically — fixed 0..1000, white at 0 — for apples-to-
            # apples comparison across NF->SP / DNN-ROI / DNN+L1SP variants.
            vmin, vmax = 0.0, 1000.0
            norm = None
            cmap = "hot_r"  # white at low end (0), dark at high end
        elif args.zrange:
            vmin, vmax = args.zrange
            norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
            cmap = "RdBu_r"
        else:
            plane_rms = float(np.std(pframe[pframe != 0])) if np.any(pframe != 0) else global_rms
            if label == "W":
                # Collection plane: unipolar (positive signal), use 0..10*RMS
                vmin, vmax = 0.0, 10 * plane_rms
                norm = None
                cmap = "hot_r"  # white at 0
            else:
                # Induction planes: bipolar, use ±10*RMS
                vmin, vmax = -10 * plane_rms, 10 * plane_rms
                norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
                cmap = "RdBu_r"
            print(f"  Plane {label}: RMS={plane_rms:.1f}, range=[{vmin:.1f}, {vmax:.1f}]")

        # pframe shape: (nch, ntick) — transpose so x=channel, y=tick
        im = ax.imshow(
            pframe.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            norm=norm,
            vmin=vmin if norm is None else None,
            vmax=vmax if norm is None else None,
            cmap=cmap,
            extent=[pch[0], pch[-1], ticks[0], ticks[-1]],
        )
        # Overlay channel masks — one color per mask type.
        #   Full-waveform entries → axvline spanning the full plot height.
        #   Time-localised entries → line segment spanning [t0, t1] only
        #     (clipped to the visible tick window; dropped if outside).
        view_t0, view_t1 = ticks[0], ticks[-1]
        legend_handles = []
        mask_summary_parts = []
        # Sort so bad is drawn last (highest zorder = on top of other masks)
        draw_order = sorted(masks.items(),
                            key=lambda kv: _MASK_ZORDER.get(kv[0], _MASK_ZORDER_DEFAULT))
        for mtype, info in draw_order:
            color = _MASK_COLORS.get(mtype, _MASK_COLOR_DEFAULT)
            zorder = _MASK_ZORDER.get(mtype, _MASK_ZORDER_DEFAULT)
            plane_full_m = [ch for ch in info["full"] if pch[0] <= ch <= pch[-1]]
            plane_loc_m: list[tuple[int, int, int]] = []
            for ch, t0, t1 in info["ranges"]:
                if not (pch[0] <= ch <= pch[-1]):
                    continue
                seg_t0 = max(t0, view_t0)
                seg_t1 = min(t1, view_t1)
                if seg_t1 < seg_t0:
                    continue
                plane_loc_m.append((ch, seg_t0, seg_t1))

            for bch in plane_full_m:
                ax.axvline(x=bch, color=color, linewidth=1.0, alpha=0.5,
                           zorder=zorder)
            for ch, seg_t0, seg_t1 in plane_loc_m:
                ax.plot([ch, ch], [seg_t0, seg_t1],
                        color=color, linewidth=1.5, alpha=0.5, zorder=zorder)

            n_total = len(plane_full_m) + len(plane_loc_m)
            if n_total:
                from matplotlib.lines import Line2D
                legend_handles.append(
                    Line2D([0], [0], color=color, linewidth=2.0,
                           label=f"{mtype} ({n_total})")
                )
                mask_summary_parts.append(f"{mtype}:{n_total}")

        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right",
                      fontsize=7, framealpha=0.6)

        mask_label = f"  [{', '.join(mask_summary_parts)}]" if mask_summary_parts else ""
        ax.set_title(f"Plane {label}  (ch {pch[0]}–{pch[-1]},  {pframe.shape[0]} wires){mask_label}")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Tick")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    axes[-1].set_xlabel("Channel")
    plt.tight_layout()

    # Strip all extensions (e.g. .tar.bz2) then add .png
    base = os.path.basename(path)
    for ext in (".tar.bz2", ".tar.gz", ".bz2", ".gz"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    else:
        base = os.path.splitext(base)[0]
    out_path = args.out or os.path.join(os.path.dirname(path) or ".", base + ".png")
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")

    # Anomaly time range per masked channel, measured from the sibling
    # orig (pre-NF) archive.  The NF output zero-fills masked channels, so
    # the orig waveform is the only place the time structure of the
    # masking cause survives.  Per channel the waveform is split into
    # NSEG segments; a segment is anomalous when its RMS is far below
    # (railed, e.g. FEMB saturation) or far above (violent noise) the
    # channel's own median segment RMS.  frac is the anomalous-segment
    # fraction; frac == 1.0 is written when there is no localised anomaly
    # (the masking cause is persistent, e.g. a dead or uniformly noisy
    # channel).
    anomaly = _mask_anomaly_ranges(path, masks_by_key)

    # Write channel lists per mask key to a companion txt file.
    txt_path = out_path[:-4] + "_masks.txt" if out_path.endswith(".png") else out_path + "_masks.txt"
    with open(txt_path, "w") as f:
        f.write(f"Channel mask report: {os.path.basename(path)}\n")
        f.write("=" * 60 + "\n\n")
        if not masks_by_key:
            f.write("No channel masks found.\n")
        for key, info in sorted(masks_by_key.items()):
            full_sorted = sorted(info["full"])
            ranges_sorted = sorted(info["ranges"])
            f.write(f"[{key}]  {len(full_sorted)} full-waveform, "
                    f"{len(ranges_sorted)} time-localised\n")
            if full_sorted:
                f.write("  full channels:\n")
                for ch in full_sorted:
                    f.write(f"    {ch}\n")
            if ranges_sorted:
                f.write("  time-localised (channel, t0, t1):\n")
                for ch, t0, t1 in ranges_sorted:
                    f.write(f"    {ch}  {t0}  {t1}\n")
            if anomaly is not None:
                rows = [(ch, *anomaly[ch]) for ch in full_sorted if ch in anomaly]
                if rows:
                    f.write("  anomaly range from orig (channel, t0, t1, frac):\n")
                    for ch, t0, t1, frac in rows:
                        f.write(f"    {ch}  {t0}  {t1}  {frac:.3f}\n")
            f.write("\n")
    print(f"Mask report: {txt_path}")
