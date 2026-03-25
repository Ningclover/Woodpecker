"""Frame masker — applies a Selection to a FrameData and writes a masked tar.bz2.

Self-registers as "mask_frames" in StepRegistry.

Reads from ctx:
    ctx.frame_data   (FrameData)
    ctx.selection    (Selection)
    ctx.config.get("out_path")   — output path; default derived from source_path

Writes to ctx.outputs:
    ctx.outputs["masked_archive"]  — path of the written file
"""

from __future__ import annotations

import io
import os
import time
import tarfile

import numpy as np

from woodpecker.core.registry import StepRegistry
from woodpecker.core.selection import Selection
from woodpecker.io.frame_data import FrameData
from woodpecker.processing.base import ProcessingStep


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _build_mask(
    frame: np.ndarray,
    channels: np.ndarray,
    start_tick: int,
    tick_range: tuple,
    ch_ranges: list,
    plane_channels_list: list,
) -> np.ndarray:
    t0, t1 = tick_range
    mask = np.zeros(frame.shape, dtype=bool)
    nticks = frame.shape[1]

    col_lo = max(0, t0 - start_tick)
    col_hi = min(nticks, t1 - start_tick + 1)

    for plane_idx, r in enumerate(ch_ranges):
        if r is None:
            pch = plane_channels_list[plane_idx]
            pch_lo, pch_hi = int(pch[0]), int(pch[-1])
        else:
            pch_lo, pch_hi = r.ch_min, r.ch_max
        row_mask = (channels >= pch_lo) & (channels <= pch_hi)
        row_indices = np.where(row_mask)[0]
        if len(row_indices) == 0:
            continue
        mask[row_indices[:, None],
             np.arange(col_lo, col_hi)[None, :]] = True

    return mask


@StepRegistry.register("mask_frames")
class FrameMasker(ProcessingStep):
    """Zero out frame data outside the user-selected tick/channel ranges."""

    def run(self, ctx) -> None:
        fd: FrameData = ctx.frame_data
        sel: Selection = ctx.selection

        out_path = ctx.config.get("out_path")
        if out_path is None:
            base = os.path.splitext(os.path.splitext(fd.source_path)[0])[0]
            out_path = base + "-selected.tar.bz2"

        tick_range = sel.tick_range if sel.tick_range else (fd.start_tick, fd.end_tick)
        plane_channels_list = [p.channels for p in fd.planes]

        anode_id = fd.anode_id
        data = fd.raw_data

        modified = {}
        for key, arr in data.items():
            if not key.startswith("frame_"):
                modified[key] = arr
                continue
            # Determine filter tag and start_tick for this frame
            if f"gauss{anode_id}" in key:
                ti_key = next(k for k in data if k.startswith(f"tickinfo_gauss{anode_id}_"))
                ch_key = next(k for k in data if k.startswith(f"channels_gauss{anode_id}_"))
            else:
                ti_key = next(k for k in data if k.startswith(f"tickinfo_wiener{anode_id}_"))
                ch_key = next(k for k in data if k.startswith(f"channels_wiener{anode_id}_"))

            start_tick = int(data[ti_key][0])
            channels = data[ch_key]

            mask = _build_mask(
                arr, channels, start_tick,
                tick_range, sel.ch_ranges, plane_channels_list,
            )
            modified[key] = np.where(mask, arr, np.float32(0))

        with tarfile.open(fd.source_path, "r:bz2") as src_tf:
            orig_members = src_tf.getmembers()

        with tarfile.open(out_path, "w:bz2") as out_tf:
            for orig_m in orig_members:
                key = orig_m.name[:-4]
                raw = _npy_bytes(modified[key])
                info = tarfile.TarInfo(name=orig_m.name)
                info.size = len(raw)
                info.mode = orig_m.mode
                info.mtime = int(time.time())
                info.type = tarfile.REGTYPE
                out_tf.addfile(info, io.BytesIO(raw))

        print(f"Saved: {out_path}")
        ctx.outputs["masked_archive"] = out_path
