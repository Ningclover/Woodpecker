"""CLI subcommand: woodpecker frames-to-root

Convert one or more WireCell FrameFileSink tar.bz2 archives to a single
Magnify-compatible ROOT file.

One output ROOT file is written per anode, named:
  magnify-run<RUN_PADDED>-evt<EVT_IDX>-anode<N>.root

<EVT_IDX> is the local event index parsed from the source dir name
(e.g. "evt_12" -> 12), matching xqian's pdvd/work convention. Falls back
to the art event number from the archive if the dir is not "evt_<N>".

Each file contains:
  TH2F  hu_gauss<N>, hv_gauss<N>, hw_gauss<N>    — deconvolved gauss waveforms
  TH2F  hu_wiener<N>, hv_wiener<N>, hw_wiener<N>  — deconvolved wiener waveforms
  TH2F  hu_raw<N>, hv_raw<N>, hw_raw<N>           — raw waveforms (if --raw given)
  TH2F  hu_orig<N>, hv_orig<N>, hw_orig<N>        — pre-NF waveforms (if --orig given)
  TH1F  hu_threshold<N>, hv_threshold<N>, hw_threshold<N> — per-channel SP thresholds
  TTree T_bad<N>                                   — bad channel mask
  TTree Trun                                       — run/subrun/event metadata

Usage
-----
  woodpecker frames-to-root protodunehd-sp-frames-anode0.tar.bz2 --detector hd --run 27409
  woodpecker frames-to-root protodunehd-sp-frames-anode*.tar.bz2 --detector hd --run 27409
  woodpecker frames-to-root protodunehd-sp-frames-anode0.tar.bz2 \\
      --raw  protodunehd-sp-frames-raw-anode0.tar.bz2 \\
      --orig protodunehd-orig-frames-anode0.tar.bz2 \\
      --detector hd --run 27409 --subrun 0
  # Override output directory:
  woodpecker frames-to-root ... --outdir /some/work/dir/
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
        "frames-to-root",
        help="Convert FrameFileSink tar.bz2 archive(s) to a Magnify ROOT file",
    )
    p.add_argument(
        "frame_files", nargs="+", metavar="SP_ARCHIVE",
        help="SP frame archive(s): protodune(hd)-sp-frames-anode<N>.tar.bz2. "
             "Multiple files produce one output ROOT file per anode.",
    )
    p.add_argument(
        "--raw", nargs="*", metavar="RAW_ARCHIVE", default=None,
        help="Raw frame archive(s) to include as hu/hv/hw_raw<N> histograms.",
    )
    p.add_argument(
        "--orig", nargs="*", metavar="ORIG_ARCHIVE", default=None,
        help="Orig (pre-NF) frame archive(s) to include as hu/hv/hw_orig<N> histograms.",
    )
    p.add_argument(
        "--outdir", default=None,
        help="Directory to write output ROOT files. Overrides the default "
             "<root-base>/<run_padded>_<evt_idx>/ layout.",
    )
    p.add_argument(
        "--root-base", default=None,
        help="Base dir under which '<run_padded>_<evt_idx>/' subdirs are created "
             "(default: <data>/ROOT/<detector>, where <data> is the parent of the "
             "SP archive's run directory). Ignored if --outdir is given.",
    )
    p.add_argument(
        "--detector", default="vd", choices=["vd", "hd"],
        help=(
            "Detector type controlling U/V/W plane splitting. "
            "'vd' (default): auto-detect from channel gaps (ProtoDUNE-VD). "
            "'hd': fixed boundaries at channels 800/1600 (ProtoDUNE-HD)."
        ),
    )
    p.add_argument("--run",    type=int, default=0,    help="Run number for Trun tree (default: 0).")
    p.add_argument("--subrun", type=int, default=0,    help="Sub-run number for Trun tree (default: 0).")
    p.add_argument("--event",  type=int, default=None,
                   help="Event number for Trun tree (default: auto-read from archive filename).")
    p.set_defaults(func=run)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_archive(path: str) -> dict:
    data = {}
    with tarfile.open(path, "r:bz2") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".npy"):
                raw = tf.extractfile(member).read()
                data[member.name[:-4]] = np.load(io.BytesIO(raw))
    return data


def _event_no_from_archive(data: dict) -> int | None:
    for k in data:
        m = re.match(r"^frame_.+_(\d+)$", k)
        if m:
            return int(m.group(1))
    return None


def _anode_id_from_filename(path: str) -> int | None:
    m = re.search(r"-anode(\d+)\.tar\.bz2$", os.path.basename(path))
    return int(m.group(1)) if m else None


def _evt_idx_from_dir(path: str) -> int | None:
    """Parse local event index from a parent dir named 'evt_<N>' or 'evt<N>'."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    m = re.match(r"^evt_?(\d+)$", parent)
    return int(m.group(1)) if m else None


def _split_planes(arr2d: np.ndarray, channels: np.ndarray,
                  boundaries: list | None = None):
    """Return [(arr2d_slice, ch_slice), ...] for each U/V/W plane.

    arr2d shape: (nch, ncols).  boundaries: channel-index offsets for HD
    (e.g. [800, 1600]); None → auto-detect from gaps in channel list (VD).
    """
    if boundaries:
        starts = [0] + boundaries
        ends   = boundaries + [len(channels)]
    else:
        diffs   = np.diff(channels)
        gap_idx = list(np.where(diffs > 1)[0])
        starts  = [0] + [i + 1 for i in gap_idx]
        ends    = [i + 1 for i in gap_idx] + [len(channels)]
    return [(arr2d[s:e], channels[s:e]) for s, e in zip(starts, ends)]


def _write_th2f(tfile, name: str, frame: np.ndarray, channels: np.ndarray,
                start_tick: int) -> None:
    """Write (nch, ntick) array as TH2F(name) into tfile."""
    import ROOT
    nch    = len(channels)
    nticks = frame.shape[1]
    ch_min = int(channels[0])
    ch_max = int(channels[-1])
    h = ROOT.TH2F(name, name,
                  nch,    ch_min - 0.5, ch_max + 0.5,
                  nticks, start_tick,   start_tick + nticks)
    h.SetDirectory(tfile)
    for col, ch in enumerate(channels):
        xbin = h.GetXaxis().FindBin(int(ch))
        col_data = frame[col]
        for tick_i in range(nticks):
            v = float(col_data[tick_i])
            if v != 0.0:
                h.SetBinContent(xbin, tick_i + 1, v)
    h.Write()
    print(f"      → TH2F '{name}' ({nch} ch × {nticks} ticks)")


def _write_th1f(tfile, name: str, channels: np.ndarray, values: np.ndarray) -> None:
    """Write per-channel 1-D threshold histogram TH1F(name) into tfile."""
    import ROOT
    ch_min = int(channels[0])
    ch_max = int(channels[-1])
    h = ROOT.TH1F(name, name, ch_max - ch_min + 1, ch_min, ch_max + 1)
    h.SetDirectory(tfile)
    for i, ch in enumerate(channels):
        if i < len(values):
            h.SetBinContent(h.FindBin(int(ch) + 0.5), float(values[i]))
    h.Write()
    print(f"      → TH1F '{name}' ({len(channels)} ch)")


def _plane_of_hd(chid: int) -> int:
    if chid < 800:
        return 0
    if chid < 1600:
        return 1
    return 2


def _write_tbad(tfile, name: str, chanmask: np.ndarray,
                boundaries: list | None, all_channels: np.ndarray) -> None:
    """Write T_bad<N> TTree from chanmask array (N×3: chid, start_time, end_time)."""
    import ROOT
    tree = ROOT.TTree(name, name)
    tree.SetDirectory(tfile)

    chid_buf  = np.zeros(1, dtype=np.int32)
    plane_buf = np.zeros(1, dtype=np.int32)
    t0_buf    = np.zeros(1, dtype=np.int32)
    t1_buf    = np.zeros(1, dtype=np.int32)

    tree.Branch("chid",       chid_buf,  "chid/I")
    tree.Branch("plane",      plane_buf, "plane/I")
    tree.Branch("start_time", t0_buf,    "start_time/I")
    tree.Branch("end_time",   t1_buf,    "end_time/I")

    # Build a fast plane-lookup from the channel list
    if boundaries:
        def _plane(ch):
            for pi, b in enumerate(boundaries):
                if ch < b:
                    return pi
            return len(boundaries)
    else:
        # VD: use gap-detected plane boundaries
        if len(all_channels) > 1:
            diffs   = np.diff(all_channels)
            gap_idx = list(np.where(diffs > 1)[0])
            starts  = [0] + [i + 1 for i in gap_idx]
            plane_start_ch = [int(all_channels[s]) for s in starts]
        else:
            plane_start_ch = [0]

        def _plane(ch):
            for pi in reversed(range(len(plane_start_ch))):
                if ch >= plane_start_ch[pi]:
                    return pi
            return 0

    nentries = 0
    if chanmask.ndim == 2 and chanmask.shape[1] >= 3 and len(chanmask) > 0:
        for row in chanmask:
            chid_buf[0]  = int(row[0])
            t0_buf[0]    = int(row[1])
            t1_buf[0]    = int(row[2])
            plane_buf[0] = _plane(int(row[0]))
            tree.Fill()
            nentries += 1

    tree.Write()
    print(f"      → TTree '{name}' ({nentries} entries)")


def _write_trun(tfile, run: int, subrun: int, event: int, anode: int) -> None:
    import ROOT
    tree = ROOT.TTree("Trun", "Trun")
    tree.SetDirectory(tfile)

    run_buf    = np.array([run],    dtype=np.int32)
    subrun_buf = np.array([subrun], dtype=np.int32)
    event_buf  = np.array([event],  dtype=np.int32)
    anode_buf  = np.array([anode],  dtype=np.int32)
    nticks_buf = np.array([0],      dtype=np.int32)

    tree.Branch("runNo",          run_buf,    "runNo/I")
    tree.Branch("subRunNo",       subrun_buf, "subRunNo/I")
    tree.Branch("eventNo",        event_buf,  "eventNo/I")
    tree.Branch("anodeNo",        anode_buf,  "anodeNo/I")
    tree.Branch("total_time_bin", nticks_buf, "total_time_bin/I")

    tree.Fill()
    tree.Write()
    print(f"      → TTree 'Trun' (run={run} subrun={subrun} event={event} anode={anode})")


def _process_sp(tfile, data: dict, anode_id: int, boundaries: list | None,
                plane_labels: list) -> None:
    """Write gauss/wiener TH2F, threshold TH1F, and T_bad TTree for one anode."""
    for tag_base in [f"gauss{anode_id}", f"wiener{anode_id}"]:
        frame_key = next((k for k in data if re.match(rf"^frame_{re.escape(tag_base)}_\d+$", k)), None)
        ch_key    = next((k for k in data if re.match(rf"^channels_{re.escape(tag_base)}_\d+$", k)), None)
        ti_key    = next((k for k in data if re.match(rf"^tickinfo_{re.escape(tag_base)}_\d+$", k)), None)
        if frame_key is None:
            print(f"  WARNING: tag '{tag_base}' not found — skipping")
            continue

        frame      = data[frame_key].astype(np.float32)
        channels   = data[ch_key]
        tickinfo   = data[ti_key] if ti_key is not None else np.array([0.0, 0.5, 0.0])
        start_tick = int(tickinfo[2]) if len(tickinfo) >= 3 else 0

        print(f"  SP '{tag_base}': {len(channels)} ch, start_tick={start_tick}")
        planes = _split_planes(frame, channels, boundaries)
        while len(planes) < 3:
            planes.append((np.zeros((1, frame.shape[1]), dtype=np.float32), np.array([0])))
        for pi, (pf, pc) in enumerate(planes[:3]):
            _write_th2f(tfile, f"h{plane_labels[pi]}_{tag_base}", pf, pc, start_tick)

    # Threshold TH1F from wiener summary
    wiener_tag = f"wiener{anode_id}"
    sum_key = next((k for k in data if re.match(rf"^summary_{re.escape(wiener_tag)}_\d+$", k)), None)
    ch_key  = next((k for k in data if re.match(rf"^channels_{re.escape(wiener_tag)}_\d+$", k)), None)
    if sum_key and ch_key:
        summary  = data[sum_key]
        ch_all   = data[ch_key]
        nch      = len(ch_all)
        thresh   = summary[:nch]   # first nch entries are per-channel thresholds
        thresh_tag = f"threshold{anode_id}"
        print(f"  Threshold '{thresh_tag}': {nch} ch")
        tplanes = _split_planes(thresh[:, np.newaxis], ch_all, boundaries)
        while len(tplanes) < 3:
            tplanes.append((np.zeros((1, 1)), np.array([0])))
        for pi, (tv, tc) in enumerate(tplanes[:3]):
            _write_th1f(tfile, f"h{plane_labels[pi]}_{thresh_tag}", tc, tv[:, 0])

    # T_bad TTree
    bad_key = next((k for k in data if re.match(r"^chanmask_bad_\d+$", k)), None)
    ch_key  = next((k for k in data if re.match(rf"^channels_{re.escape(wiener_tag)}_\d+$", k)), None)
    all_channels = data[ch_key] if ch_key else np.array([], dtype=np.int32)
    chanmask     = data[bad_key] if bad_key else np.zeros((0, 3), dtype=np.int32)
    _write_tbad(tfile, f"T_bad{anode_id}", chanmask, boundaries, all_channels)


def _process_extra(tfile, path: str, out_tag_prefix: str, boundaries: list | None,
                   plane_labels: list) -> None:
    """Write TH2F histograms for raw or orig archives (all tags found, renamed to out_tag_prefix)."""
    data = _load_archive(path)
    tags_found = []
    for k in data:
        m = re.match(r"^frame_(.+)_\d+$", k)
        if m and m.group(1) not in tags_found:
            tags_found.append(m.group(1))

    for src_tag in tags_found:
        frame_key = next((k for k in data if re.match(rf"^frame_{re.escape(src_tag)}_\d+$", k)), None)
        ch_key    = next((k for k in data if re.match(rf"^channels_{re.escape(src_tag)}_\d+$", k)), None)
        ti_key    = next((k for k in data if re.match(rf"^tickinfo_{re.escape(src_tag)}_\d+$", k)), None)
        if frame_key is None:
            continue

        frame      = data[frame_key].astype(np.float32)
        channels   = data[ch_key]
        tickinfo   = data[ti_key] if ti_key is not None else np.array([0.0, 0.5, 0.0])
        start_tick = int(tickinfo[2]) if len(tickinfo) >= 3 else 0

        print(f"  Extra '{src_tag}' → '{out_tag_prefix}': {len(channels)} ch, start_tick={start_tick}")
        planes = _split_planes(frame, channels, boundaries)
        while len(planes) < 3:
            planes.append((np.zeros((1, frame.shape[1]), dtype=np.float32), np.array([0])))
        for pi, (pf, pc) in enumerate(planes[:3]):
            _write_th2f(tfile, f"h{plane_labels[pi]}_{out_tag_prefix}", pf, pc, start_tick)


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    try:
        import ROOT
        ROOT.gROOT.SetBatch(True)
    except ImportError:
        print("ERROR: PyROOT is required.", file=sys.stderr)
        sys.exit(1)

    sp_files   = args.frame_files
    raw_files  = args.raw  or []
    orig_files = args.orig or []

    for f in sp_files + raw_files + orig_files:
        if not os.path.isfile(f):
            print(f"ERROR: file not found: {f}", file=sys.stderr)
            sys.exit(1)

    boundaries   = [800, 1600] if args.detector == "hd" else None
    plane_labels = ["u", "v", "w"]

    for sp_path in sorted(sp_files):
        anode_id = _anode_id_from_filename(sp_path)
        if anode_id is None:
            print(f"WARNING: cannot parse anode id from {sp_path} — skipping.", file=sys.stderr)
            continue

        print(f"\n=== Anode {anode_id}: {sp_path}")
        sp_data  = _load_archive(sp_path)
        art_event_no = _event_no_from_archive(sp_data)
        evt_idx_dir  = _evt_idx_from_dir(sp_path)

        # filename evt token: explicit --event > parent dir index > art event > 0
        if args.event is not None:
            evt_token = args.event
        elif evt_idx_dir is not None:
            evt_token = evt_idx_dir
        elif art_event_no is not None:
            evt_token = art_event_no
        else:
            evt_token = 0

        # Trun.eventNo always uses the art event number when known
        event_no = art_event_no if art_event_no is not None else evt_token

        run_padded = f"{args.run:06d}"

        if args.outdir:
            out_dir = args.outdir
        else:
            if args.root_base:
                base = args.root_base
            else:
                # Default: '<data>/ROOT/<detector>' where <data> is two levels above
                # the run dir (i.e. parent of the detector dir).
                # e.g. .../data/hd/run027425/evt_12/foo.tar.bz2
                #      sp_path → evt_12/foo.tar.bz2
                #      dirname×1 = evt_dir, ×2 = run_dir, ×3 = det_dir, ×4 = data_dir
                #      → .../data/ROOT/hd
                evt_dir = os.path.dirname(os.path.abspath(sp_path))
                run_dir = os.path.dirname(evt_dir)
                det_dir = os.path.dirname(run_dir)
                data_dir = os.path.dirname(det_dir)
                base = os.path.join(data_dir, "ROOT", args.detector)
            out_dir = os.path.join(base, f"{run_padded}_{evt_token}")
        os.makedirs(out_dir, exist_ok=True)
        out_name = f"magnify-run{run_padded}-evt{evt_token}-anode{anode_id}.root"
        out_path = os.path.join(out_dir, out_name)

        tfile = ROOT.TFile(out_path, "RECREATE")
        if tfile.IsZombie():
            print(f"ERROR: could not open {out_path}", file=sys.stderr)
            sys.exit(1)

        _write_trun(tfile, args.run, args.subrun, event_no, anode_id)
        _process_sp(tfile, sp_data, anode_id, boundaries, plane_labels)

        raw_path = next((f for f in raw_files if _anode_id_from_filename(f) == anode_id), None)
        if raw_path:
            print(f"  Raw: {raw_path}")
            _process_extra(tfile, raw_path, f"raw{anode_id}", boundaries, plane_labels)

        orig_path = next((f for f in orig_files if _anode_id_from_filename(f) == anode_id), None)
        if orig_path:
            print(f"  Orig: {orig_path}")
            _process_extra(tfile, orig_path, f"orig{anode_id}", boundaries, plane_labels)

        tfile.Close()
        print(f"  Saved → {out_path}")
