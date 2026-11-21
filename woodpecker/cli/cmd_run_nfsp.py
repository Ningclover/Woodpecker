"""CLI subcommand: woodpecker run-nfsp

Run the WireCell noise-filtering + signal-processing (NF+SP) pipeline on
per-anode orig-frame archives produced by the LArSoft stage.

The pipeline reads:
  {input_dir}/{orig_prefix}-anode{N}.tar.bz2

and writes:
  {output_dir}/{raw_prefix}-anode{N}.tar.bz2   (NF output)
  {output_dir}/{sp_prefix}-anode{N}.tar.bz2    (SP output)

Default prefixes depend on --detector:
  vd : protodune-orig-frames / protodune-sp-frames-raw / protodune-sp-frames
  hd : protodunehd-orig-frames / protodunehd-sp-frames-raw / protodunehd-sp-frames

Usage
-----
  woodpecker run-nfsp --input data/vd/run039324/evt1/
  woodpecker run-nfsp --input data/hd/run027409/evt_1/ --detector hd
  woodpecker run-nfsp --input data/vd/run039324/evt1/ --anode-indices '[0,1,2,3]'
  woodpecker run-nfsp --input data/vd/run039324/evt1/ --output woodpecker_data/
  woodpecker run-nfsp --input data/vd/run039324/evt1/ --dry-run
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

# Per-detector defaults
_DETECTOR_DEFAULTS = {
    "vd": {
        "orig_prefix": "protodune-orig-frames",
        "raw_prefix":  "protodune-sp-frames-raw",
        "sp_prefix":   "protodune-sp-frames",
        "jsonnet_subdir": os.path.join("wcp-porting-img", "pdvd", "wct-nf-sp.jsonnet"),
        # DNN-ROI+L1SP variant (the default chain; see --no-dnnroi to opt out).
        "dnnroi_subdir": os.path.join("wcp-porting-img", "pdvd", "wct-nf-sp-dnnroi.jsonnet"),
        "dnnroi_model": "dnnroi/pdvd/pipe_distill_nestedunet_6ch.ts",
        "dnnroi_nchan": "6",
        "has_resampler": True,
        "resampler_tla": "reality",         # VD: --tla-str reality=data/sim
        "has_sigoutform": True,
        "elec_gain": None,
    },
    "hd": {
        "orig_prefix": "protodunehd-orig-frames",
        "raw_prefix":  "protodunehd-sp-frames-raw",
        "sp_prefix":   "protodunehd-sp-frames",
        "jsonnet_subdir": os.path.join("wcp-porting-img", "pdhd", "wct-nf-sp.jsonnet"),
        # DNN-ROI+L1SP variant (the default chain; see --no-dnnroi to opt out).
        "dnnroi_subdir": os.path.join("wcp-porting-img", "pdhd", "wct-nf-sp-dnnroi.jsonnet"),
        "dnnroi_model": "dnnroi/pdhd/pipe_distill_transformer_6ch.ts",
        "dnnroi_nchan": "6",
        "has_resampler": True,
        "resampler_tla": "reality",         # HD: --tla-str reality=data/sim
        "has_sigoutform": False,
        "elec_gain": "14",   # mV/fC; required extVar in pdhd params.jsonnet
    },
}


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "run-nfsp",
        help="Run wire-cell NF+SP on protodune(hd)-orig-frames-anode*.tar.bz2",
    )
    p.add_argument(
        "--input", default=None, required=True,
        help="Directory containing orig-frames-anode*.tar.bz2 input files",
    )
    p.add_argument(
        "--output", default=None,
        help="Directory to write output frames (default: same as --input)",
    )
    p.add_argument(
        "--detector", default="vd", choices=["vd", "hd"],
        help="Detector type: 'vd' (ProtoDUNE-VD, default) or 'hd' (ProtoDUNE-HD). "
             "Controls default file prefixes and jsonnet.",
    )
    p.add_argument(
        "--raw-prefix", default=None,
        help="Override NF output filename prefix (default: detector-specific)",
    )
    p.add_argument(
        "--anode-indices", default=None,
        help="Anode indices as JSON list e.g. '[0,1,2,3]' "
             "(default: auto-detect from files in --input)",
    )
    p.add_argument(
        "--jsonnet", default=None,
        help="Path to wct-nf-sp.jsonnet (default: auto-search wcp-porting-img/<detector>/)",
    )
    p.add_argument(
        "--wct-base", default=None,
        help="WCT_BASE directory. Sets WIRECELL_PATH to include toolkit/cfg.",
    )
    p.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"],
        help="wire-cell -L log level (default: info)",
    )
    p.add_argument(
        "--no-resampler", action="store_true",
        help="Disable the resampler (default: enabled for VD and HD)",
    )
    p.add_argument(
        "--sigoutform", default="dense", choices=["dense", "sparse"],
        help="SP output format (default: dense)",
    )
    p.add_argument(
        "--elec-gain", default=None,
        help="FE amplifier gain in mV/fC for HD detector (default: 14). "
             "Use 7.8 for data taken after mid-July 2024.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the wire-cell command but do not execute it",
    )
    # ── DNN-ROI + L1SP (the default chain) ──────────────────────────────────
    p.add_argument(
        "--no-dnnroi", action="store_true",
        help="Disable DNN-ROI + L1SP and run the plain NF->SP chain "
             "(wct-nf-sp.jsonnet). By default run-nfsp runs the DNN-ROI + "
             "L1SP-DNN chain (wct-nf-sp-dnnroi.jsonnet).",
    )
    p.add_argument(
        "--dnnroi-device", default="gpu", choices=["gpu", "cpu"],
        help="TorchService device for DNN-ROI and L1SP-DNN (default: gpu).",
    )
    p.add_argument(
        "--dnnroi-model", default=None,
        help="Override the DNN-ROI TorchScript model (default: detector-specific, "
             "resolved via WIRECELL_PATH under wire-cell-data/dnnroi/<det>/).",
    )
    p.add_argument(
        "--l1sp-mode", default="hybrid", choices=["dnn", "hybrid", "heur"],
        help="L1SP tagger flavour after DNN-ROI (default: hybrid = 5-arm "
             "heuristic gates the TorchScript tagger; the DNN only confirms "
             "candidates the heuristic already flagged). 'dnn' = TorchScript "
             "tagger decides every candidate alone; 'heur' = legacy 5-arm "
             "heuristic (no L1SP Torch model). Ignored with --no-dnnroi or "
             "--no-l1sp.",
    )
    p.add_argument(
        "--no-l1sp", action="store_true",
        help="Run DNN-ROI but omit the L1SPFilterPD stage after it "
             "(use_l1sp_dnn=false). The output frame carries the bare DNN-ROI "
             "result, not L1SP-corrected. Only meaningful with DNN-ROI on; "
             "ignored with --no-dnnroi.",
    )
    p.set_defaults(func=run)


# ── helpers ───────────────────────────────────────────────────────────────────

def _detect_anode_ids(input_dir: str, orig_prefix: str) -> list[int]:
    pattern = os.path.join(input_dir, f"{orig_prefix}-anode*.tar.bz2")
    fname_re = re.compile(rf"^{re.escape(orig_prefix)}-anode(\d+)\.tar\.bz2$")
    ids = []
    for path in sorted(glob.glob(pattern)):
        m = fname_re.match(os.path.basename(path))
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def _resolve_jsonnet(hint: str | None, detector: str, subdir: str | None = None) -> str | None:
    candidates = []
    if hint:
        candidates.append(hint)
    if subdir is None:
        subdir = _DETECTOR_DEFAULTS[detector]["jsonnet_subdir"]
    cwd = os.path.abspath(".")
    for _ in range(5):
        candidates.append(os.path.join(cwd, subdir))
        parent = os.path.dirname(cwd)
        if parent == cwd:
            break
        cwd = parent
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def _find_wire_cell_data() -> str | None:
    """Search upward from cwd for the wire-cell-data dir (holds DNN-ROI/L1SP
    .ts models). Mirrors _resolve_jsonnet's upward walk."""
    cwd = os.path.abspath(".")
    for _ in range(6):
        cand = os.path.join(cwd, "wire-cell-data")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(cwd)
        if parent == cwd:
            break
        cwd = parent
    return None


def _find_torch_lib() -> str | None:
    """Locate the pip torch lib dir (holds libtorch_cpu.so) relative to the
    python prefix. Used to LD_PRELOAD libtorch for the DNN chain."""
    # sys.prefix is the active venv/prefix (e.g. .../local); torch installs
    # under lib/pythonX.Y/site-packages/torch/lib.
    import glob as _glob
    for base in (sys.prefix, os.environ.get("PREFIX", "")):
        if not base:
            continue
        hits = _glob.glob(os.path.join(base, "lib", "python*", "site-packages", "torch", "lib"))
        for h in hits:
            if os.path.isfile(os.path.join(h, "libtorch_cpu.so")):
                return h
    return None


def _build_env(wct_base: str | None, need_models: bool = False) -> dict:
    env = os.environ.copy()
    parts = []
    if wct_base and os.path.isdir(wct_base):
        parts.append(os.path.join(wct_base, "toolkit", "cfg"))
    # wire-cell-data must be on WIRECELL_PATH for every chain (it is NOT
    # there by default on wcgpu1): DNN-ROI/L1SP-DNN resolve their .ts models
    # through it, and detector params (e.g. PDVD's
    # protodunevd-wires-larsoft-v5.json.bz2) live only there.
    wcd = (os.path.join(wct_base, "wire-cell-data")
           if wct_base and os.path.isdir(os.path.join(wct_base, "wire-cell-data"))
           else _find_wire_cell_data())
    if wcd:
        parts.append(wcd)
    if parts:
        current = env.get("WIRECELL_PATH", "")
        env["WIRECELL_PATH"] = os.pathsep.join(parts) + (os.pathsep + current if current else "")
    # libtorch needs LD_PRELOAD inside the wire-cell process for two reasons:
    #  1) libtorch_cpu.so first, so torch's bundled zlib-ng symbols win over
    #     WCT's own libz — otherwise PyTorchStreamReader::init() (the .ts ZIP
    #     reader) segfaults.  Mirrors what `import torch` does in Python.
    #  2) libtorch_cuda.so (+ libc10_cuda.so) so the CUDA dispatch kernels are
    #     registered — without them a device="gpu" load throws NotImplementedError
    #     ('aten::empty_strided' missing for the CUDA backend).
    # Harmless on CPU runs (the cuda libs just go unused).
    if need_models:
        tlib = _find_torch_lib()
        if tlib:
            preload = [os.path.join(tlib, lib) for lib in
                       ("libtorch_cpu.so", "libc10_cuda.so", "libtorch_cuda.so")]
            preload = [p for p in preload if os.path.isfile(p)]
            cur = env.get("LD_PRELOAD", "")
            existing = cur.split(os.pathsep) if cur else []
            new = [p for p in preload if p not in existing]
            if new:
                env["LD_PRELOAD"] = os.pathsep.join(new + existing) if existing \
                                    else os.pathsep.join(new)
    return env


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    det      = args.detector
    defaults = _DETECTOR_DEFAULTS[det]

    input_dir  = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output) if args.output else input_dir

    if not os.path.isdir(input_dir):
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    orig_prefix = defaults["orig_prefix"]
    raw_prefix  = args.raw_prefix if args.raw_prefix else os.path.join(output_dir, defaults["raw_prefix"])
    sp_prefix   = os.path.join(output_dir, defaults["sp_prefix"])

    if args.anode_indices:
        anode_ids = [int(x) for x in re.findall(r"\d+", args.anode_indices)]
    else:
        anode_ids = _detect_anode_ids(input_dir, orig_prefix)
        if not anode_ids:
            print(
                f"ERROR: no {orig_prefix}-anode*.tar.bz2 files found in {input_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

    use_dnnroi = not args.no_dnnroi
    subdir = defaults["dnnroi_subdir"] if use_dnnroi else defaults["jsonnet_subdir"]
    jsonnet = _resolve_jsonnet(args.jsonnet, det, subdir)
    if jsonnet is None:
        print(
            f"ERROR: could not find {os.path.basename(subdir)} for detector '{det}'.\n"
            "Use --jsonnet /path/to/jsonnet",
            file=sys.stderr,
        )
        sys.exit(1)

    env = _build_env(args.wct_base, need_models=use_dnnroi)

    anode_list = "[" + ",".join(str(i) for i in anode_ids) + "]"

    cmd = [
        "wire-cell",
        "-l", "stdout",
        "-L", args.log_level,
        "--tla-str",  f"orig_prefix={os.path.join(input_dir, orig_prefix)}",
        "--tla-str",  f"sp_prefix={sp_prefix}",
        "--tla-code", f"anode_indices={anode_list}",
    ]

    # raw_prefix / sigoutform are parameters of the plain wct-nf-sp.jsonnet
    # only; the DNN-ROI variant has neither (it writes one SP archive carrying
    # gauss/wiener/raw tags, no separate -raw file).
    if not use_dnnroi:
        cmd += ["--tla-str", f"raw_prefix={raw_prefix}"]
        if defaults["has_sigoutform"]:
            cmd += ["--tla-str", f"sigoutform={args.sigoutform}"]
        # The plain chain's L1SP defaults to 'process' (HD) / 'dump' (VD).
        # --no-l1sp turns it fully off (l1sp_pd_mode='') so "DNN + L1SP both
        # disabled" is a true bare NF->SP.
        if args.no_l1sp:
            cmd += ["--tla-str", "l1sp_pd_mode="]

    # DNN-ROI + L1SP TLAs (mirrors run_nf_sp_dnnroi_evt.sh). l1sp_pd_mode:
    # 'dnn'|'hybrid' use the L1SP Torch tagger; 'heur' -> '' keeps the legacy
    # 5-arm heuristic. use_l1sp_dnn=true wires L1SPFilterPD after DNN-ROI.
    if use_dnnroi:
        use_l1sp = not args.no_l1sp
        # heur -> '' (legacy 5-arm); when L1SP is omitted entirely l1sp_pd_mode
        # is moot, pass '' too.
        l1sp_pd_mode = "" if (args.l1sp_mode == "heur" or not use_l1sp) else args.l1sp_mode
        dnn_model = args.dnnroi_model or defaults["dnnroi_model"]
        cmd += [
            "--tla-str",  f"dnnroi_model={dnn_model}",
            "--tla-str",  f"dnnroi_device={args.dnnroi_device}",
            "--tla-code", f"dnnroi_nchan={defaults['dnnroi_nchan']}",
            "--tla-code", "use_dnnroi=true",
            "--tla-code", f"use_l1sp_dnn={'true' if use_l1sp else 'false'}",
            "--tla-str",  f"l1sp_pd_mode={l1sp_pd_mode}",
            "--tla-str",  f"l1sp_pd_dnn_device={args.dnnroi_device}",
        ]

    # HD requires elecGain as an external variable. Pass as --ext-code so the
    # value is a JSON number: params.jsonnet does arithmetic on it (gain*wc.mV),
    # which fails if it arrives as an ext-str string.
    if defaults["elec_gain"] is not None:
        gain = args.elec_gain if args.elec_gain else defaults["elec_gain"]
        cmd += ["--ext-code", f"elecGain={gain}"]

    if defaults["has_resampler"]:
        tla = defaults["resampler_tla"]
        if tla == "reality":
            # HD jsonnet: reality='data' enables resampler, 'sim' disables it
            tla_val = "sim" if args.no_resampler else "data"
        else:
            # VD jsonnet: use_resampler=true/false
            tla_val = "false" if args.no_resampler else "true"
        use_resampler = tla_val
        cmd += ["--tla-str", f"{tla}={tla_val}"]
    else:
        use_resampler = "n/a"

    # Optional extra TLAs injected via env, e.g. to enable the L1SP per-ROI
    # waveform dump without a dedicated flag. Semicolon-separated items:
    #   WP_EXTRA_TLA="str:l1sp_pd_wf_dump_path=/path;code:l1sp_pd_dump_all_rois=false"
    _extra = os.environ.get("WP_EXTRA_TLA", "").strip()
    if _extra:
        for item in _extra.split(";"):
            item = item.strip()
            if not item:
                continue
            kind, _, kv = item.partition(":")
            flag = "--tla-code" if kind == "code" else "--tla-str"
            cmd += [flag, kv]

    script_dir = os.path.dirname(jsonnet)
    cmd += ["-c", os.path.basename(jsonnet)]

    print("\n" + "=" * 60)
    print(f"wire-cell NF+SP command  [detector={det}]")
    print("=" * 60)
    print(f"  detector      : {det}")
    if use_dnnroi:
        _chain = ("DNN-ROI (no L1SP, " + args.dnnroi_device + ")" if args.no_l1sp
                  else "DNN-ROI + L1SP (" + args.l1sp_mode + ", " + args.dnnroi_device + ")")
    else:
        _chain = "plain NF->SP"
    print(f"  chain         : {_chain}")
    print(f"  input_dir     : {input_dir}")
    print(f"  output_dir    : {output_dir}")
    print(f"  orig_prefix   : {orig_prefix}")
    if not use_dnnroi:
        print(f"  raw_prefix    : {raw_prefix}")
    print(f"  sp_prefix     : {sp_prefix}")
    print(f"  anode_indices : {anode_list}")
    print(f"  use_resampler : {use_resampler}")
    if use_dnnroi:
        print(f"  dnnroi_model  : {args.dnnroi_model or defaults['dnnroi_model']}")
    else:
        print(f"  sigoutform    : {args.sigoutform}")
    if defaults["elec_gain"] is not None:
        print(f"  elecGain      : {args.elec_gain or defaults['elec_gain']} mV/fC")
    print(f"  jsonnet       : {jsonnet}")
    print(f"  script_dir    : {script_dir}")
    print(f"  wct_base      : {args.wct_base or '(not set, using current WIRECELL_PATH)'}")
    print(f"  WIRECELL_PATH : {env.get('WIRECELL_PATH', '(not set)')}")
    print("  input files   :")
    for aid in anode_ids:
        fname = f"{orig_prefix}-anode{aid}.tar.bz2"
        fpath = os.path.join(input_dir, fname)
        exists = "✓" if os.path.exists(fpath) else "✗ MISSING"
        print(f"    anode {aid}  →  {fpath}  [{exists}]")
    print()
    print(f"  (cwd: {script_dir})")
    print("Command:")
    print("  " + " \\\n    ".join(cmd))
    print("=" * 60 + "\n")

    if args.dry_run:
        print("(dry-run: not executing)")
        return

    result = subprocess.run(cmd, env=env, cwd=script_dir)
    if result.returncode != 0:
        sys.exit(result.returncode)
