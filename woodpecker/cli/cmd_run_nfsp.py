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


def _resolve_jsonnet(hint: str | None, detector: str) -> str | None:
    candidates = []
    if hint:
        candidates.append(hint)
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


def _build_env(wct_base: str | None) -> dict:
    env = os.environ.copy()
    if wct_base and os.path.isdir(wct_base):
        extra = os.path.join(wct_base, "toolkit", "cfg")
        current = env.get("WIRECELL_PATH", "")
        env["WIRECELL_PATH"] = extra + (os.pathsep + current if current else "")
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

    jsonnet = _resolve_jsonnet(args.jsonnet, det)
    if jsonnet is None:
        print(
            f"ERROR: could not find wct-nf-sp.jsonnet for detector '{det}'.\n"
            "Use --jsonnet /path/to/wct-nf-sp.jsonnet",
            file=sys.stderr,
        )
        sys.exit(1)

    env = _build_env(args.wct_base)

    anode_list = "[" + ",".join(str(i) for i in anode_ids) + "]"

    cmd = [
        "wire-cell",
        "-l", "stdout",
        "-L", args.log_level,
        "--tla-str",  f"orig_prefix={os.path.join(input_dir, orig_prefix)}",
        "--tla-str",  f"raw_prefix={raw_prefix}",
        "--tla-str",  f"sp_prefix={sp_prefix}",
        "--tla-code", f"anode_indices={anode_list}",
    ]

    if defaults["has_sigoutform"]:
        cmd += ["--tla-str", f"sigoutform={args.sigoutform}"]

    # HD requires elecGain as an external variable (-V)
    if defaults["elec_gain"] is not None:
        gain = args.elec_gain if args.elec_gain else defaults["elec_gain"]
        cmd += ["-V", f"elecGain={gain}"]

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

    script_dir = os.path.dirname(jsonnet)
    cmd += ["-c", os.path.basename(jsonnet)]

    print("\n" + "=" * 60)
    print(f"wire-cell NF+SP command  [detector={det}]")
    print("=" * 60)
    print(f"  detector      : {det}")
    print(f"  input_dir     : {input_dir}")
    print(f"  output_dir    : {output_dir}")
    print(f"  orig_prefix   : {orig_prefix}")
    print(f"  raw_prefix    : {raw_prefix}")
    print(f"  sp_prefix     : {sp_prefix}")
    print(f"  anode_indices : {anode_list}")
    print(f"  use_resampler : {use_resampler}")
    print(f"  sigoutform    : {args.sigoutform}")
    if defaults["elec_gain"] is not None:
        print(f"  elecGain      : {args.elec_gain or defaults['elec_gain']} mV/fC")
    print(f"  jsonnet       : {jsonnet}")
    print(f"  script_dir    : {script_dir}")
    print(f"  wct_base      : {args.wct_base or '(not set, using current WIRECELL_PATH)'}")
    print(f"  WIRECELL_PATH : {env.get('WIRECELL_PATH', '(not set)')}")
    print(f"  input files   :")
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
