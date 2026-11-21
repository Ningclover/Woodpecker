"""CLI subcommand: woodpecker run-nf

Run the WireCell noise-filtering (NF only, no SP) pipeline on per-anode
orig-frame archives produced by the LArSoft stage.

The pipeline reads:
  {input_dir}/{orig_prefix}-anode{N}.tar.bz2

and writes:
  {output_dir}/{raw_prefix}-anode{N}.tar.bz2   (NF output)

Default prefixes by detector:
  vd : protodune-orig-frames      / protodune-sp-frames-raw
  hd : protodunehd-orig-frames    / protodunehd-sp-frames-raw

The output file name and location are identical to what run-nfsp writes for
the NF (raw) frames, so downstream tools see no difference.

Mask tag handling
-----------------
By default each NF mask tag is preserved under its own name in the output
frame.  Use --maskmap to override.  Examples:

  # keep tags distinct (default, HD)
  woodpecker run-nf --input data/hd/run027409/evt_1/ --detector hd

  # rename all to "bad" (old run-nfsp behaviour, HD)
  woodpecker run-nf --input ... --detector hd --maskmap '{chirp:"bad",noisy:"bad",femb_noise:"bad"}'

  # only route femb_noise to "bad", drop the others (HD)
  woodpecker run-nf --input ... --detector hd --maskmap '{femb_noise:"bad"}'

  # VD example
  woodpecker run-nf --input data/vd/run039324/evt1/ --detector vd

Usage
-----
  woodpecker run-nf --input data/vd/run039324/evt1/
  woodpecker run-nf --input data/hd/run027409/evt_1/ --detector hd
  woodpecker run-nf --input data/hd/run027409/evt_1/ --detector hd --anode-indices '[0,1,2,3]'
  woodpecker run-nf --input data/hd/run027409/evt_1/ --detector hd --output woodpecker_data/
  woodpecker run-nf --input data/hd/run027409/evt_1/ --detector hd --dry-run
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

# Per-detector defaults  (NF-only; no SP prefix needed)
_DETECTOR_DEFAULTS = {
    "vd": {
        "orig_prefix": "protodune-orig-frames",
        "raw_prefix":  "protodune-sp-frames-raw",
        "jsonnet_subdir": os.path.join("wcp-porting-img", "pdvd", "wct-nf.jsonnet"),
        "has_resampler": True,
        "elec_gain": None,   # VD does not use elecGain extVar
        "default_maskmap": '{chirp:"chirp",noisy:"noisy"}',
    },
    "hd": {
        "orig_prefix": "protodunehd-orig-frames",
        "raw_prefix":  "protodunehd-sp-frames-raw",
        "jsonnet_subdir": os.path.join("wcp-porting-img", "pdhd", "wct-nf.jsonnet"),
        "has_resampler": True,
        "elec_gain": "14",   # mV/fC; required extVar in pdhd params.jsonnet
        "default_maskmap": '{chirp:"chirp",noisy:"noisy",femb_noise:"femb_noise"}',
    },
}


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "run-nf",
        help="Run wire-cell NF only on protodunehd-orig-frames-anode*.tar.bz2",
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
        help="Detector type: 'vd' (ProtoDUNE-VD, default) or 'hd' (ProtoDUNE-HD)",
    )
    p.add_argument(
        "--raw-prefix", default=None,
        help="Override NF output filename prefix (default: protodunehd-sp-frames-raw)",
    )
    p.add_argument(
        "--anode-indices", default=None,
        help="Anode indices as JSON list e.g. '[0,1,2,3]' "
             "(default: auto-detect from files in --input)",
    )
    p.add_argument(
        "--jsonnet", default=None,
        help="Path to wct-nf.jsonnet (default: auto-search wcp-porting-img/pdhd/)",
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
        help="Disable the 512->500 ns resampler (default: enabled for data)",
    )
    p.add_argument(
        "--elec-gain", default=None,
        help="FE amplifier gain in mV/fC (default: 14). "
             "Use 7.8 for data taken after mid-July 2024.",
    )
    p.add_argument(
        "--maskmap", default=None,
        help=(
            "Jsonnet object mapping NF tag names to output tag names. "
            "Default (VD): {chirp:\"chirp\",noisy:\"noisy\"}. "
            "Default (HD): {chirp:\"chirp\",noisy:\"noisy\",femb_noise:\"femb_noise\"}. "
            "Example to merge all to 'bad' (HD): "
            "'{chirp:\"bad\",noisy:\"bad\",femb_noise:\"bad\"}'"
        ),
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


def _find_wire_cell_data() -> str | None:
    """Search upward from cwd for the wire-cell-data dir. Mirrors
    cmd_run_nfsp: detector params (e.g. PDVD's
    protodunevd-wires-larsoft-v5.json.bz2) live only there."""
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


def _build_env(wct_base: str | None) -> dict:
    env = os.environ.copy()
    parts = []
    if wct_base and os.path.isdir(wct_base):
        parts.append(os.path.join(wct_base, "toolkit", "cfg"))
    # wire-cell-data must be on WIRECELL_PATH for every chain (it is NOT
    # there by default on wcgpu1): detector params such as PDVD's wires
    # v5 json live only there.  Mirrors cmd_run_nfsp._build_env.
    wcd = (os.path.join(wct_base, "wire-cell-data")
           if wct_base and os.path.isdir(os.path.join(wct_base, "wire-cell-data"))
           else _find_wire_cell_data())
    if wcd:
        parts.append(wcd)
    if parts:
        current = env.get("WIRECELL_PATH", "")
        env["WIRECELL_PATH"] = os.pathsep.join(parts) + (os.pathsep + current if current else "")
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
            f"ERROR: could not find wct-nf.jsonnet for detector '{det}'.\n"
            "Use --jsonnet /path/to/wct-nf.jsonnet",
            file=sys.stderr,
        )
        sys.exit(1)

    env = _build_env(args.wct_base)

    anode_list = "[" + ",".join(str(i) for i in anode_ids) + "]"
    maskmap = args.maskmap if args.maskmap else defaults["default_maskmap"]

    cmd = [
        "wire-cell",
        "-l", "stdout",
        "-L", args.log_level,
        "--tla-str",  f"orig_prefix={os.path.join(input_dir, orig_prefix)}",
        "--tla-str",  f"raw_prefix={raw_prefix}",
        "--tla-code", f"anode_indices={anode_list}",
        "--tla-code", f"maskmap={maskmap}",
    ]

    # HD requires elecGain as an external variable. Pass as --ext-code so it
    # arrives as a JSON number: params.jsonnet does arithmetic (gain*wc.mV),
    # which throws "Unexpected type string, expected number" if -V (ext-str)
    # is used. Matches run-nfsp.
    if defaults["elec_gain"] is not None:
        gain = args.elec_gain if args.elec_gain else defaults["elec_gain"]
        cmd += ["--ext-code", f"elecGain={gain}"]
    else:
        gain = None

    if defaults["has_resampler"]:
        tla_val = "sim" if args.no_resampler else "data"
        cmd += ["--tla-str", f"reality={tla_val}"]

    script_dir = os.path.dirname(jsonnet)
    cmd += ["-c", os.path.basename(jsonnet)]

    print("\n" + "=" * 60)
    print(f"wire-cell NF-only command  [detector={det}]")
    print("=" * 60)
    print(f"  detector      : {det}")
    print(f"  input_dir     : {input_dir}")
    print(f"  output_dir    : {output_dir}")
    print(f"  orig_prefix   : {orig_prefix}")
    print(f"  raw_prefix    : {raw_prefix}")
    print(f"  anode_indices : {anode_list}")
    print(f"  use_resampler : {'no' if args.no_resampler else 'yes (512->500 ns)'}")
    if gain is not None:
        print(f"  elecGain      : {gain} mV/fC")
    print(f"  maskmap       : {maskmap}")
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
