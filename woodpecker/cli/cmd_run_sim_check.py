"""CLI subcommand: woodpecker run-sim-check

Run a wire-cell simulation using the longest track found in a
'woodpecker extract-tracks' JSON file, then write SP frames in the same
format as 'woodpecker run-img' output so they can be fed back into
'woodpecker run-img' for a direct data-vs-simulation comparison.

The command is equivalent to:
  wire-cell \\
    --tla-code "tracks_json=$(cat woodpecker_data/tracks-upload.json)" \\
    --tla-str  output_prefix='woodpecker_data/protodune-sp-frames-sim' \\
    --tla-code anode_indices='[N,...]' \\
    -c wcp-porting-img/pdvd/wct-sim-check-track.jsonnet

Usage
-----
  woodpecker run-sim-check
  woodpecker run-sim-check --tracks-file woodpecker_data/tracks-upload.json
  woodpecker run-sim-check --anode-indices '[2]'
  woodpecker run-sim-check --dry-run
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys


_FNAME_RE = re.compile(r"^(.+)-anode(\d+)\.tar\.bz2$")


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "run-sim-check",
        help="Simulate the longest extracted track and save SP frames for comparison",
    )
    p.add_argument(
        "--tracks-file", default=None,
        help="Path to tracks-*.json from 'woodpecker extract-tracks' "
             "(default: auto-detect woodpecker_data/tracks-*.json)",
    )
    p.add_argument(
        "--datadir", default="woodpecker_data",
        help="Directory with masked tar.bz2 files — used to auto-detect anode indices "
             "(default: ./woodpecker_data/)",
    )
    p.add_argument(
        "--anode-indices", default=None,
        help="Override anode indices as JSON list e.g. '[1,2]' "
             "(default: auto-detect from woodpecker_data/ masked frame files)",
    )
    p.add_argument(
        "--output-prefix", default=None,
        help="Prefix for output tar.bz2 files: <prefix>-anodeN.tar.bz2 "
             "(default: <datadir>/protodune-sp-frames-sim)",
    )
    p.add_argument(
        "--jsonnet", default=None,
        help="Path to wct-sim-check-track.jsonnet "
             "(default: auto-search for wcp-porting-img/pdvd relative to CWD)",
    )
    p.add_argument(
        "--script-dir", default=None,
        help="Directory containing wct-sim-check-track.jsonnet",
    )
    p.add_argument(
        "--wct-base", default=None,
        help="WCT_BASE directory (required)",
    )
    p.add_argument(
        "--log-level", default="debug", choices=["debug", "info", "warning", "error"],
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print command but do not execute",
    )
    p.set_defaults(func=run)


# ── helpers ───────────────────────────────────────────────────────────────────

def _detect_anode_ids(datadir: str):
    """Read anode IDs from woodpecker_data/ masked frame files."""
    ids = []
    for path in sorted(glob.glob(os.path.join(datadir, "*.tar.bz2"))):
        m = _FNAME_RE.match(os.path.basename(path))
        if m:
            ids.append(int(m.group(2)))
    return sorted(ids)


def _find_tracks_file(datadir: str) -> str | None:
    """Auto-detect the first tracks-*.json file in datadir."""
    candidates = sorted(glob.glob(os.path.join(datadir, "tracks-*.json")))
    return candidates[0] if candidates else None


def _resolve_jsonnet(script_dir: str | None) -> str | None:
    candidates = []
    if script_dir:
        candidates.append(os.path.join(script_dir, "wct-sim-check-track.jsonnet"))
    cwd = os.path.abspath(".")
    for _ in range(5):
        candidates.append(
            os.path.join(cwd, "wcp-porting-img", "pdvd", "wct-sim-check-track.jsonnet")
        )
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
        dunereco_base = os.path.join(wct_base, "dunereco", "dunereco", "DUNEWireCell")
        extra = os.pathsep.join([
            # DUNEWireCell must come before toolkit/cfg so that
            # 'common/tools.jsonnet' resolves to the dunereco version
            # which defines elec_resps (plural array, needed by sp.jsonnet)
            dunereco_base,
            os.path.join(wct_base, "toolkit", "cfg"),
            os.path.join(dunereco_base, "protodunevd"),
            os.path.join(wct_base, "local", "share", "wirecell"),
        ])
        current = env.get("WIRECELL_PATH", "")
        env["WIRECELL_PATH"] = extra + (os.pathsep + current if current else "")
    return env


def _run_or_print(cmd, dry_run: bool, env: dict) -> None:
    print("\n  " + " \\\n    ".join(str(c) for c in cmd))
    if not dry_run:
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"ERROR: wire-cell exited with code {result.returncode}",
                  file=sys.stderr)
            sys.exit(result.returncode)


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    datadir = args.datadir

    # Resolve tracks file
    tracks_file = args.tracks_file or _find_tracks_file(datadir)
    if tracks_file is None:
        print(f"ERROR: no tracks-*.json found in '{datadir}'.\n"
              f"Run 'woodpecker extract-tracks' first, or use --tracks-file.",
              file=sys.stderr)
        sys.exit(1)
    tracks_file = os.path.abspath(tracks_file)
    with open(tracks_file) as f:
        tracks_json = f.read()

    # Print the longest track that will be simulated
    import json as _json
    all_tracks = _json.loads(tracks_json)
    best = max(all_tracks, key=lambda t: t["length_cm"])
    print("\n--- longest track selected for simulation ---")
    print(f"  cluster_id : {best['cluster_id']}")
    print(f"  source     : {best['source_file']}")
    print(f"  length_cm  : {best['length_cm']}")
    print(f"  linearity  : {best['linearity']}")
    print(f"  theta_deg  : {best['theta_deg']}")
    print(f"  phi_deg    : {best['phi_deg']}")
    print(f"  start (cm) : {best['start']}")
    print(f"  end   (cm) : {best['end']}")
    print("--------------------------------------------")

    # Resolve anode indices
    if args.anode_indices:
        anode_list = args.anode_indices
    else:
        ids = _detect_anode_ids(datadir)
        if not ids:
            print(f"ERROR: no masked frame files found in '{datadir}'.\n"
                  f"Use --anode-indices.",
                  file=sys.stderr)
            sys.exit(1)
        anode_list = "[" + ",".join(str(i) for i in ids) + "]"

    # Resolve output prefix
    output_prefix = args.output_prefix or os.path.join(datadir, "protodune-sp-frames-sim")

    # Resolve jsonnet
    jsonnet = args.jsonnet or _resolve_jsonnet(args.script_dir)
    if jsonnet is None:
        print("ERROR: could not find wct-sim-check-track.jsonnet.\n"
              "Use --jsonnet /path/to/wct-sim-check-track.jsonnet", file=sys.stderr)
        sys.exit(1)

    env = _build_env(args.wct_base)

    print("\n" + "=" * 60)
    print("wire-cell sim-check")
    print("=" * 60)
    print(f"  tracks_file    : {tracks_file} ({len(tracks_json)} bytes)")
    print(f"  anode_indices  : {anode_list}")
    print(f"  output_prefix  : {output_prefix}")
    print(f"  jsonnet        : {jsonnet}")
    print(f"  WIRECELL_PATH  : {env.get('WIRECELL_PATH', '(not set)')}")
    print("=" * 60)

    cmd = [
        "wire-cell",
        "-l", "stdout",
        "-L", args.log_level,
        "--tla-code", f"tracks_json={tracks_json}",
        "--tla-str",  f"output_prefix={output_prefix}",
        "--tla-code", f"anode_indices={anode_list}",
        "-c", jsonnet,
    ]

    _run_or_print(cmd, args.dry_run, env)

    if args.dry_run:
        print("\n(dry-run: not executing)")
