"""CLI subcommand: woodpecker mask <archive> --selection <json>

Non-interactive masking: load a previously saved Selection JSON and apply it.
"""

from __future__ import annotations

import os
import sys

import woodpecker.io.frame_source      # noqa: F401
import woodpecker.processing.masker    # noqa: F401

from woodpecker.core.registry import SourceRegistry
from woodpecker.core.selection import Selection
from woodpecker.pipeline.context import PipelineContext
from woodpecker.pipeline.runner import PipelineRunner


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "mask",
        help="Non-interactive: apply a saved Selection JSON to an archive",
    )
    p.add_argument("archive", help="protodune-sp-frames-anodeN.tar.bz2")
    p.add_argument("--selection", required=True, metavar="JSON",
                   help="Selection JSON file (from 'select --save-selection')")
    p.add_argument("--out", default=None,
                   help="Output tar.bz2 path (default: ./woodpecker_data/<prefix>-anodeN.tar.bz2)")
    p.add_argument("--prefix", default="protodune-sp-frames-part",
                   help="Output filename prefix (default: protodune-sp-frames-part)")
    p.add_argument("--outdir", default="woodpecker_data",
                   help="Output directory (default: ./woodpecker_data/)")
    p.set_defaults(func=run)


def run(args) -> None:
    with open(args.selection) as f:
        selection = Selection.from_json(f.read())

    source_cls = SourceRegistry.get("frames")
    frame_data = source_cls().load(args.archive)

    out_path = args.out
    if out_path is None:
        os.makedirs(args.outdir, exist_ok=True)
        out_path = os.path.join(args.outdir,
                                f"{args.prefix}-anode{frame_data.anode_id}.tar.bz2")

    ctx = PipelineContext(
        frame_data=frame_data,
        selection=selection,
        config={"out_path": out_path},
    )
    PipelineRunner(["mask_frames"]).run(ctx)
