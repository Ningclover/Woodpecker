"""Top-level CLI entry point with subcommands."""

from __future__ import annotations

import argparse
import sys

from woodpecker.cli import cmd_mask, cmd_select, cmd_extract, cmd_run_img, cmd_run_clustering, cmd_run_sim_check, cmd_plot_frames


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="woodpecker",
        description="WireCell targeted region selection and debugging tool",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    cmd_select.add_parser(subparsers)
    cmd_mask.add_parser(subparsers)
    cmd_extract.add_parser(subparsers)
    cmd_run_img.add_parser(subparsers)
    cmd_run_clustering.add_parser(subparsers)
    cmd_run_sim_check.add_parser(subparsers)
    cmd_plot_frames.add_parser(subparsers)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
