"""Standalone entry point for the EgoHand3D processing workflow."""
from __future__ import annotations

import argparse

from .workflow_cli import add_workflow_commands


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m egohand3d.workflow",
        description="Egocentric hand processing, parameter export and evaluation",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    add_workflow_commands(subcommands)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
