"""Reproduce manuscript tables from verified derived outputs.

This interface documents expected arguments only. It does not regenerate the
full NFL analysis during repository QA.
Software smoke test only; not an additional validation study.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check inputs for table reproduction.")
    parser.add_argument("--derived-output-root", type=Path, default=Path("derived_outputs"), help="Directory with permitted derived summary outputs.")
    parser.add_argument("--config", type=Path, default=Path("configs/nfl_frozen.yaml"), help="Frozen audit configuration.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the command-line interface only.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        print("reproduce_tables interface check passed; no tables were regenerated.")
        return
    if not args.derived_output_root.exists():
        raise SystemExit(f"Derived output root does not exist: {args.derived_output_root}")
    raise SystemExit("Full NFL table reproduction is not verified in this public scaffolding package.")

if __name__ == '__main__':
    main()
