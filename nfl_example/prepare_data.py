"""Prepare local NFL competition files for RPVA.

This is implementation scaffolding for authors or users who have locally
obtained the official NFL Big Data Bowl files. Raw data are not redistributed.
Software smoke test only; not an additional validation study.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check inputs for local NFL data preparation.")
    parser.add_argument("--raw-data-root", type=Path, help="Local directory containing official competition files.")
    parser.add_argument("--config", type=Path, default=Path("configs/nfl_frozen.yaml"), help="Frozen audit configuration.")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments without reading restricted data.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        print("prepare_data interface check passed; no restricted data were read.")
        return
    if args.raw_data_root is None:
        raise SystemExit("Provide --raw-data-root pointing to locally obtained official competition files.")
    if not args.raw_data_root.exists():
        raise SystemExit(f"Raw data root does not exist: {args.raw_data_root}")
    raise SystemExit("Full NFL data preparation is not verified in this public scaffolding package.")

if __name__ == '__main__':
    main()
