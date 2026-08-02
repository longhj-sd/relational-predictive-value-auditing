"""Fit the primary Histogram Gradient Boosting model from local NFL files.

This interface documents expected arguments only. It does not retrain the
full NFL analysis during repository QA.
Software smoke test only; not an additional validation study.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check inputs for primary-model fitting.")
    parser.add_argument("--prepared-data", type=Path, help="Prepared local player-play modeling table.")
    parser.add_argument("--config", type=Path, default=Path("configs/nfl_frozen.yaml"), help="Frozen audit configuration.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the command-line interface only.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        print("fit_primary_model interface check passed; no model was trained.")
        return
    if args.prepared_data is None:
        raise SystemExit("Provide --prepared-data generated from locally obtained official competition files.")
    if not args.prepared_data.exists():
        raise SystemExit(f"Prepared data file does not exist: {args.prepared_data}")
    raise SystemExit("Full NFL primary-model fitting is not verified in this public scaffolding package.")

if __name__ == '__main__':
    main()
