"""Run RPVA from frozen or locally generated held-out predictions.

This interface documents expected arguments only. It does not rerun the full
NFL audit during repository QA.
Software smoke test only; not an additional validation study.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check inputs for running NFL RPVA.")
    parser.add_argument("--predictions", type=Path, help="Held-out prediction table with A/H/D/C states.")
    parser.add_argument("--config", type=Path, default=Path("configs/nfl_frozen.yaml"), help="Frozen audit configuration.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the command-line interface only.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        print("run_rpva interface check passed; no NFL audit was run.")
        return
    if args.predictions is None:
        raise SystemExit("Provide --predictions from locally generated held-out model outputs.")
    if not args.predictions.exists():
        raise SystemExit(f"Prediction file does not exist: {args.predictions}")
    raise SystemExit("Full NFL RPVA execution is not verified in this public scaffolding package.")

if __name__ == '__main__':
    main()
