from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "nfl_external_validation" / "2018"
FORBIDDEN_PATH_PATTERNS = [
    re.compile(r"D:\\work\\NFL", re.IGNORECASE),
    re.compile(r"C:\\Users\\Administrator", re.IGNORECASE),
]
RAW_FILE_NAMES = {"games.csv", "plays.csv", "players.csv", "targetedReceiver.csv"}
RAW_FILE_PATTERNS = [re.compile(r"week\d+\.csv$", re.IGNORECASE), re.compile(r"tracking.*\.csv$", re.IGNORECASE)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def check_required_files(errors: list[str]) -> None:
    required = [
        PKG / "README.md",
        PKG / "DATA_ACCESS.md",
        PKG / "protocol" / "FROZEN_2018_REPLICATION_PROTOCOL.md",
        PKG / "protocol" / "protocol_config.yaml",
        PKG / "protocol" / "protocol_sha256.txt",
        PKG / "protocol" / "DEVIATIONS.md",
        PKG / "scripts" / "run_2018_external_replication.py",
        PKG / "environment" / "requirements-2018.txt",
        PKG / "environment" / "SOFTWARE_ENVIRONMENT.md",
        PKG / "expected_outputs" / "primary_rpva_estimates.csv",
        PKG / "expected_outputs" / "sensitivity_summary.csv",
        PKG / "expected_outputs" / "pseudo_nearest_summary.csv",
        PKG / "expected_outputs" / "bootstrap_summary.csv",
        PKG / "reports" / "2018_EXTERNAL_REPLICATION_RESULTS.md",
        PKG / "reports" / "2018_2023_HARMONIZATION_REPORT.md",
        PKG / "reports" / "ANALYSIS_QC_REPORT.md",
    ]
    for path in required:
        if not path.exists():
            fail(errors, f"Missing required file: {path.relative_to(ROOT)}")


def check_config(errors: list[str]) -> None:
    try:
        with (PKG / "protocol" / "protocol_config.yaml").open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        fail(errors, f"Config does not parse: {exc}")
        return
    if cfg.get("temporal_split", {}).get("heldout") != "weeks 14-17":
        fail(errors, "Config held-out split is not weeks 14-17.")
    if cfg.get("target_receiver_source") != "public targeted-receiver bonus labels from tombliss/nfl-big-data-bowl-2021-bonus":
        fail(errors, "Config target-receiver source is not the documented tombliss dataset.")


def check_no_forbidden_paths(errors: list[str]) -> None:
    for path in PKG.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".png", ".pdf"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_PATH_PATTERNS:
            if pattern.search(text):
                fail(errors, f"Forbidden author-local path found in {path.relative_to(ROOT)}")


def check_no_raw_data(errors: list[str]) -> None:
    for path in PKG.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in RAW_FILE_NAMES or any(p.search(name) for p in RAW_FILE_PATTERNS):
            fail(errors, f"Raw or restricted data file appears in release package: {path.relative_to(ROOT)}")
        if path.stat().st_size > 10 * 1024 * 1024:
            fail(errors, f"Unexpected large file in release package: {path.relative_to(ROOT)}")


def check_protocol_hash(errors: list[str]) -> None:
    hash_file = PKG / "protocol" / "protocol_sha256.txt"
    if not hash_file.exists():
        return
    expected: dict[str, str] = {}
    for line in hash_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            expected[parts[-1]] = parts[0]
    for rel, digest in expected.items():
        path = PKG / "protocol" / rel
        if not path.exists():
            fail(errors, f"Protocol hash target missing: {rel}")
        elif sha256(path) != digest:
            fail(errors, f"Protocol hash mismatch: {rel}")


def close(actual: float, expected: float, tol: float = 0.001) -> bool:
    return abs(actual - expected) <= tol


def check_expected_outputs(errors: list[str]) -> None:
    primary = {row["estimand"]: row for row in read_csv(PKG / "expected_outputs" / "primary_rpva_estimates.csv")}
    expected = {
        "G_R": 0.9177859354384885,
        "G_N": 0.7812103861312145,
        "G_O": 0.3350894437250878,
        "lambda_D": 0.4461209424061267,
    }
    for key, value in expected.items():
        if key not in primary or not close(float(primary[key]["estimate"]), value):
            fail(errors, f"Primary expected output mismatch for {key}")
    lam = primary.get("lambda_D", {})
    if not (close(float(lam.get("ci_low", "nan")), 0.38439750752999496) and close(float(lam.get("ci_high", "nan")), 0.5098169945386133)):
        fail(errors, "lambda_D confidence interval mismatch.")

    pseudo = read_csv(PKG / "expected_outputs" / "pseudo_nearest_summary.csv")[0]
    for key, value in {"null_low": -0.057224847505317374, "null_high": 0.05886041436992461, "p_value": 0.0001999600079984003}.items():
        if not close(float(pseudo[key]), value):
            fail(errors, f"Pseudo-nearest expected output mismatch for {key}")

    sens = {row["analysis"]: row for row in read_csv(PKG / "expected_outputs" / "sensitivity_summary.csv")}
    expanded = sens.get("S2_expanded_defender_set")
    if not expanded or not close(float(expanded["lambda_D"]), 0.5352871944276119):
        fail(errors, "Expanded defender sensitivity summary mismatch.")

    for path in (PKG / "expected_outputs").glob("*"):
        if path.is_file() and path.stat().st_size == 0:
            fail(errors, f"Empty expected output: {path.relative_to(ROOT)}")


def check_deviations(errors: list[str]) -> None:
    text = (PKG / "protocol" / "DEVIATIONS.md").read_text(encoding="utf-8", errors="ignore").lower()
    if "post-freeze" in text and "no" not in text:
        fail(errors, "Protocol deviation file appears to contain a post-freeze deviation.")


def run_pipeline(args: argparse.Namespace, errors: list[str]) -> None:
    cmd = [
        sys.executable,
        str(PKG / "scripts" / "run_2018_external_replication.py"),
        "--config",
        str(PKG / "protocol" / "protocol_config.yaml"),
        "--raw-data-root",
        str(args.raw_data_root),
        "--target-label-file",
        str(args.target_label_file),
        "--output-dir",
        str(args.output_dir),
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        fail(errors, f"Full pipeline rerun failed with exit code {result.returncode}.")
        return
    regenerated = Path(args.output_dir) / "outputs" / "tables" / "Table2_2018_primary_RPVA_estimates.csv"
    if not regenerated.exists():
        fail(errors, "Full pipeline rerun did not regenerate primary estimates.")


def write_hash_manifests() -> None:
    rows = []
    sums = []
    for path in sorted(PKG.rglob("*")):
        if not path.is_file() or path.relative_to(PKG).as_posix() in {"manifests/FILE_MANIFEST.csv", "manifests/SHA256SUMS.txt"}:
            continue
        rel = path.relative_to(PKG).as_posix()
        digest = sha256(path)
        rows.append({"path": rel, "size_bytes": path.stat().st_size, "sha256": digest})
        sums.append(f"{digest}  {rel}")
    manifest = PKG / "manifests" / "FILE_MANIFEST.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    (PKG / "manifests" / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the RPVA 2018 public release package.")
    parser.add_argument("--run-pipeline", action="store_true")
    parser.add_argument("--raw-data-root", type=Path)
    parser.add_argument("--target-label-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs_2018_verify")
    args = parser.parse_args()
    errors: list[str] = []
    check_required_files(errors)
    check_config(errors)
    check_no_forbidden_paths(errors)
    check_no_raw_data(errors)
    check_protocol_hash(errors)
    check_expected_outputs(errors)
    check_deviations(errors)
    write_hash_manifests()
    if args.run_pipeline:
        if not args.raw_data_root or not args.target_label_file:
            fail(errors, "--run-pipeline requires --raw-data-root and --target-label-file.")
        else:
            run_pipeline(args, errors)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("RPVA 2018 release verification PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
