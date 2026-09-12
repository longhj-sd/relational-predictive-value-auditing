from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
EXEC = ROOT / "execution"
OUT_MD = AUDIT / "C_SHUFFLED_DONOR_UNIVERSE_FORENSIC_RESOLUTION_v4.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def guard_not_val(path: Path) -> None:
    resolved = path.resolve()
    parts = [p.lower() for p in resolved.parts]
    if "official_s3" in parts and any(p in {"val", "validation"} for p in parts):
        raise RuntimeError(f"official AV2 VAL path access blocked before read: {resolved}")


def read_text(path: Path) -> str:
    guard_not_val(path)
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def classify() -> dict:
    sources = {
        "tools/phase3b_train_only_execution.py": ROOT / "tools" / "phase3b_train_only_execution.py",
        "tools/stage_b_full_train_development.py": ROOT / "tools" / "stage_b_full_train_development.py",
        "execution/SHUFFLE_CONTROL_SPEC.yaml": EXEC / "SHUFFLE_CONTROL_SPEC.yaml",
        "audit/C_SHUFFLED_FINAL_PREVAL_QA.md": AUDIT / "C_SHUFFLED_FINAL_PREVAL_QA.md",
    }
    texts = {name: read_text(path) for name, path in sources.items()}
    joined = "\n".join(texts.values())
    evidence = []
    if "full_train_folds" in texts["tools/stage_b_full_train_development.py"] and "compute_full_shuffle(folds" in texts["tools/stage_b_full_train_development.py"]:
        evidence.append("Stage-B full builder computes C_SHUFFLED donors from the full eligible TRAIN fold table passed to compute_full_shuffle().")
    if "city_stratified_sample_and_folds" in texts["tools/phase3b_train_only_execution.py"] and "compute_shuffle(folds" in texts["tools/phase3b_train_only_execution.py"]:
        evidence.append("Stage-A development builder computes donors from its development fold table only.")
    if "train_only_bin_derivation: true" in joined and "train_seed:" in joined:
        evidence.append("Frozen protocol records TRAIN-only deterministic bin derivation and train seed, not official VAL outcome use.")
    has_full = "full eligible TRAIN" in joined or "all eligible TRAIN" in joined or "full_train_folds" in joined
    has_development = "development" in joined and "compute_shuffle(folds" in texts["tools/phase3b_train_only_execution.py"]
    if has_full and has_development:
        classification = "CSHUF_SCOPE_AMBIGUOUS_PREOUTCOME"
        reason = "pre-outcome sources show both a development-only historical donor pool and a full-TRAIN Stage-B donor pool; predictive performance is not used to choose between them"
    elif has_full:
        classification = "CSHUF_SCOPE_B_FULL_TRAIN_POOL_FROZEN"
        reason = "pre-outcome sources identify the complete eligible TRAIN donor pool"
    elif has_development:
        classification = "CSHUF_SCOPE_A_DEVELOPMENT_POOL_FROZEN"
        reason = "pre-outcome sources identify the deterministic development population as the donor pool"
    else:
        classification = "CSHUF_SCOPE_AMBIGUOUS_PREOUTCOME"
        reason = "pre-outcome sources do not identify a unique donor universe"
    return {
        "created_utc": utc_now(),
        "classification": classification,
        "reason": reason,
        "official_val_accessed": False,
        "sources": {name: {"path": str(path), "exists": path.exists()} for name, path in sources.items()},
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PRE-OUTCOME C_SHUFFLED donor-universe forensic classification")
    parser.add_argument("--write", action="store_true", help="write audit/C_SHUFFLED_DONOR_UNIVERSE_FORENSIC_RESOLUTION_v4.md")
    args = parser.parse_args()
    payload = classify()
    if args.write:
        lines = [
            "# C_SHUFFLED Donor-Universe Forensic Resolution v4",
            "",
            f"- classification: `{payload['classification']}`",
            f"- created_utc: `{payload['created_utc']}`",
            "- official_val_accessed: `false`",
            f"- reason: {payload['reason']}",
            "",
            "## Evidence",
            "",
            *[f"- {item}" for item in payload["evidence"]],
            "",
        ]
        write_text_atomic(OUT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
