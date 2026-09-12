from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import phase3b_train_only_execution as p3b
from maplite_features import MAP_LITE_FEATURES


ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "execution"
AUDIT = ROOT / "audit"
PROTOCOL = ROOT / "protocol"
PROCESSED = ROOT / "data" / "processed" / "phase3b"

DECISION_JSON = EXEC / "MAPLITE_FORENSIC_DECISION_v4.json"
FORENSIC_MANIFEST = EXEC / "FORENSIC_MAPLITE_V4_MANIFEST.json"
STAGEA_V3_MATRIX = PROCESSED / "TRAIN_50K_FEATURES_MAPLITE_CORRECTED_v2.parquet"
FULL_TRAIN_MATRIX = PROCESSED / "FULL_TRAIN_MAPLITE_CORRECTED_MATRIX.parquet"
V4_MATRIX = PROCESSED / "TRAIN_FEATURES_MAPLITE_KEYALIGNED_v4.parquet"
V4_MANIFEST = EXEC / "MAPLITE_KEYALIGNED_V4_DEVELOPMENT_MANIFEST.json"
V4_CV = EXEC / "PRIMARY_CV_RESULTS_MAPLITE_KEYALIGNED_v4.csv"
STATUS_DEFAULT = EXEC / "MAPLITE_KEYALIGNED_V4_CV_STATUS.json"
CV_LOG_DEFAULT = ROOT / "logs" / "maplite_keyaligned_v4_cv.log"
V3_CONFIGS = EXEC / "PRIMARY_MODEL_CONFIGS_MAPLITE_CORRECTED_v3.yaml"
FEATURE_SCHEMA = EXEC / "FEATURE_SCHEMA_v1.1.csv"
SHUFFLE_QA = EXEC / "SHUFFLE_CONTROL_QA.csv"

V3_SUPER_MD = AUDIT / "MAPLITE_V3_SUPERSESSION_DECLARATION_PREVAL_v4.md"
V3_SUPER_JSON = AUDIT / "MAPLITE_V3_SUPERSESSION_DECLARATION_PREVAL_v4.json"
CSHUF_CLAR_MD = PROTOCOL / "C_SHUFFLED_DONOR_UNIVERSE_CLARIFICATION_PREVAL_v4.md"
CSHUF_CLAR_JSON = PROTOCOL / "C_SHUFFLED_DONOR_UNIVERSE_CLARIFICATION_PREVAL_v4.json"
CSHUF_JUST_MD = AUDIT / "C_SHUFFLED_PREOUTCOME_AMENDMENT_JUSTIFICATION_v4.md"
V4_AUDIT_MD = AUDIT / "MAPLITE_KEYALIGNED_DEVELOPMENT_MATRIX_AUDIT_v4.md"
V4_AUDIT_JSON = AUDIT / "MAPLITE_KEYALIGNED_DEVELOPMENT_MATRIX_AUDIT_v4.json"
V4_IDENTITY_MD = AUDIT / "FULL_TRAIN_DEVELOPMENT_SUBSET_IDENTITY_QA_v4.md"
V4_COMPLETION_MD = AUDIT / "PRIMARY_CV_480_COMPLETION_AUDIT_MAPLITE_KEYALIGNED_v4.md"
V4_COMPLETION_JSON = AUDIT / "PRIMARY_CV_480_COMPLETION_AUDIT_MAPLITE_KEYALIGNED_v4.json"
V4_STATE_SUMMARY = EXEC / "PRIMARY_CV_STATE_SUMMARY_MAPLITE_KEYALIGNED_v4.csv"
V4_SELECTION_MD = EXEC / "PRIMARY_MODEL_SELECTION_MAPLITE_KEYALIGNED_v4.md"
V4_CONFIG_YAML = EXEC / "PRIMARY_MODEL_CONFIGS_MAPLITE_KEYALIGNED_v4.yaml"
V4_REGISTRY_JSON = EXEC / "PRIMARY_HGB_CANDIDATE_REGISTRY_MAPLITE_KEYALIGNED_v4.json"

STATE_ORDER = p3b.STATE_ORDER
HGB_UNIT_COLUMNS = p3b.HGB_UNIT_COLUMNS
PRIMARY_CV_COLUMNS = p3b.PRIMARY_CV_COLUMNS
KEY_COLS = ["scenario_id", "track_id"]
TARGET_COLUMNS = p3b.TARGET_COLUMNS
ROLE_COLUMNS = ["is_nearest"]
PROVENANCE_COLUMNS = [
    "input_matrix_hash",
    "candidate_registry_hash",
    "feature_schema_hash",
    "scientific_key_hash",
    "software_environment_hash",
    "unit_provenance_hash",
]
V4_CV_COLUMNS = PRIMARY_CV_COLUMNS + PROVENANCE_COLUMNS
EXPECTED_ROWS = 216_170
EXPECTED_UNITS = 480


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def guard_not_val(path: Path) -> None:
    resolved = path.resolve()
    parts = [p.lower() for p in resolved.parts]
    if "official_s3" in parts and any(p in {"val", "validation"} for p in parts):
        logging.critical("official VAL path access blocked before read: %s", resolved)
        raise RuntimeError(f"official AV2 VAL path access blocked before read: {resolved}")


def sha256_file(path: Path) -> str:
    guard_not_val(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_frame(df: pd.DataFrame, cols: list[str]) -> str:
    h = hashlib.sha256()
    for value in pd.util.hash_pandas_object(df[cols], index=False).to_numpy(dtype=np.uint64):
        h.update(int(value).to_bytes(8, "little", signed=False))
    return h.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return sha256_file(path)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return sha256_file(path)


def configure_logging(path: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    guard_not_val(path)
    return pd.read_parquet(path, columns=columns)


def map_cols() -> list[str]:
    return MAP_LITE_FEATURES + [f"{c}_missing" for c in MAP_LITE_FEATURES]


def shuffled_cols(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith("shuffled_av_future_")]


def assert_unique_keys(df: pd.DataFrame, keys: list[str], label: str) -> None:
    missing = [c for c in keys if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing key columns: {missing}")
    dups = int(df.duplicated(keys).sum())
    if dups:
        raise RuntimeError(f"{label} has duplicate scientific keys: {dups}")


def keyed_merge(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], label: str) -> pd.DataFrame:
    assert_unique_keys(left, keys, f"{label}.left")
    assert_unique_keys(right, keys, f"{label}.right")
    out = left.merge(right, on=keys, how="left", validate="one_to_one", indicator=True)
    unmatched = int((out["_merge"] != "both").sum())
    if unmatched:
        raise RuntimeError(f"{label} has unmatched keyed rows: {unmatched}")
    return out.drop(columns=["_merge"])


def software_environment() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "sklearn": getattr(__import__("sklearn"), "__version__", "unavailable"),
        "tabulate_reporting_only": getattr(__import__("tabulate"), "__version__", "unavailable"),
    }


def load_frozen_candidate_registry() -> list[dict[str, Any]]:
    if not V3_CONFIGS.exists():
        raise RuntimeError(f"frozen v3 candidate registry source missing: {V3_CONFIGS}")
    payload = yaml.safe_load(V3_CONFIGS.read_text(encoding="utf-8"))
    registry = payload.get("candidate_registry")
    if not isinstance(registry, list) or len(registry) != 24:
        raise RuntimeError("frozen candidate registry must contain exactly 24 records")
    if registry != p3b.config_candidates(24):
        raise RuntimeError("on-disk frozen candidate registry does not match historical deterministic registry")
    write_json_atomic(
        V4_REGISTRY_JSON,
        {
            "created_utc": utc_now(),
            "source": str(V3_CONFIGS),
            "randomized_search_seed": p3b.TUNING_SEED,
            "candidate_count": 24,
            "candidate_registry": registry,
            "official_val_accessed": False,
        },
    )
    return registry


def candidate_registry_hash() -> str:
    return sha256_json(load_frozen_candidate_registry())


def require_authorized() -> None:
    if not DECISION_JSON.exists():
        raise RuntimeError("Refusing v4 development: execution/MAPLITE_FORENSIC_DECISION_v4.json is missing")
    decision = json.loads(DECISION_JSON.read_text(encoding="utf-8"))
    if decision.get("final_decision") != "V3_MAPLITE_MISALIGNMENT_CONFIRMED":
        raise RuntimeError(f"Refusing v4 development: forensic decision is {decision.get('final_decision')!r}")


def create_freeze_documents() -> dict[str, Any]:
    require_authorized()
    frozen_paths = [V3_SUPER_MD, V3_SUPER_JSON, CSHUF_CLAR_MD, CSHUF_CLAR_JSON, CSHUF_JUST_MD]
    if all(path.exists() for path in frozen_paths):
        return {
            "v3_supersession_md_sha256": sha256_file(V3_SUPER_MD),
            "v3_supersession_json_sha256": sha256_file(V3_SUPER_JSON),
            "cshuffled_clarification_md_sha256": sha256_file(CSHUF_CLAR_MD),
            "cshuffled_clarification_json_sha256": sha256_file(CSHUF_CLAR_JSON),
            "cshuffled_justification_md_sha256": sha256_file(CSHUF_JUST_MD),
            "official_val_accessed": False,
        }
    forensic_manifest = json.loads(FORENSIC_MANIFEST.read_text(encoding="utf-8"))
    inputs = forensic_manifest["inputs"]
    stage_hash = inputs[str(STAGEA_V3_MATRIX)]["sha256"]
    full_hash = inputs[str(FULL_TRAIN_MATRIX)]["sha256"]
    folds_path = EXEC / "TRAIN_SCENARIO_FOLDS_v2.csv"
    folds_hash = inputs.get(str(folds_path), {}).get("sha256", sha256_file(folds_path))
    feature_hash = sha256_file(ROOT / "tools" / "maplite_features.py")
    key_hash = "08ff8ef8287bead94cb2a380cfd8974e98d88c9efac90d66d0176197262e8351"
    decision_hash = sha256_file(DECISION_JSON)
    three_way_hash = sha256_file(AUDIT / "MAPLITE_THREE_WAY_RAW_RECOMPUTATION_AUDIT_v4.csv")

    supersession = {
        "artifact_id": "MAPLITE_V3_SUPERSESSION_DECLARATION_PREVAL_v4",
        "created_utc": utc_now(),
        "v3_status": "PRE-VAL MAP-LITE IMPLEMENTATION CORRECTION WITH SUBSEQUENTLY DISCOVERED ROW-KEY ALIGNMENT DEFECT",
        "v4_status": "PRE-VAL KEY-ALIGNED IMPLEMENTATION OF THE ALREADY-FROZEN MAP-LITE FEATURES",
        "final_verdict": "V3_MAPLITE_MISALIGNMENT_CONFIRMED",
        "official_val_accessed": False,
        "scientific_feature_definition_changes": False,
        "target_fold_role_non_map_changes": False,
        "v3_preserved_as_provenance": True,
        "v4_supersedes_v3_for_authoritative_model_development": True,
        "evidence_hashes": {
            "stage_a_v3_matrix": stage_hash,
            "full_train_matrix": full_hash,
            "canonical_feature_implementation": feature_hash,
            "development_key_hash": key_hash,
            "train_scenario_folds_v2": folds_hash,
            "forensic_decision_json": decision_hash,
            "three_way_raw_recomputation_csv": three_way_hash,
        },
    }
    write_json_atomic(V3_SUPER_JSON, supersession)
    super_md_hash = write_text_atomic(
        V3_SUPER_MD,
        "\n".join(
            [
                "# MAP-LITE v3 Supersession Declaration PREVAL v4",
                "",
                f"- created_utc: `{supersession['created_utc']}`",
                f"- final_verdict: `{supersession['final_verdict']}`",
                "- official_val_accessed: `false`",
                "- v3: `PRE-VAL MAP-LITE IMPLEMENTATION CORRECTION WITH SUBSEQUENTLY DISCOVERED ROW-KEY ALIGNMENT DEFECT`",
                "- v4: `PRE-VAL KEY-ALIGNED IMPLEMENTATION OF THE ALREADY-FROZEN MAP-LITE FEATURES`",
                "",
                "The original frozen protocol already specified MAP-LITE. v2 did not implement it. v3 attempted the frozen-feature implementation correction, but later full-scale TRAIN-only identity QA exposed row-key misalignment.",
                "",
                "The full 216,170-row direct forensic recomputation confirmed the defect: canonical raw recomputation agrees with the keyed full TRAIN matrix, and the historical positional Stage-A assignment reconstructs the v3 mismatches.",
                "",
                "This declaration does not change the scientific hypothesis, add predictors, alter target/fold/role/non-map definitions, or adapt to official VAL. v3 remains immutable historical provenance. v4 supersedes v3 for authoritative model development.",
                "",
                "## Evidence Hashes",
                "",
                *[f"- {k}: `{v}`" for k, v in supersession["evidence_hashes"].items()],
                "",
            ]
        ),
    )

    clarification = {
        "artifact_id": "C_SHUFFLED_DONOR_UNIVERSE_CLARIFICATION_PREVAL_v4",
        "created_utc": utc_now(),
        "classification": "CSHUF_SCOPE_AMBIGUOUS_PREOUTCOME",
        "general_rule": "STAGE-LOCAL DONOR UNIVERSE",
        "definition": "The C_SHUFFLED donor universe is the population legitimately belonging to the current modeling/evaluation stage.",
        "stage_universes": {
            "stage_a_v4_development_cv": "frozen deterministic 50k TRAIN development population",
            "final_full_train_fitting": "complete eligible TRAIN population",
            "future_official_val_evaluation": "official VAL population only",
        },
        "unchanged_rules": [
            "within-city shuffle",
            "TRAIN seed = 2026090731",
            "VAL seed = 2026090732",
            "TRAIN city quartiles of speed",
            "TRAIN city quartiles of mean absolute heading change t45-49",
            "sparse-bin merging",
            "no self-assignment unless no alternative exists",
        ],
        "official_val_accessed": False,
        "predictive_performance_used": False,
    }
    write_json_atomic(CSHUF_CLAR_JSON, clarification)
    clar_md_hash = write_text_atomic(
        CSHUF_CLAR_MD,
        "\n".join(
            [
                "# C_SHUFFLED Donor-Universe Clarification PREVAL v4",
                "",
                f"- created_utc: `{clarification['created_utc']}`",
                "- classification: `CSHUF_SCOPE_AMBIGUOUS_PREOUTCOME`",
                "- general_rule: `STAGE-LOCAL DONOR UNIVERSE`",
                "- official_val_accessed: `false`",
                "- predictive_performance_used: `false`",
                "",
                "The C_SHUFFLED donor universe is the population legitimately belonging to the current modeling/evaluation stage.",
                "",
                "- Stage-A/v4 development CV donor universe: frozen deterministic 50k TRAIN development population",
                "- final full-TRAIN fitting donor universe: complete eligible TRAIN population",
                "- future official VAL evaluation donor universe: official VAL population only",
                "",
                "Never transfer donors across TRAIN and VAL. Existing within-city shuffle, TRAIN/VAL seeds, TRAIN city quartile binning, sparse-bin merging, and no-self-assignment rules remain unchanged.",
                "",
            ]
        ),
    )
    clar_json_hash = sha256_file(CSHUF_CLAR_JSON)
    just_md_hash = write_text_atomic(
        CSHUF_JUST_MD,
        "\n".join(
            [
                "# C_SHUFFLED PREOUTCOME Amendment Justification v4",
                "",
                "- status: `FROZEN_PREOUTCOME`",
                "- official_val_accessed: `false`",
                "",
                "Historical Stage-A development used the development population donor universe, while Stage-B final full-TRAIN used the complete eligible TRAIN donor universe. The frozen protocol specified deterministic TRAIN shuffle logic and seeds, but did not uniquely state the donor-universe scope across development versus final fitting.",
                "",
                "The stage-local rule resolves that ambiguity before any v4 predictive CV result is seen, aligns the donor universe with the population available at the relevant scientific stage, preserves development isolation, and prevents TRAIN/VAL donor leakage. The rule is not chosen or modified based on predictive performance.",
                "",
                f"- clarification_md_sha256: `{clar_md_hash}`",
                f"- clarification_json_sha256: `{clar_json_hash}`",
                "",
            ]
        ),
    )
    return {
        "v3_supersession_md_sha256": super_md_hash,
        "v3_supersession_json_sha256": sha256_file(V3_SUPER_JSON),
        "cshuffled_clarification_md_sha256": clar_md_hash,
        "cshuffled_clarification_json_sha256": clar_json_hash,
        "cshuffled_justification_md_sha256": just_md_hash,
        "official_val_accessed": False,
    }


def read_raw_maplite_checkpoints() -> pd.DataFrame:
    from forensic_maplite_identity_v4 import CHECKPOINT_DEFAULT

    full_dir = CHECKPOINT_DEFAULT / "full"
    raw_parts = sorted(full_dir.glob("scenario_chunk_*.parquet"))
    if not raw_parts:
        raise RuntimeError(f"No full forensic MAP-LITE checkpoints found under {full_dir}")
    raw = pd.concat([read_parquet(p, columns=KEY_COLS + map_cols()) for p in raw_parts], ignore_index=True)
    assert_unique_keys(raw, KEY_COLS, "full forensic raw MAP-LITE checkpoints")
    if len(raw) != EXPECTED_ROWS:
        raise RuntimeError(f"full forensic raw MAP-LITE checkpoints row count {len(raw)} != {EXPECTED_ROWS}")
    return raw


def compare_frames_exact(left: pd.DataFrame, right: pd.DataFrame, cols: list[str]) -> tuple[bool, list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    for col in cols:
        a = left[col]
        b = right[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av = a.to_numpy(dtype=float)
            bv = b.to_numpy(dtype=float)
            ok = bool(np.allclose(av, bv, rtol=0.0, atol=1e-9, equal_nan=True))
            max_abs = float(np.nanmax(np.abs(av - bv))) if len(av) else 0.0
        else:
            ok = bool(a.astype(str).equals(b.astype(str)))
            max_abs = None
        if not ok:
            mismatches.append({"column": col, "max_abs_discrepancy": max_abs})
    return len(mismatches) == 0, mismatches


def build_v4_matrix() -> dict[str, Any]:
    require_authorized()
    if not CSHUF_CLAR_JSON.exists() or not CSHUF_CLAR_MD.exists():
        create_freeze_documents()

    stage = read_parquet(STAGEA_V3_MATRIX)
    raw = read_raw_maplite_checkpoints()
    assert_unique_keys(stage, KEY_COLS, "Stage-A v3 development matrix")
    if len(stage) != EXPECTED_ROWS:
        raise RuntimeError(f"Stage-A v3 development matrix row count {len(stage)} != {EXPECTED_ROWS}")

    order = stage[KEY_COLS].copy()
    aligned_raw = keyed_merge(order, raw[KEY_COLS + map_cols()], KEY_COLS, "v4 raw MAP-LITE assignment")
    v4 = stage.copy()
    for col in map_cols():
        v4[col] = aligned_raw[col].to_numpy()

    tmp = V4_MATRIX.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(v4, preserve_index=False), tmp)
    os.replace(tmp, V4_MATRIX)

    full = read_parquet(FULL_TRAIN_MATRIX, columns=KEY_COLS + map_cols())
    full_dev = keyed_merge(order, full, KEY_COLS, "full TRAIN development MAP-LITE subset")
    non_map_cols = [c for c in stage.columns if c not in map_cols()]
    shuffled = shuffled_cols(stage.columns.tolist())
    target_role_cols = [c for c in TARGET_COLUMNS + ["fold"] + ROLE_COLUMNS if c in stage.columns]
    key_ok = bool(stage[KEY_COLS].astype(str).equals(v4[KEY_COLS].astype(str)))
    target_role_ok, target_role_mismatches = compare_frames_exact(stage, v4, target_role_cols)
    non_map_ok, non_map_mismatches = compare_frames_exact(stage, v4, non_map_cols)
    raw_ok, raw_mismatches = compare_frames_exact(v4, aligned_raw, map_cols())
    full_map_ok, full_map_mismatches = compare_frames_exact(v4, full_dev, map_cols())
    cshuffle_ok, cshuffle_mismatches = compare_frames_exact(stage, v4, shuffled)

    key_hash = sha256_frame(v4, KEY_COLS)
    audit = {
        "audit_id": "MAPLITE_KEYALIGNED_DEVELOPMENT_MATRIX_AUDIT_v4",
        "created_utc": utc_now(),
        "matrix_path": str(V4_MATRIX),
        "matrix_sha256": sha256_file(V4_MATRIX),
        "rows": int(len(v4)),
        "scenario_count": int(v4["scenario_id"].nunique()),
        "scientific_key_hash": key_hash,
        "stage_a_v3_matrix_sha256": sha256_file(STAGEA_V3_MATRIX),
        "full_train_matrix_sha256": sha256_file(FULL_TRAIN_MATRIX),
        "cshuffled_clarification_sha256": sha256_file(CSHUF_CLAR_MD),
        "key_preservation": key_ok,
        "target_fold_role_preservation": target_role_ok,
        "target_fold_role_mismatches": target_role_mismatches[:20],
        "non_map_preservation": non_map_ok,
        "non_map_mismatches": non_map_mismatches[:20],
        "canonical_maplite_raw_agreement": raw_ok,
        "canonical_maplite_raw_mismatches": raw_mismatches[:20],
        "development_full_maplite_identity": full_map_ok,
        "development_full_maplite_mismatches": full_map_mismatches[:20],
        "cshuffled_development_determinism": cshuffle_ok,
        "cshuffled_mismatches": cshuffle_mismatches[:20],
        "development_shuffle_reference_sha256": sha256_file(SHUFFLE_QA) if SHUFFLE_QA.exists() else None,
        "official_val_accessed": False,
    }
    audit["status"] = (
        "PASS"
        if audit["rows"] == EXPECTED_ROWS
        and key_ok
        and target_role_ok
        and non_map_ok
        and raw_ok
        and full_map_ok
        and cshuffle_ok
        else "FAIL"
    )
    write_json_atomic(V4_AUDIT_JSON, audit)
    write_text_atomic(
        V4_AUDIT_MD,
        "\n".join(
            [
                "# MAP-LITE Key-Aligned Development Matrix Audit v4",
                "",
                f"- status: `{audit['status']}`",
                f"- matrix_sha256: `{audit['matrix_sha256']}`",
                f"- rows: `{audit['rows']}`",
                f"- scientific_key_hash: `{audit['scientific_key_hash']}`",
                f"- key_preservation: `{str(key_ok).lower()}`",
                f"- target_fold_role_preservation: `{str(target_role_ok).lower()}`",
                f"- non_map_preservation: `{str(non_map_ok).lower()}`",
                f"- canonical_maplite_raw_agreement: `{str(raw_ok).lower()}`",
                f"- development_full_maplite_identity: `{str(full_map_ok).lower()}`",
                f"- cshuffled_development_determinism: `{str(cshuffle_ok).lower()}`",
                "- official_val_accessed: `false`",
                "",
            ]
        ),
    )
    write_text_atomic(
        V4_IDENTITY_MD,
        "\n".join(
            [
                "# FULL TRAIN Development Subset Identity QA v4",
                "",
                f"- status: `{'PASS' if full_map_ok else 'FAIL'}`",
                "- scope: MAP-LITE base features and missingness flags only for scientifically scope-identical columns",
                f"- development_rows: `{len(v4)}`",
                f"- full_train_development_subset_rows: `{len(full_dev)}`",
                f"- development_full_maplite_identity: `{str(full_map_ok).lower()}`",
                "- cshuffled_full_train_equality_required: `false`",
                "- cshuffled_reason: stage-local donor-universe clarification intentionally gives development and full-TRAIN different donor universes",
                "- official_val_accessed: `false`",
                "",
            ]
        ),
    )
    write_json_atomic(V4_MANIFEST, audit)
    if audit["status"] != "PASS":
        raise RuntimeError("v4 development matrix audit failed")
    return audit


def current_provenance() -> dict[str, str]:
    registry_hash = candidate_registry_hash()
    schema_hash = sha256_file(FEATURE_SCHEMA)
    software_hash = sha256_json(software_environment())
    if not V4_MATRIX.exists():
        return {
            "input_matrix_hash": "",
            "candidate_registry_hash": registry_hash,
            "feature_schema_hash": schema_hash,
            "scientific_key_hash": "",
            "software_environment_hash": software_hash,
        }
    key_hash = sha256_frame(read_parquet(V4_MATRIX, columns=KEY_COLS), KEY_COLS)
    return {
        "input_matrix_hash": sha256_file(V4_MATRIX),
        "candidate_registry_hash": registry_hash,
        "feature_schema_hash": schema_hash,
        "scientific_key_hash": key_hash,
        "software_environment_hash": software_hash,
    }


def add_unit_provenance(row: dict[str, Any], provenance: dict[str, str]) -> dict[str, Any]:
    unit_payload = {**provenance, "state": row["state"], "candidate_index": int(row["candidate_index"]), "fold": int(row["fold"])}
    out = dict(row)
    out.update(provenance)
    out["unit_provenance_hash"] = sha256_json(unit_payload)
    return out


def completed_keys(path: Path, candidate_limit: int, provenance: dict[str, str] | None = None) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    missing = [c for c in V4_CV_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing v4 CV columns: {missing}")
    duplicates = df.duplicated(HGB_UNIT_COLUMNS, keep=False)
    if bool(duplicates.any()):
        raise RuntimeError(f"{path} contains duplicate deterministic HGB units")
    expected = p3b.hgb_expected_units(candidate_limit)
    keys = {(str(r.state), int(r.candidate_index), int(r.fold)) for r in df.itertuples(index=False)}
    unexpected = keys - expected
    if unexpected:
        raise RuntimeError(f"{path} contains unexpected HGB units: {sorted(unexpected)[:3]}")
    if provenance is not None:
        for col, expected_value in provenance.items():
            bad = df[col].astype(str) != str(expected_value)
            if bool(bad.any()):
                raise RuntimeError(f"{path} contains CV unit provenance mismatch for {col}")
        for r in df.itertuples(index=False):
            base = {k: provenance[k] for k in provenance}
            base.update({"state": r.state, "candidate_index": int(r.candidate_index), "fold": int(r.fold)})
            if str(r.unit_provenance_hash) != sha256_json(base):
                raise RuntimeError(f"{path} contains invalid unit_provenance_hash for {(r.state, r.candidate_index, r.fold)}")
    return keys


def update_status(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["last_updated_at"] = utc_now()
    payload["official_val_accessed"] = False
    write_json_atomic(path, payload)


def status_payload(status_file: Path, candidate_limit: int, phase: str = "preflight") -> dict[str, Any]:
    provenance = current_provenance()
    completed = completed_keys(V4_CV, candidate_limit, provenance if V4_MATRIX.exists() else None)
    expected = p3b.hgb_expected_units(candidate_limit)
    payload = {
        "phase": phase,
        "created_utc": utc_now(),
        "last_updated_at": utc_now(),
        "process_state": "preflight_pass",
        "matrix_status": "present" if V4_MATRIX.exists() else "missing",
        **provenance,
        "total_expected_units": len(expected),
        "completed_units": len(completed),
        "remaining_units": len(expected - completed),
        "state_completion": {s: sum(1 for k in completed if k[0] == s) for s in STATE_ORDER},
        "current_state": None,
        "current_candidate": None,
        "current_fold": None,
        "failed_units": 0,
        "elapsed_seconds": 0.0,
        "estimated_remaining_seconds": None,
        "software": software_environment(),
        "scientific_dependencies": {
            "python": "3.12.3",
            "scikit_learn": "1.8",
            "numpy": "2.4.3",
            "pandas": "3.0.1",
            "pyarrow": "24.0",
        },
        "reporting_dependencies": {"tabulate": "0.10.0"},
        "official_val_accessed": False,
    }
    update_status(status_file, payload)
    return payload


def checkpoint_resume_smoke(provenance: dict[str, str]) -> dict[str, Any]:
    row = {
        "state": "H",
        "candidate_index": 0,
        "fold": 0,
        "learning_rate": 0.06,
        "max_iter": 600,
        "max_leaf_nodes": 127,
        "l2_regularization": 0.01,
        "min_samples_leaf": 50,
        "max_bins": 255,
        "dx_6s_mse": 1.0,
        "dy_6s_mse": 2.0,
        "mean_endpoint_mse": 3.0,
        "mean_fde": 1.5,
        "median_fde": 1.4,
        "n_train_rows": 10,
        "n_dev_rows": 5,
    }
    with tempfile.TemporaryDirectory(prefix="v4_resume_smoke_") as tmpdir:
        path = Path(tmpdir) / "cv.csv"
        append_row_atomic(path, add_unit_provenance(row, provenance), 24)
        ok = completed_keys(path, 24, provenance) == {("H", 0, 0)}
        df = pd.read_csv(path)
        df.loc[0, "input_matrix_hash"] = "bad"
        df.to_csv(path, index=False)
        rejected = False
        try:
            completed_keys(path, 24, provenance)
        except RuntimeError:
            rejected = True
    return {"status": "PASS" if ok and rejected else "FAIL", "valid_reuse_passed": ok, "mismatch_rejected": rejected}


def official_val_firewall_smoke() -> dict[str, Any]:
    try:
        guard_not_val(ROOT / "data" / "raw" / "av2_motion_forecasting" / "official_s3" / "val" / "sentinel.parquet")
    except RuntimeError:
        return {"status": "PASS", "official_val_accessed": False}
    return {"status": "FAIL", "official_val_accessed": False}


def preflight(status_file: Path, candidate_limit: int) -> dict[str, Any]:
    docs = create_freeze_documents()
    registry = load_frozen_candidate_registry()
    reg_hash = candidate_registry_hash()
    provenance = current_provenance()
    resume = checkpoint_resume_smoke({k: v or "not-built" for k, v in provenance.items()})
    firewall = official_val_firewall_smoke()
    status = status_payload(status_file, candidate_limit)
    payload = {
        "phase": "preflight",
        "process_state": "preflight_pass",
        "documents": docs,
        "candidate_registry_hash": reg_hash,
        "candidate_count": len(registry),
        "expected_cv_units": len(p3b.hgb_expected_units(candidate_limit)),
        "checkpoint_resume_test": resume,
        "official_val_firewall": firewall,
        "status": status,
        "official_val_accessed": False,
    }
    if payload["expected_cv_units"] != EXPECTED_UNITS or resume["status"] != "PASS" or firewall["status"] != "PASS":
        payload["process_state"] = "preflight_fail"
        update_status(status_file, {**status, "process_state": "preflight_fail"})
        raise RuntimeError("v4 preflight failed")
    return payload


def append_row_atomic(path: Path, row: dict[str, Any], candidate_limit: int) -> str:
    provenance = {k: row[k] for k in PROVENANCE_COLUMNS if k != "unit_provenance_hash"}
    before = completed_keys(path, candidate_limit, provenance) if path.exists() else set()
    key = (str(row["state"]), int(row["candidate_index"]), int(row["fold"]))
    if key in before:
        raise RuntimeError(f"duplicate CV unit refused: {key}")
    exists = path.exists() and path.stat().st_size > 0
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=V4_CV_COLUMNS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        else:
            with path.open("r", encoding="utf-8", newline="") as src:
                out.write(src.read())
            if path.stat().st_size > 0 and not path.read_bytes().endswith((b"\n", b"\r")):
                out.write("\n")
        writer.writerow({c: row[c] for c in V4_CV_COLUMNS})
    os.replace(tmp, path)
    completed_keys(path, candidate_limit, provenance)
    return row["unit_provenance_hash"]


def run_cv(args: argparse.Namespace) -> None:
    require_authorized()
    if not V4_MATRIX.exists():
        build_v4_matrix()
    provenance = current_provenance()
    started = time.time()
    expected = p3b.hgb_expected_units(args.hgb_candidates)
    completed = completed_keys(V4_CV, args.hgb_candidates, provenance)
    status = status_payload(args.status_file, args.hgb_candidates, phase="cv")
    if completed and not args.resume:
        raise RuntimeError(f"{V4_CV} contains {len(completed)} completed units; rerun with --resume")
    for state, candidate_index, fold in p3b.hgb_units_in_order(args.hgb_candidates):
        key = (state, candidate_index, fold)
        if key in completed:
            continue
        status.update({"phase": "cv", "current_state": state, "current_candidate": candidate_index, "current_fold": fold, "process_state": "running"})
        update_status(args.status_file, status)
        df = read_parquet(V4_MATRIX)
        row = p3b.fit_evaluate_hgb_unit(df, state, candidate_index, fold, args.hgb_candidates)
        row_hash = append_row_atomic(V4_CV, add_unit_provenance(row, provenance), args.hgb_candidates)
        completed.add(key)
        elapsed = time.time() - started
        rate = len(completed) / max(1e-9, elapsed)
        status.update(
            {
                "completed_units": len(completed),
                "remaining_units": len(expected - completed),
                "state_completion": {s: sum(1 for k in completed if k[0] == s) for s in STATE_ORDER},
                "elapsed_seconds": round(elapsed, 3),
                "estimated_remaining_seconds": round(len(expected - completed) / rate, 1) if rate > 0 else None,
                "last_unit_hash": row_hash,
            }
        )
        update_status(args.status_file, status)
        logging.info("completed state=%s candidate=%s fold=%s hash=%s", state, candidate_index, fold, row_hash)
    status.update({"phase": "complete", "process_state": "complete", "current_state": None, "current_candidate": None, "current_fold": None})
    update_status(args.status_file, status)


def finalize(status_file: Path, candidate_limit: int) -> dict[str, Any]:
    provenance = current_provenance()
    completed = completed_keys(V4_CV, candidate_limit, provenance)
    expected = p3b.hgb_expected_units(candidate_limit)
    if completed != expected:
        raise RuntimeError(f"finalize requires 480 valid units; found {len(completed)}")
    res = pd.read_csv(V4_CV)
    registry = load_frozen_candidate_registry()
    summary = res.groupby(["state", "candidate_index"], as_index=False).agg(
        mean_endpoint_mse=("mean_endpoint_mse", "mean"), mean_fde=("mean_fde", "mean")
    )
    write_csv_atomic(summary, V4_STATE_SUMMARY)
    selected: dict[str, Any] = {}
    for state in STATE_ORDER:
        best = summary[summary["state"] == state].sort_values(["mean_endpoint_mse", "candidate_index"]).iloc[0]
        selected[state] = {
            **registry[int(best.candidate_index)],
            "candidate_index": int(best.candidate_index),
            "mean_cv_endpoint_mse": float(best.mean_endpoint_mse),
        }
    payload = {
        "selection_rule": "TRAIN-only grouped-CV minimum mean endpoint MSE; ties by candidate_index order",
        "downstream_rpva_gains_used": False,
        "official_val_used": False,
        "randomized_search_seed": p3b.TUNING_SEED,
        "candidate_count_per_state": candidate_limit,
        "candidate_registry": registry,
        "selected_configs": selected,
    }
    V4_CONFIG_YAML.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    completion = {
        "audit_id": "PRIMARY_CV_480_COMPLETION_AUDIT_MAPLITE_KEYALIGNED_v4",
        "created_utc": utc_now(),
        "input_path": str(V4_CV),
        "input_sha256": sha256_file(V4_CV),
        "observed_rows": int(len(res)),
        "expected_rows": EXPECTED_UNITS,
        "unique_keys": len(completed),
        "state_counts": {s: int((res["state"] == s).sum()) for s in STATE_ORDER},
        "candidate_registry_hash": candidate_registry_hash(),
        "official_val_accessed": False,
        "status": "PASS",
    }
    write_json_atomic(V4_COMPLETION_JSON, completion)
    write_text_atomic(
        V4_COMPLETION_MD,
        "\n".join(
            [
                "# Primary CV 480 Completion Audit MAP-LITE Key-Aligned v4",
                "",
                "- status: `PASS`",
                f"- observed_rows: `{completion['observed_rows']}`",
                f"- input_sha256: `{completion['input_sha256']}`",
                "- official_val_accessed: `false`",
                "",
            ]
        ),
    )
    write_text_atomic(
        V4_SELECTION_MD,
        "\n".join(
            [
                "# Primary Model Selection MAP-LITE Key-Aligned v4",
                "",
                "Empirical winners selected using TRAIN-only grouped city-stratified folds and predictive endpoint loss only.",
                "",
                summary.sort_values(["state", "mean_endpoint_mse", "candidate_index"]).groupby("state").head(1).to_markdown(index=False),
                "",
            ]
        ),
    )
    update_status(status_file, {**status_payload(status_file, candidate_limit, phase="complete"), "process_state": "complete"})
    return completion


def main() -> int:
    parser = argparse.ArgumentParser(description="TRAIN-only key-aligned MAP-LITE v4 development package")
    parser.add_argument("--mode", choices=["preflight", "build-matrix", "cv", "finalize", "all"], default="preflight")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--authorized-after-human-review", action="store_true")
    parser.add_argument("--status-file", type=Path, default=STATUS_DEFAULT)
    parser.add_argument("--log-file", type=Path, default=CV_LOG_DEFAULT)
    parser.add_argument("--hgb-candidates", type=int, default=24)
    parser.add_argument("--preflight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--build-matrix", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-cv", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.preflight:
        args.mode = "preflight"
    if args.build_matrix:
        args.mode = "build-matrix"
    if args.run_cv:
        args.mode = "cv"
    configure_logging(args.log_file)
    logging.info("startup config=%s", {k: str(v) for k, v in vars(args).items()})
    if args.workers < 1:
        raise RuntimeError("--workers must be >= 1")
    if args.hgb_candidates != 24:
        raise RuntimeError("v4 authoritative run requires exactly 24 frozen HGB candidates")

    if args.mode == "preflight":
        print(json.dumps(preflight(args.status_file, args.hgb_candidates), indent=2, sort_keys=True))
    elif args.mode == "build-matrix":
        print(json.dumps(build_v4_matrix(), indent=2, sort_keys=True))
    elif args.mode == "cv":
        if not args.authorized_after_human_review:
            raise RuntimeError("Refusing v4 CV without --authorized-after-human-review")
        run_cv(args)
    elif args.mode == "finalize":
        print(json.dumps(finalize(args.status_file, args.hgb_candidates), indent=2, sort_keys=True))
    elif args.mode == "all":
        if not args.authorized_after_human_review:
            raise RuntimeError("Refusing v4 all-mode without --authorized-after-human-review")
        build_v4_matrix()
        run_cv(args)
        print(json.dumps(finalize(args.status_file, args.hgb_candidates), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
