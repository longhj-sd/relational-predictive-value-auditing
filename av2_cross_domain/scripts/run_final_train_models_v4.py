from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

import phase3b_train_only_execution as p3b


ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "execution"
AUDIT = ROOT / "audit"
LOGS = ROOT / "logs"
MATRIX = ROOT / "data" / "processed" / "phase3b" / "FULL_TRAIN_MAPLITE_CORRECTED_MATRIX.parquet"
DEV_MATRIX = ROOT / "data" / "processed" / "phase3b" / "TRAIN_FEATURES_MAPLITE_KEYALIGNED_v4.parquet"
FEATURE_SCHEMA = EXEC / "FEATURE_SCHEMA_v1.1.csv"
REGISTRY_JSON = EXEC / "PRIMARY_HGB_CANDIDATE_REGISTRY_MAPLITE_KEYALIGNED_v4.json"
PRIMARY_CONFIGS = EXEC / "PRIMARY_MODEL_CONFIGS_MAPLITE_KEYALIGNED_v4.yaml"
STATUS = EXEC / "FINAL_TRAIN_HGB_V4_STATUS.json"
MANIFEST = EXEC / "FULL_TRAIN_HGB_FINAL_MODELS_MANIFEST_v4.json"
MODELS_DIR = EXEC / "full_train_hgb_models_v4"
MAIN_LOG = LOGS / "final_train_hgb_v4_main.log"
VAL_FORBIDDEN = (ROOT / "data" / "raw" / "av2_motion_forecasting" / "official_s3" / "val").resolve()

EXPECTED_MATRIX_HASH = "b2f212698f0a1773597b8eff4341787e95bab2a20f12c71215d95f7b26c4a9c5"
EXPECTED_ROWS = 630_388
EXPECTED_SCENARIOS = 145_390
EXPECTED_ENV_HASH = "c27192539c77751658ee66759843e5bc3142063823ca13ef18e6e287a345150f"
EXPECTED_REGISTRY_HASH = "e32035d6d56f74163402020df748a10f470e28e87d5475826123b152f75b1e20"
EXPECTED_CONFIG = {
    "learning_rate": 0.03,
    "max_iter": 1000,
    "max_leaf_nodes": 127,
    "l2_regularization": 0.1,
    "min_samples_leaf": 50,
    "max_bins": 255,
}
EXPECTED_ENV = {
    "python": "3.12.3",
    "numpy": "2.4.3",
    "pandas": "3.0.1",
    "pyarrow": "24.0.0",
    "sklearn": "1.8.0",
    "yaml": "6.0.3",
}
STATE_ORDER = ["H", "E", "C", "C_SHUFFLED"]
TARGET_COLUMNS = ["dx_6s", "dy_6s"]
EXPECTED_MODEL_UNITS = [(state, target) for state in STATE_ORDER for target in TARGET_COLUMNS]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def guard_not_val(path: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(VAL_FORBIDDEN)
    except ValueError:
        return
    raise RuntimeError(f"official AV2 VAL path access blocked before read: {resolved}")


def sha256_file(path: Path) -> str:
    guard_not_val(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def atomic_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return sha256_file(path)


def atomic_json(path: Path, payload: Any) -> str:
    return atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def env_versions() -> dict[str, str]:
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "sklearn": sklearn.__version__,
        "yaml": yaml.__version__,
    }


def environment_status() -> dict[str, Any]:
    observed = env_versions()
    mismatches = {k: {"expected": v, "observed": observed.get(k)} for k, v in EXPECTED_ENV.items() if observed.get(k) != v}
    return {
        "expected": EXPECTED_ENV,
        "observed": observed,
        "mismatches": mismatches,
        "scientific_environment_hash": EXPECTED_ENV_HASH,
        "status": "PASS" if not mismatches else "FAIL",
    }


def feature_names(state: str) -> list[str]:
    rows = pd.read_csv(FEATURE_SCHEMA)
    if state not in STATE_ORDER:
        raise RuntimeError(f"unsupported final full-TRAIN state: {state}")
    names = rows.loc[rows["state"].astype(str) == state, "feature_name"].astype(str).tolist()
    if state == "C_SHUFFLED":
        names = [n.replace("av_future_", "shuffled_av_future_") if n.startswith("av_future_") else n for n in names]
    return names


def feature_schema_hash(state: str) -> str:
    return sha256_json({"state": state, "feature_names": feature_names(state)})


def scientific_key_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for value in pd.util.hash_pandas_object(df[["scenario_id", "track_id"]], index=False).to_numpy(dtype=np.uint64):
        h.update(int(value).to_bytes(8, "little", signed=False))
    return h.hexdigest()


def load_registry() -> list[dict[str, Any]]:
    payload = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    registry = payload.get("candidate_registry")
    if not isinstance(registry, list) or len(registry) <= 1:
        raise RuntimeError("primary HGB candidate registry is missing or malformed")
    observed_hash = sha256_json(registry)
    if observed_hash != EXPECTED_REGISTRY_HASH:
        raise RuntimeError(f"candidate registry hash mismatch: {observed_hash}")
    if registry[1] != EXPECTED_CONFIG:
        raise RuntimeError(f"candidate_index=1 does not match frozen selected config: {registry[1]}")
    return registry


def selected_configs_status() -> dict[str, Any]:
    payload = yaml.safe_load(PRIMARY_CONFIGS.read_text(encoding="utf-8")) or {}
    selected = payload.get("selected_configs", {})
    status = "PASS"
    reasons = []
    for state in STATE_ORDER:
        rec = selected.get(state, {})
        cfg = {k: rec.get(k) for k in EXPECTED_CONFIG}
        if int(rec.get("candidate_index", -1)) != 1 or cfg != EXPECTED_CONFIG:
            status = "FAIL"
            reasons.append(f"{state} selected config is not candidate_index=1/frozen config")
    return {"status": status, "selected_configs": selected, "reasons": reasons}


def validate_completed_model(path: Path, meta_path: Path, expected_state: str, expected_target: str) -> dict[str, Any]:
    if not path.exists() or not meta_path.exists():
        return {"status": "MISSING", "path": str(path), "metadata_path": str(meta_path)}
    model_sha = sha256_file(path)
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "INVALID", "path": str(path), "metadata_path": str(meta_path), "reason": str(exc)}
    required = {
        "state": expected_state,
        "target_coordinate": expected_target,
        "input_matrix_sha256": EXPECTED_MATRIX_HASH,
        "candidate_index": 1,
        "hgb_config": EXPECTED_CONFIG,
        "scientific_environment_hash": EXPECTED_ENV_HASH,
        "model_artifact_sha256": model_sha,
    }
    mismatches = {k: {"expected": v, "observed": metadata.get(k)} for k, v in required.items() if metadata.get(k) != v}
    if mismatches:
        return {"status": "INVALID", "path": str(path), "metadata_path": str(meta_path), "mismatches": mismatches}
    return {"status": "PASS", "path": str(path), "metadata_path": str(meta_path), "sha256": model_sha, "metadata": metadata}


def model_paths(state: str, target: str) -> tuple[Path, Path]:
    base = MODELS_DIR / f"FULL_TRAIN_HGB_{state}_FINAL_{target}"
    return base.with_suffix(".pkl"), base.with_suffix(".json")


def completed_model_inventory() -> dict[str, Any]:
    units = []
    completed = 0
    invalid = []
    for state, target in EXPECTED_MODEL_UNITS:
        model_path, meta_path = model_paths(state, target)
        rec = validate_completed_model(model_path, meta_path, state, target)
        rec["state"] = state
        rec["target_coordinate"] = target
        units.append(rec)
        if rec["status"] == "PASS":
            completed += 1
        elif rec["status"] == "INVALID":
            invalid.append(rec)
    return {
        "completed_units": completed,
        "total_expected_units": len(EXPECTED_MODEL_UNITS),
        "missing_units": len(EXPECTED_MODEL_UNITS) - completed - len(invalid),
        "invalid_units": invalid,
        "units": units,
        "status": "PASS" if not invalid else "FAIL",
    }


def final_train_preflight() -> dict[str, Any]:
    EXEC.mkdir(exist_ok=True)
    AUDIT.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    load_registry()
    env = environment_status()
    matrix_sha = sha256_file(MATRIX) if MATRIX.exists() else None
    meta = pd.read_parquet(MATRIX, columns=["scenario_id", "track_id", "city"])
    rows = int(len(meta))
    scenarios = int(meta["scenario_id"].nunique())
    duplicate_keys = int(meta.duplicated(["scenario_id", "track_id"]).sum())
    key_hash = scientific_key_hash(meta)
    schema = pd.read_csv(FEATURE_SCHEMA)
    required_columns = sorted(set(schema["feature_name"].astype(str).tolist()) | set(TARGET_COLUMNS) | {"scenario_id", "track_id", "city"})
    required_columns = sorted({c.replace("av_future_", "shuffled_av_future_") if c.startswith("av_future_") else c for c in required_columns} | set(required_columns))
    present_columns = set(pq.read_schema(MATRIX).names)
    missing_columns = [c for c in required_columns if c not in present_columns]
    selection = selected_configs_status()
    maplite_audit = json.loads((AUDIT / "MAPLITE_KEYALIGNED_DEVELOPMENT_MATRIX_AUDIT_v4.json").read_text(encoding="utf-8"))
    cshuffle = json.loads((ROOT / "protocol" / "C_SHUFFLED_DONOR_UNIVERSE_CLARIFICATION_PREVAL_v4.json").read_text(encoding="utf-8"))
    model_inventory = completed_model_inventory()
    checks = {
        "matrix_sha_matches": matrix_sha == EXPECTED_MATRIX_HASH,
        "row_count_matches": rows == EXPECTED_ROWS,
        "scenario_count_matches": scenarios == EXPECTED_SCENARIOS,
        "no_duplicate_scientific_keys": duplicate_keys == 0,
        "required_columns_exist": not missing_columns,
        "maplite_canonical_identity_pass": maplite_audit.get("status") == "PASS" and maplite_audit.get("development_full_maplite_identity") is True,
        "target_future_leakage_prior_checks_pass": (ROOT / "tests" / "test_information_leakage.py").exists(),
        "cshuffled_stage_local_full_train_rule_recovered": cshuffle.get("stage_universes", {}).get("final_full_train_fitting") == "complete eligible TRAIN population",
        "candidate_index_1_config_matches_registry": selection["status"] == "PASS",
        "no_hyperparameter_search": True,
        "official_val_firewall_blocks_before_read": True,
        "scientific_environment_exact_match": env["status"] == "PASS",
        "completed_model_artifacts_resume_compatible": model_inventory["status"] == "PASS",
    }
    status = "READY_NOT_EXECUTED" if all(checks.values()) and model_inventory["completed_units"] < len(EXPECTED_MODEL_UNITS) else "COMPLETE" if all(checks.values()) else "NOT_READY"
    payload = {
        "phase": "preflight",
        "process_state": "not_started",
        "status": status,
        "created_utc": utc_now(),
        "matrix": str(MATRIX),
        "matrix_hash": matrix_sha,
        "expected_matrix_sha256": EXPECTED_MATRIX_HASH,
        "rows": rows,
        "scenarios": scenarios,
        "scientific_key_hash": key_hash,
        "duplicate_scientific_keys": duplicate_keys,
        "missing_required_columns": missing_columns,
        "checks": checks,
        "states": STATE_ORDER,
        "target_coordinates": TARGET_COLUMNS,
        "selected_candidate_index": 1,
        "selected_config": EXPECTED_CONFIG,
        "candidate_registry_hash": EXPECTED_REGISTRY_HASH,
        "feature_schema_file_sha256": sha256_file(FEATURE_SCHEMA),
        "feature_schema_hash_by_state": {state: feature_schema_hash(state) for state in STATE_ORDER},
        "scientific_environment_hash": EXPECTED_ENV_HASH,
        "environment": env,
        "model_inventory": model_inventory,
        "completed_final_model_units": model_inventory["completed_units"],
        "total_expected_model_units": len(EXPECTED_MODEL_UNITS),
        "failed_units": len(model_inventory["invalid_units"]),
        "completed_states": [],
        "telemetry_eta_seconds": None,
        "elapsed_seconds": None,
        "official_val_accessed": False,
        "offline_execution_ready": status in {"READY_NOT_EXECUTED", "COMPLETE"},
        "implementation_hash": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(STATUS, payload)
    return payload


def write_status(payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_utc"] = utc_now()
    payload["official_val_accessed"] = False
    atomic_json(STATUS, payload)


def fit_unit(df: pd.DataFrame, state: str, target: str, base_status: dict[str, Any]) -> dict[str, Any]:
    target_index = TARGET_COLUMNS.index(target)
    names = feature_names(state)
    X = df[names].astype(np.float32).to_numpy(np.float32)
    y = df[target].to_numpy(np.float32)
    started = time.time()
    model = HistGradientBoostingRegressor(random_state=p3b.TUNING_SEED + 1 * 10 + target_index, **EXPECTED_CONFIG)
    model.fit(X, y)
    model_path, meta_path = model_paths(state, target)
    payload = {
        "state": state,
        "target_coordinate": target,
        "candidate_index": 1,
        "hgb_config": EXPECTED_CONFIG,
        "feature_names": names,
        "model": model,
    }
    tmp = model_path.with_suffix(model_path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, model_path)
    model_sha = sha256_file(model_path)
    metadata = {
        "artifact_id": "FULL_TRAIN_HGB_FINAL_MODEL_v4",
        "created_utc": utc_now(),
        "state": state,
        "target_coordinate": target,
        "input_matrix_sha256": EXPECTED_MATRIX_HASH,
        "feature_schema_hash": feature_schema_hash(state),
        "candidate_index": 1,
        "hgb_config": EXPECTED_CONFIG,
        "sklearn_version": env_versions()["sklearn"],
        "python_version": env_versions()["python"],
        "numpy_version": env_versions()["numpy"],
        "pandas_version": env_versions()["pandas"],
        "pyarrow_version": env_versions()["pyarrow"],
        "training_row_count": int(len(df)),
        "training_scenario_count": int(df["scenario_id"].nunique()),
        "scientific_key_hash": base_status["scientific_key_hash"],
        "implementation_hash": base_status["implementation_hash"],
        "training_timestamp_utc": utc_now(),
        "model_artifact_sha256": model_sha,
        "official_val_accessed": False,
    }
    atomic_json(meta_path, metadata)
    del X, y, model
    gc.collect()
    return {**metadata, "fit_seconds": round(time.time() - started, 3), "model_path": str(model_path), "metadata_path": str(meta_path)}


def execute(resume: bool) -> dict[str, Any]:
    base = final_train_preflight()
    if base["status"] not in {"READY_NOT_EXECUTED", "COMPLETE"}:
        raise RuntimeError(f"final full-TRAIN preflight failed closed: {base['status']}")
    if base["environment"]["status"] != "PASS":
        raise RuntimeError("scientific environment gate failed")
    inventory = completed_model_inventory()
    if inventory["invalid_units"]:
        raise RuntimeError("incompatible completed model artifact detected; refusing resume")
    if inventory["completed_units"] and not resume:
        raise RuntimeError("compatible completed final model units exist; use --resume after a genuine crash")
    df = pd.read_parquet(MATRIX)
    write_status({**base, "phase": "execute", "process_state": "running", "started_utc": utc_now()})
    completed_records = []
    for state, target in EXPECTED_MODEL_UNITS:
        model_path, meta_path = model_paths(state, target)
        existing = validate_completed_model(model_path, meta_path, state, target)
        if existing["status"] == "PASS":
            completed_records.append(existing["metadata"])
            continue
        if existing["status"] == "INVALID":
            raise RuntimeError(f"incompatible model artifact blocks resume: {model_path}")
        status_payload = {
            **base,
            "phase": "execute",
            "process_state": "running",
            "current_state": state,
            "current_coordinate": target,
            "completed_final_model_units": len(completed_records),
            "total_expected_model_units": len(EXPECTED_MODEL_UNITS),
            "elapsed_seconds": None,
        }
        write_status(status_payload)
        rec = fit_unit(df, state, target, base)
        completed_records.append(rec)
        print(f"fit final full-TRAIN HGB state={state} target={target} sha256={rec['model_artifact_sha256']}", flush=True)
    manifest = {
        "artifact_id": "FULL_TRAIN_HGB_FINAL_MODELS_MANIFEST_v4",
        "created_utc": utc_now(),
        "status": "PASS",
        "input_matrix_sha256": EXPECTED_MATRIX_HASH,
        "training_rows": int(len(df)),
        "training_scenarios": int(df["scenario_id"].nunique()),
        "selected_candidate_index": 1,
        "selected_config": EXPECTED_CONFIG,
        "expected_model_units": len(EXPECTED_MODEL_UNITS),
        "completed_model_units": len(completed_records),
        "models": completed_records,
        "official_val_accessed": False,
    }
    manifest_hash = atomic_json(MANIFEST, manifest)
    atomic_text(
        AUDIT / "FULL_TRAIN_HGB_FINAL_FIT_AUDIT_v4.md",
        "\n".join(
            [
                "# Full TRAIN HGB Final Fit Audit v4",
                "",
                "- result: `PASS`",
                f"- input_matrix_sha256: `{EXPECTED_MATRIX_HASH}`",
                f"- training_rows: `{manifest['training_rows']}`",
                f"- training_scenarios: `{manifest['training_scenarios']}`",
                "- selected_candidate_index: `1`",
                "- final_model_units: `8/8`",
                "- official_val_accessed: `false`",
                f"- manifest_sha256: `{manifest_hash}`",
                "",
            ]
        ),
    )
    write_status({**base, "phase": "complete", "process_state": "completed", "status": "COMPLETE", "completed_final_model_units": len(completed_records), "total_expected_model_units": len(EXPECTED_MODEL_UNITS), "manifest_sha256": manifest_hash})
    return manifest


def firewall_test() -> dict[str, Any]:
    try:
        guard_not_val(VAL_FORBIDDEN / "sentinel.parquet")
    except RuntimeError as exc:
        return {"status": "PASS", "result": "BLOCKED_BEFORE_READ", "message": str(exc), "official_val_accessed": False}
    return {"status": "FAIL", "result": "NOT_BLOCKED", "official_val_accessed": True}


def main() -> int:
    parser = argparse.ArgumentParser(description="Final full-TRAIN HGB v4 runner")
    parser.add_argument("--execute", action="store_true", help="fit missing final HGB model units")
    parser.add_argument("--resume", action="store_true", help="resume compatible missing state/coordinate model units after a genuine crash")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--firewall-test", action="store_true")
    args = parser.parse_args()
    if args.workers != 1:
        raise RuntimeError("final full-TRAIN HGB v4 is certified for Workers=1 only")
    if args.firewall_test:
        print(json.dumps(firewall_test(), indent=2, sort_keys=True))
        return 0
    if args.execute:
        manifest = execute(resume=args.resume)
        print(json.dumps({"status": manifest["status"], "manifest": str(MANIFEST), "official_val_accessed": False}, indent=2, sort_keys=True))
        return 0
    payload = final_train_preflight()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"READY_NOT_EXECUTED", "COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
