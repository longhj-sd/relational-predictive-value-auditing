from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from maplite_features import MAP_LITE_FEATURES, compute_maplite_features


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
EXEC = ROOT / "execution"
PROCESSED = ROOT / "data" / "processed" / "phase3b"
TRAIN_ROOT = ROOT / "data" / "raw" / "av2_motion_forecasting" / "official_s3" / "train"
OFFICIAL_VAL_ROOTS = [
    ROOT / "data" / "raw" / "av2_motion_forecasting" / "official_s3" / "val",
    ROOT / "data" / "raw" / "av2_motion_forecasting" / "official_s3" / "validation",
]

STAGEA_V3_MATRIX = PROCESSED / "TRAIN_50K_FEATURES_MAPLITE_CORRECTED_v2.parquet"
FULL_TRAIN_MATRIX = PROCESSED / "FULL_TRAIN_MAPLITE_CORRECTED_MATRIX.parquet"
DEV_FOLDS = EXEC / "TRAIN_SCENARIO_FOLDS_v2.csv"
IDENTITY_QA = AUDIT / "FULL_TRAIN_DEVELOPMENT_SUBSET_IDENTITY_QA.json"
STATUS_DEFAULT = EXEC / "FORENSIC_MAPLITE_V4_STATUS.json"
CHECKPOINT_DEFAULT = EXEC / "forensic_maplite_v4_checkpoints"
MANIFEST_PATH = EXEC / "FORENSIC_MAPLITE_V4_MANIFEST.json"
DECISION_JSON = EXEC / "MAPLITE_FORENSIC_DECISION_v4.json"

KEY_COLS = ["scenario_id", "track_id"]
ORDER_COL = "__stagea_row_position"
ANGULAR_COLS = {
    "map_nearest_lane_heading",
    "map_target_to_lane_heading",
    "map_target_av_lane_heading_diff",
}
DECISIONS = {
    "V3_MAPLITE_MISALIGNMENT_CONFIRMED",
    "V3_MAPLITE_NOT_CONFIRMED",
    "FORENSIC_AMBIGUOUS_STOP",
}
CHECKPOINT_PROVENANCE_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    guard_not_val(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame(df: pd.DataFrame, cols: list[str]) -> str:
    h = hashlib.sha256()
    for value in pd.util.hash_pandas_object(df[cols], index=False).to_numpy(dtype=np.uint64):
        h.update(int(value).to_bytes(8, "little", signed=False))
    return h.hexdigest()


def sha256_json(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def guard_not_val(path: Path) -> None:
    resolved = path.resolve()
    lower_parts = [p.lower() for p in resolved.parts]
    if "official_s3" in lower_parts and any(p in {"val", "validation"} for p in lower_parts):
        logging.critical("official VAL path access blocked before read: %s", resolved)
        raise RuntimeError(f"official AV2 VAL path access blocked before read: {resolved}")
    for root in OFFICIAL_VAL_ROOTS:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        logging.critical("official VAL path access blocked before read: %s", resolved)
        raise RuntimeError(f"official AV2 VAL path access blocked before read: {resolved}")


def guarded_read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    guard_not_val(path)
    return pd.read_parquet(path, columns=columns)


def guarded_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    guard_not_val(path)
    return pd.read_csv(path, **kwargs)


def guarded_pq_read(path: Path, columns: list[str]) -> pd.DataFrame:
    guard_not_val(path)
    return pq.read_table(path, columns=columns).to_pandas()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with tmp.open("r+b") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def checkpoint_namespace(root: Path, mode: str) -> Path:
    if mode in {"sample", "full"} and root.name.lower() != mode:
        return root / mode
    return root


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    with tmp.open("r+b") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    df.to_csv(tmp, index=False)
    with tmp.open("r+b") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp)
    with tmp.open("r+b") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def checkpoint_provenance_path(path: Path) -> Path:
    return path.with_suffix(".provenance.json")


def configure_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def map_missing_cols() -> list[str]:
    return [f"{c}_missing" for c in MAP_LITE_FEATURES]


def map_all_cols() -> list[str]:
    return MAP_LITE_FEATURES + map_missing_cols()


def shuffled_cols(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith("shuffled_av_future_")]


def parquet_columns(path: Path) -> list[str]:
    guard_not_val(path)
    return pq.read_schema(path).names


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


def scenario_paths(sid: str) -> tuple[Path, Path]:
    folder = TRAIN_ROOT / sid
    return folder / f"scenario_{sid}.parquet", folder / f"log_map_archive_{sid}.json"


def load_stagea_and_full(map_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    dev_cols = parquet_columns(STAGEA_V3_MATRIX)
    full_cols = parquet_columns(FULL_TRAIN_MATRIX)
    common = [c for c in dev_cols if c in full_cols]
    needed = KEY_COLS + ["city", "fold", "is_nearest"] + map_all_cols() + shuffled_cols(common)
    if not map_only:
        needed = common
    needed = [c for c in dict.fromkeys(needed) if c in common]
    dev = guarded_read_parquet(STAGEA_V3_MATRIX, columns=needed).copy()
    dev[ORDER_COL] = np.arange(len(dev), dtype=np.int64)
    assert_unique_keys(dev, KEY_COLS, "Stage-A v3 development matrix")

    key_frame = dev[KEY_COLS].copy()
    full = guarded_read_parquet(FULL_TRAIN_MATRIX, columns=needed).copy()
    assert_unique_keys(full, KEY_COLS, "full TRAIN matrix")
    key_set = set(map(tuple, key_frame.astype(str).to_numpy()))
    mask = [key in key_set for key in map(tuple, full[KEY_COLS].astype(str).to_numpy())]
    full_dev = full.loc[mask].copy()
    aligned = keyed_merge(key_frame, full_dev, KEY_COLS, "full TRAIN development subset")
    full_dev = aligned
    if len(full_dev) != len(dev):
        raise RuntimeError(f"development/full cardinality mismatch: {len(dev)} vs {len(full_dev)}")
    return dev, full_dev, common


def raw_compute_scenario(item: tuple[str, list[str]]) -> list[dict[str, Any]]:
    sid, tracks = item
    scen_path, map_path = scenario_paths(sid)
    guard_not_val(scen_path)
    guard_not_val(map_path)
    df = guarded_pq_read(
        scen_path,
        columns=["track_id", "timestep", "position_x", "position_y", "heading"],
    )
    t49 = df[df["timestep"] == 49].copy()
    t49["track_id_str"] = t49["track_id"].astype(str)
    av_rows = t49[t49["track_id_str"] == "AV"]
    if len(av_rows) != 1:
        raise RuntimeError(f"{sid}: expected exactly one AV row at timestep 49, found {len(av_rows)}")
    av = av_rows.iloc[0]
    by_track = {str(r.track_id_str): r for r in t49.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for track in tracks:
        if track not in by_track:
            raise RuntimeError(f"{sid}/{track}: target track missing at timestep 49")
        t = by_track[track]
        values = compute_maplite_features(
            map_path,
            target_xy_49=(float(t.position_x), float(t.position_y)),
            target_heading_49=float(t.heading),
            av_xy_49=(float(av.position_x), float(av.position_y)),
            av_heading_49=float(av.heading),
        )
        rows.append({"scenario_id": sid, "track_id": str(track), **values})
    return rows


def build_units(keys: pd.DataFrame, chunk_size: int) -> list[dict[str, Any]]:
    scen = (
        keys[KEY_COLS]
        .astype(str)
        .groupby("scenario_id", sort=True)["track_id"]
        .apply(lambda s: sorted(s.tolist()))
        .reset_index()
    )
    units = []
    for i in range(0, len(scen), chunk_size):
        part = scen.iloc[i : i + chunk_size]
        unit_id = f"scenario_chunk_{i // chunk_size:05d}"
        expected_rows = int(part["track_id"].map(len).sum())
        units.append(
            {
                "unit_id": unit_id,
                "scenario_ids": part["scenario_id"].astype(str).tolist(),
                "tracks_by_scenario": {str(r.scenario_id): list(r.track_id) for r in part.itertuples(index=False)},
                "expected_rows": expected_rows,
            }
        )
    return units


def development_key_set_hash(keys: pd.DataFrame) -> str:
    key_set = keys[KEY_COLS].astype(str).drop_duplicates().sort_values(KEY_COLS).reset_index(drop=True)
    return sha256_frame(key_set, KEY_COLS)


def expected_checkpoint_schema_columns() -> list[str]:
    return KEY_COLS + map_all_cols()


def checkpoint_schema_hash() -> str:
    return sha256_json(expected_checkpoint_schema_columns())


def chunk_key_list_hash(unit: dict[str, Any]) -> str:
    payload = [
        {
            "scenario_id": str(sid),
            "track_ids": [str(t) for t in unit["tracks_by_scenario"][sid]],
        }
        for sid in unit["scenario_ids"]
    ]
    return sha256_json(payload)


def checkpoint_provenance(
    args: argparse.Namespace,
    input_hashes: dict[str, str],
    key_hash: str,
    unit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_provenance_version": CHECKPOINT_PROVENANCE_VERSION,
        "mode": args.mode,
        "input_stagea_matrix_sha256": input_hashes["stagea_matrix_sha256"],
        "full_train_matrix_sha256": input_hashes["full_train_matrix_sha256"],
        "canonical_feature_implementation_hash": input_hashes["canonical_feature_implementation_hash"],
        "development_key_set_hash": key_hash,
        "chunk_key_list_hash": chunk_key_list_hash(unit),
        "scenario_chunk_size": int(args.scenario_chunk_size),
        "expected_chunk_row_count": int(unit["expected_rows"]),
        "schema_hash": checkpoint_schema_hash(),
        "schema_columns": expected_checkpoint_schema_columns(),
        "unit_id": str(unit["unit_id"]),
    }


def checkpoint_provenance_diff(found: dict[str, Any], expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    diff = {}
    for key, value in expected.items():
        if found.get(key) != value:
            diff[key] = {"found": found.get(key), "expected": value}
    return diff


def completed_checkpoint(path: Path, expected_provenance: dict[str, Any], fail_closed: bool = True) -> str | None:
    if not path.exists() or path.suffix != ".parquet":
        return None
    try:
        guard_not_val(path)
        rows = pq.read_metadata(path).num_rows
    except Exception as exc:
        if fail_closed:
            raise RuntimeError(f"checkpoint validation failed for {path}: {exc}") from exc
        logging.warning("Ignoring invalid checkpoint %s: %s", path, exc)
        return None
    expected_rows = int(expected_provenance["expected_chunk_row_count"])
    if rows != expected_rows:
        msg = f"checkpoint {path} row count {rows} != expected {expected_rows}"
        if fail_closed:
            raise RuntimeError(msg)
        logging.warning("Ignoring %s", msg)
        return None
    actual_schema_hash = sha256_json(pq.read_schema(path).names)
    if actual_schema_hash != expected_provenance["schema_hash"]:
        msg = f"checkpoint {path} schema hash {actual_schema_hash} != expected {expected_provenance['schema_hash']}"
        if fail_closed:
            raise RuntimeError(msg)
        logging.warning("Ignoring %s", msg)
        return None
    sidecar = checkpoint_provenance_path(path)
    if not sidecar.exists():
        msg = f"checkpoint {path} missing provenance sidecar {sidecar}"
        if fail_closed:
            raise RuntimeError(msg)
        logging.warning("Ignoring %s", msg)
        return None
    try:
        found = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        if fail_closed:
            raise RuntimeError(f"checkpoint provenance unreadable for {path}: {exc}") from exc
        logging.warning("Ignoring unreadable checkpoint provenance %s: %s", sidecar, exc)
        return None
    diff = checkpoint_provenance_diff(found, expected_provenance)
    if diff:
        msg = f"checkpoint provenance mismatch for {path}: {json.dumps(diff, sort_keys=True)}"
        if fail_closed:
            raise RuntimeError(msg)
        logging.warning("Ignoring %s", msg)
        return None
    return sha256_file(path)


def write_checkpoint_with_provenance(df: pd.DataFrame, path: Path, provenance: dict[str, Any]) -> str:
    digest = write_parquet_atomic(df, path)
    payload = dict(provenance)
    payload["checkpoint_sha256"] = digest
    payload["created_utc"] = utc_now()
    write_json_atomic(checkpoint_provenance_path(path), payload)
    return digest


def update_status(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["last_updated_at"] = utc_now()
    payload["official_val_accessed"] = False
    write_json_atomic(path, payload)


def input_hashes_from_artifacts(artifacts: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {
        "stagea_matrix_sha256": str(artifacts[str(STAGEA_V3_MATRIX)]["sha256"]),
        "full_train_matrix_sha256": str(artifacts[str(FULL_TRAIN_MATRIX)]["sha256"]),
        "canonical_feature_implementation_hash": sha256_file(ROOT / "tools" / "maplite_features.py"),
    }


def migrate_legacy_sample_checkpoints(
    args: argparse.Namespace,
    input_hashes: dict[str, str],
    dev: pd.DataFrame,
    full: pd.DataFrame,
) -> dict[str, Any] | None:
    legacy_root = args.checkpoint_root
    if legacy_root.name.lower() in {"sample", "full"}:
        return None
    legacy_files = sorted(legacy_root.glob("scenario_chunk_*.parquet"))
    if not legacy_files:
        return None
    sample_namespace = checkpoint_namespace(legacy_root, "sample")
    sample_namespace.mkdir(parents=True, exist_ok=True)
    sample_args = argparse.Namespace(**vars(args))
    sample_args.mode = "sample"
    sample_args.checkpoint_dir = sample_namespace
    sample_keys = select_sample_keys(dev, full, args.sample_per_city)
    sample_units = build_units(sample_keys, args.scenario_chunk_size)
    key_hash = development_key_set_hash(sample_keys)
    migrated = []
    skipped = []
    unit_by_id = {u["unit_id"]: u for u in sample_units}
    for src in legacy_files:
        unit = unit_by_id.get(src.stem)
        if unit is None:
            skipped.append({"source": str(src), "reason": "no matching sample unit"})
            continue
        expected = checkpoint_provenance(sample_args, input_hashes, key_hash, unit)
        try:
            rows = pq.read_metadata(src).num_rows
        except Exception as exc:
            skipped.append({"source": str(src), "reason": f"unreadable parquet: {exc}"})
            continue
        if rows != int(unit["expected_rows"]):
            skipped.append({"source": str(src), "reason": f"row count {rows} != expected {unit['expected_rows']}"})
            continue
        dst = sample_namespace / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        digest = sha256_file(dst)
        sidecar_payload = dict(expected)
        sidecar_payload.update(
            {
                "checkpoint_sha256": digest,
                "created_utc": utc_now(),
                "legacy_migration_source": str(src),
                "legacy_migration_source_sha256": sha256_file(src),
            }
        )
        write_json_atomic(checkpoint_provenance_path(dst), sidecar_payload)
        migrated.append({"source": str(src), "destination": str(dst), "sha256": digest})
    manifest = {
        "created_utc": utc_now(),
        "migration_version": 1,
        "reason": "preserve legacy un-namespaced SAMPLE checkpoints before FULL launch",
        "source_checkpoint_dir": str(legacy_root),
        "sample_checkpoint_namespace": str(sample_namespace),
        "migrated": migrated,
        "skipped": skipped,
        "official_val_accessed": False,
    }
    manifest_path = sample_namespace / f"MIGRATION_MANIFEST_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    write_json_atomic(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def accepted_checkpoint_count(
    units: list[dict[str, Any]],
    args: argparse.Namespace,
    input_hashes: dict[str, str],
    key_hash: str,
    fail_closed: bool,
) -> int:
    accepted = 0
    for unit in units:
        final = args.checkpoint_dir / f"{unit['unit_id']}.parquet"
        expected = checkpoint_provenance(args, input_hashes, key_hash, unit)
        if completed_checkpoint(final, expected, fail_closed=fail_closed):
            accepted += 1
    return accepted


def recompute_raw(
    keys: pd.DataFrame,
    args: argparse.Namespace,
    started: float,
    input_hashes: dict[str, str],
) -> pd.DataFrame:
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    units = build_units(keys, args.scenario_chunk_size)
    key_hash = development_key_set_hash(keys)
    status = {
        "phase": "raw_recomputation",
        "mode": args.mode,
        "checkpoint_namespace": str(args.checkpoint_dir),
        "development_key_hash": key_hash,
        "input_hashes": input_hashes,
        "started_at": utc_now(),
        "total_units": len(units),
        "completed_units": 0,
        "failed_units": 0,
        "current_unit": None,
        "processed_rows": 0,
        "expected_rows": int(len(keys)),
        "percent_complete": 0.0,
        "elapsed_seconds": 0.0,
        "estimated_remaining_seconds": None,
        "last_checkpoint": None,
        "last_checkpoint_sha256": None,
        "process_state": "running",
        "final_verdict": None,
    }
    valid: dict[str, tuple[Path, str, int]] = {}
    for unit in units:
        final = args.checkpoint_dir / f"{unit['unit_id']}.parquet"
        provenance = checkpoint_provenance(args, input_hashes, key_hash, unit)
        digest = completed_checkpoint(final, provenance, fail_closed=True) if args.resume else None
        if digest:
            valid[unit["unit_id"]] = (final, digest, int(unit["expected_rows"]))
    status["completed_units"] = len(valid)
    status["processed_rows"] = sum(v[2] for v in valid.values())
    status["percent_complete"] = 100.0 * status["processed_rows"] / max(1, status["expected_rows"])
    update_status(args.status_file, status)

    for unit in units:
        if unit["unit_id"] in valid:
            continue
        status["current_unit"] = unit["unit_id"]
        update_status(args.status_file, status)
        logging.info("Computing %s scenarios=%d rows=%d", unit["unit_id"], len(unit["scenario_ids"]), unit["expected_rows"])
        rows: list[dict[str, Any]] = []
        items = [(sid, unit["tracks_by_scenario"][sid]) for sid in unit["scenario_ids"]]
        try:
            if args.workers > 1 and len(items) > 1:
                with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
                    for scenario_rows in pool.map(raw_compute_scenario, items):
                        rows.extend(scenario_rows)
            else:
                for item in items:
                    rows.extend(raw_compute_scenario(item))
            df = pd.DataFrame(rows)
            assert_unique_keys(df, KEY_COLS, f"raw recomputation {unit['unit_id']}")
            if len(df) != int(unit["expected_rows"]):
                raise RuntimeError(f"{unit['unit_id']} row count mismatch: {len(df)} vs {unit['expected_rows']}")
            final = args.checkpoint_dir / f"{unit['unit_id']}.parquet"
            provenance = checkpoint_provenance(args, input_hashes, key_hash, unit)
            digest = write_checkpoint_with_provenance(df, final, provenance)
            valid[unit["unit_id"]] = (final, digest, len(df))
            status["completed_units"] = len(valid)
            status["processed_rows"] = sum(v[2] for v in valid.values())
            status["percent_complete"] = 100.0 * status["processed_rows"] / max(1, status["expected_rows"])
            status["elapsed_seconds"] = round(time.time() - started, 3)
            rate = status["processed_rows"] / max(1e-9, status["elapsed_seconds"])
            remaining = status["expected_rows"] - status["processed_rows"]
            status["estimated_remaining_seconds"] = round(remaining / rate, 1) if rate > 0 and remaining > 0 else 0
            status["last_checkpoint"] = str(final)
            status["last_checkpoint_sha256"] = digest
            update_status(args.status_file, status)
            logging.info("Completed %s sha256=%s", unit["unit_id"], digest)
        except Exception:
            status["failed_units"] = int(status["failed_units"]) + 1
            status["process_state"] = "failed"
            update_status(args.status_file, status)
            raise

    frames = [guarded_read_parquet(valid[u["unit_id"]][0]) for u in units]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEY_COLS + map_all_cols())
    assert_unique_keys(out, KEY_COLS, "assembled raw recomputation")
    status["process_state"] = "raw_recomputation_complete"
    status["current_unit"] = None
    status["completed_units"] = len(units)
    status["processed_rows"] = int(len(out))
    status["percent_complete"] = 100.0
    status["elapsed_seconds"] = round(time.time() - started, 3)
    status["estimated_remaining_seconds"] = 0
    update_status(args.status_file, status)
    return out


def absdiff(a: pd.Series, b: pd.Series) -> np.ndarray:
    return np.abs(a.astype(float).to_numpy() - b.astype(float).to_numpy())


def wrapped_diff(a: pd.Series, b: pd.Series) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(a.astype(float).to_numpy() - b.astype(float).to_numpy()), np.cos(a.astype(float).to_numpy() - b.astype(float).to_numpy())))


def classify_three_way(raw: float, full: float, v3: float, angular: bool, tol: float) -> str:
    rf = abs(raw - full) <= tol
    rv = abs(raw - v3) <= tol
    fv = abs(full - v3) <= tol
    if rf and rv and fv:
        return "RAW_FULL_V3_ALL_AGREE"
    if rf and not rv:
        return "RAW_EQUALS_FULL_NOT_V3"
    if rv and not rf:
        return "RAW_EQUALS_V3_NOT_FULL"
    if angular:
        wrf = abs(math.atan2(math.sin(raw - full), math.cos(raw - full))) <= tol
        wrv = abs(math.atan2(math.sin(raw - v3), math.cos(raw - v3))) <= tol
        if (wrf or wrv) and not (rf or rv):
            return "NUMERICAL_EQUIVALENCE_ONLY"
    return "NONE_AGREE"


def compare_maplite(dev: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(dev)
    for col in MAP_LITE_FEATURES:
        raw_d = absdiff(dev[col], full[col])
        mismatch = raw_d > 1e-6
        rec: dict[str, Any] = {
            "feature": col,
            "rows": n,
            "mismatched_rows": int(mismatch.sum()),
            "mismatched_percent": float(100.0 * mismatch.mean()),
            "max_abs_discrepancy": float(np.nanmax(raw_d)) if len(raw_d) else 0.0,
            "median_abs_discrepancy": float(np.nanmedian(raw_d[mismatch])) if mismatch.any() else 0.0,
            "mean_abs_discrepancy": float(np.nanmean(raw_d[mismatch])) if mismatch.any() else 0.0,
            "angular": col in ANGULAR_COLS,
        }
        if col in ANGULAR_COLS:
            wd = wrapped_diff(dev[col], full[col])
            rec["max_wrapped_abs_discrepancy"] = float(np.nanmax(wd)) if len(wd) else 0.0
            rec["median_wrapped_abs_discrepancy"] = float(np.nanmedian(wd[mismatch])) if mismatch.any() else 0.0
            rec["mean_wrapped_abs_discrepancy"] = float(np.nanmean(wd[mismatch])) if mismatch.any() else 0.0
        by_city = dev.loc[mismatch].groupby("city").size().to_dict() if "city" in dev else {}
        by_role = dev.loc[mismatch].assign(role=np.where(dev.loc[mismatch, "is_nearest"].astype(int) == 1, "N", "O")).groupby("role").size().to_dict()
        pos = dev.loc[mismatch, ORDER_COL]
        rec["city_distribution_json"] = json.dumps({str(k): int(v) for k, v in by_city.items()}, sort_keys=True)
        rec["role_distribution_json"] = json.dumps({str(k): int(v) for k, v in by_role.items()}, sort_keys=True)
        rec["row_position_min"] = int(pos.min()) if len(pos) else None
        rec["row_position_median"] = float(pos.median()) if len(pos) else None
        rec["row_position_max"] = int(pos.max()) if len(pos) else None
        rows.append(rec)
    return pd.DataFrame(rows)


def root_cause_alignment(dev: pd.DataFrame, full: pd.DataFrame) -> dict[str, Any]:
    keyed_full = full[KEY_COLS + MAP_LITE_FEATURES].copy()
    sorted_stagea_order = (
        pd.concat([g[KEY_COLS] for _, g in dev.groupby("scenario_id", sort=True)], ignore_index=True)
        .reset_index(drop=True)
    )
    generated = keyed_merge(sorted_stagea_order, keyed_full, KEY_COLS, "historical sorted generation simulation")
    predicted = dev[KEY_COLS + [ORDER_COL]].copy().reset_index(drop=True)
    for col in MAP_LITE_FEATURES:
        predicted[col] = generated[col].to_numpy()
    exact_cols = []
    mismatched_exact_cols = []
    rates = {}
    for col in MAP_LITE_FEATURES:
        same = np.isclose(dev[col].astype(float), predicted[col].astype(float), rtol=0, atol=1e-9, equal_nan=True)
        exact_cols.append(bool(same.all()))
        v3_full_mismatch = absdiff(dev[col], full[col]) > 1e-6
        rates[col] = float(same[v3_full_mismatch].mean()) if bool(v3_full_mismatch.any()) else 1.0
        mismatched_exact_cols.append(bool(same[v3_full_mismatch].all()) if bool(v3_full_mismatch.any()) else True)
    same_order = bool(dev[KEY_COLS].astype(str).reset_index(drop=True).equals(sorted_stagea_order[KEY_COLS].astype(str)))
    return {
        "stagea_key_order_already_sorted_by_scenario_track": same_order,
        "historical_generated_order_hash": sha256_frame(sorted_stagea_order, KEY_COLS),
        "stagea_original_order_hash": sha256_frame(dev[KEY_COLS], KEY_COLS),
        "positional_assignment_all_maplite_columns_match_v3": bool(all(exact_cols)),
        "positional_assignment_all_mismatched_cells_match_v3": bool(all(mismatched_exact_cols)),
        "per_feature_mismatched_cell_positional_match_rate": rates,
        "demonstrated_synthetic_update_misalignment": synthetic_update_misalignment_demo()["misassignment_demonstrated"],
    }


def synthetic_update_misalignment_demo() -> dict[str, Any]:
    target = pd.DataFrame(
        {"scenario_id": ["b", "a"], "track_id": ["2", "1"], "map_nearest_lane_heading": [0.0, 0.0]}
    )
    generated = pd.DataFrame(
        {"scenario_id": ["a", "b"], "track_id": ["1", "2"], "map_nearest_lane_heading": [10.0, 20.0]}
    )
    positional = target.copy()
    positional.update(generated[["map_nearest_lane_heading"]])
    keyed = target[["scenario_id", "track_id"]].merge(generated, on=KEY_COLS, how="left", validate="one_to_one")
    return {
        "misassignment_demonstrated": bool(positional["map_nearest_lane_heading"].tolist() == [10.0, 20.0] and keyed["map_nearest_lane_heading"].tolist() == [20.0, 10.0]),
        "positional_values": positional["map_nearest_lane_heading"].tolist(),
        "keyed_values": keyed["map_nearest_lane_heading"].tolist(),
    }


def select_sample_keys(dev: pd.DataFrame, full: pd.DataFrame, per_city: int) -> pd.DataFrame:
    base = dev[KEY_COLS + ["city", "is_nearest", ORDER_COL]].copy()
    pieces = []
    for _, g in base.groupby("city", sort=True):
        pieces.append(g.head(min(10, len(g))))
        pieces.append(g.iloc[max(0, len(g) // 2 - 5) : min(len(g), len(g) // 2 + 5)])
        pieces.append(g.tail(min(10, len(g))))
        pieces.append(g[g["is_nearest"].astype(int) == 1].head(min(20, int((g["is_nearest"].astype(int) == 1).sum()))))
        pieces.append(g[g["is_nearest"].astype(int) == 0].head(min(20, int((g["is_nearest"].astype(int) == 0).sum()))))
    max_diff = np.zeros(len(dev), dtype=float)
    for col in MAP_LITE_FEATURES:
        max_diff = np.maximum(max_diff, absdiff(dev[col], full[col]))
    ranked = base.assign(max_maplite_abs_diff=max_diff).sort_values(["max_maplite_abs_diff", ORDER_COL], ascending=[False, True])
    pieces.append(ranked.head(500))
    known = base[(base["scenario_id"].astype(str) == "b309928d-856e-4c19-a2e9-fd1151d1d5fe") & (base["track_id"].astype(str) == "34959")]
    if not known.empty:
        pieces.append(known)
    mismatch_any = ranked[ranked["max_maplite_abs_diff"] > 1e-6]
    match_all = ranked[ranked["max_maplite_abs_diff"] <= 1e-6]
    for _, g in mismatch_any.groupby("city", sort=True):
        pieces.append(g.sample(n=min(per_city, len(g)), random_state=20260909))
    for _, g in match_all.groupby("city", sort=True):
        pieces.append(g.sample(n=min(per_city, len(g)), random_state=20260910))
    sample = pd.concat(pieces, ignore_index=True).drop_duplicates(KEY_COLS).sort_values(ORDER_COL).reset_index(drop=True)
    return sample


def three_way_audit(dev: pd.DataFrame, full: pd.DataFrame, raw: pd.DataFrame, mode: str) -> pd.DataFrame:
    cols = KEY_COLS + ["city", "is_nearest", ORDER_COL] + MAP_LITE_FEATURES
    d = keyed_merge(dev[cols], full[KEY_COLS + MAP_LITE_FEATURES].rename(columns={c: f"{c}__full" for c in MAP_LITE_FEATURES}), KEY_COLS, "three-way full")
    d = keyed_merge(d, raw[KEY_COLS + MAP_LITE_FEATURES].rename(columns={c: f"{c}__raw" for c in MAP_LITE_FEATURES}), KEY_COLS, "three-way raw")
    rows = []
    for _, r in d.iterrows():
        base = {"scenario_id": r["scenario_id"], "track_id": str(r["track_id"]), "city": r["city"], "is_nearest": int(r["is_nearest"]), "row_position": int(r[ORDER_COL])}
        for col in MAP_LITE_FEATURES:
            v3 = float(r[col])
            full_v = float(r[f"{col}__full"])
            raw_v = float(r[f"{col}__raw"])
            raw_abs = abs(raw_v - v3)
            full_abs = abs(full_v - v3)
            rows.append(
                {
                    **base,
                    "feature": col,
                    "mode": mode,
                    "v3_value": v3,
                    "full_value": full_v,
                    "raw_value": raw_v,
                    "raw_minus_v3": raw_v - v3,
                    "full_minus_v3": full_v - v3,
                    "abs_raw_minus_v3": raw_abs,
                    "abs_raw_minus_full": abs(raw_v - full_v),
                    "abs_full_minus_v3": full_abs,
                    "wrapped_abs_raw_minus_v3": abs(math.atan2(math.sin(raw_v - v3), math.cos(raw_v - v3))) if col in ANGULAR_COLS else np.nan,
                    "wrapped_abs_raw_minus_full": abs(math.atan2(math.sin(raw_v - full_v), math.cos(raw_v - full_v))) if col in ANGULAR_COLS else np.nan,
                    "classification": classify_three_way(raw_v, full_v, v3, col in ANGULAR_COLS, 1e-6),
                }
            )
    return pd.DataFrame(rows)


def write_markdown_outputs(comparison: pd.DataFrame, three_way: pd.DataFrame | None, root: dict[str, Any], decision: dict[str, Any]) -> None:
    write_csv_atomic(comparison, AUDIT / "MAPLITE_V3_VS_FULL_FORENSIC_COMPARISON_v4.csv")
    summary = [
        "# MAP-LITE v3 vs Full TRAIN Forensic Comparison v4",
        "",
        f"- created_utc: `{utc_now()}`",
        "- scope: TRAIN-only; official AV2 VAL not read",
        f"- stagea_matrix: `{STAGEA_V3_MATRIX}`",
        f"- full_train_matrix: `{FULL_TRAIN_MATRIX}`",
        "",
        comparison.to_markdown(index=False),
        "",
    ]
    write_text_atomic(AUDIT / "MAPLITE_V3_VS_FULL_FORENSIC_COMPARISON_v4.md", "\n".join(summary))

    if three_way is not None:
        write_csv_atomic(three_way, AUDIT / "MAPLITE_THREE_WAY_RAW_RECOMPUTATION_AUDIT_v4.csv")
        counts = three_way.groupby(["feature", "classification"]).size().reset_index(name="rows")
        lines = [
            "# MAP-LITE Three-Way Raw Recomputaton Audit v4",
            "",
            f"- created_utc: `{utc_now()}`",
            f"- mode: `{decision['mode']}`",
            "- scope: TRAIN-only; official AV2 VAL not read",
            "",
            counts.to_markdown(index=False),
            "",
        ]
        write_text_atomic(AUDIT / "MAPLITE_THREE_WAY_RAW_RECOMPUTATION_AUDIT_v4.md", "\n".join(lines))

    root_lines = [
        "# MAP-LITE Stage-A v3 Root-Cause Analysis v4",
        "",
        f"- created_utc: `{utc_now()}`",
        "- scope: TRAIN-only; official AV2 VAL not read",
        "",
        "```json",
        json.dumps(root, indent=2, sort_keys=True),
        "```",
        "",
    ]
    write_text_atomic(AUDIT / "MAPLITE_STAGEA_V3_ROOT_CAUSE_ANALYSIS_v4.md", "\n".join(root_lines))

    decision_lines = [
        "# MAP-LITE Forensic Decision v4",
        "",
        f"- final_decision: `{decision['final_decision']}`",
        f"- mode: `{decision['mode']}`",
        f"- official_val_accessed: `{str(decision['official_val_accessed']).lower()}`",
        f"- reason: {decision['reason']}",
        "",
    ]
    write_text_atomic(AUDIT / "MAPLITE_FORENSIC_DECISION_v4.md", "\n".join(decision_lines))
    write_json_atomic(DECISION_JSON, decision)


def decide(mode: str, three_way: pd.DataFrame | None, root: dict[str, Any]) -> dict[str, Any]:
    if mode != "full" or three_way is None:
        final = "FORENSIC_AMBIGUOUS_STOP"
        reason = "preflight/sample mode cannot establish a full-population final forensic decision"
    else:
        raw_full = float((three_way["classification"] == "RAW_EQUALS_FULL_NOT_V3").mean())
        none = int((three_way["classification"] == "NONE_AGREE").sum())
        if raw_full > 0.0 and none == 0 and root["positional_assignment_all_mismatched_cells_match_v3"] and root["demonstrated_synthetic_update_misalignment"]:
            final = "V3_MAPLITE_MISALIGNMENT_CONFIRMED"
            reason = "canonical raw recomputation agrees with full TRAIN and positional Stage-A simulation explains all mismatched MAP-LITE cells"
        else:
            final = "V3_MAPLITE_NOT_CONFIRMED"
            reason = "full-population evidence did not satisfy the strict raw/full agreement plus implementation-root-cause gate"
    assert final in DECISIONS
    return {"final_decision": final, "mode": mode, "reason": reason, "created_utc": utc_now(), "official_val_accessed": False}


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    AUDIT.mkdir(exist_ok=True)
    EXEC.mkdir(exist_ok=True)
    artifacts = {}
    for path in [STAGEA_V3_MATRIX, FULL_TRAIN_MATRIX, DEV_FOLDS, IDENTITY_QA]:
        guard_not_val(path)
        artifacts[str(path)] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
        }
    missing = [p for p, rec in artifacts.items() if not rec["exists"]]
    stage_cols = parquet_columns(STAGEA_V3_MATRIX) if STAGEA_V3_MATRIX.exists() else []
    full_cols = parquet_columns(FULL_TRAIN_MATRIX) if FULL_TRAIN_MATRIX.exists() else []
    input_hashes = input_hashes_from_artifacts(artifacts)
    total_units = 0
    completed_units = 0
    expected_rows = None
    development_hash = None
    migration = None
    target_mode = args.target_mode
    if target_mode in {"sample", "full"} and not missing:
        dev, full, _common = load_stagea_and_full(map_only=True)
        migration = migrate_legacy_sample_checkpoints(args, input_hashes, dev, full)
        intended_args = argparse.Namespace(**vars(args))
        intended_args.mode = target_mode
        if target_mode == "sample":
            keys = select_sample_keys(dev, full, args.sample_per_city)
        else:
            keys = dev[KEY_COLS + ["city", "is_nearest", ORDER_COL]].copy()
        units = build_units(keys, args.scenario_chunk_size)
        total_units = len(units)
        expected_rows = int(len(keys))
        development_hash = development_key_set_hash(keys)
        completed_units = accepted_checkpoint_count(
            units,
            intended_args,
            input_hashes,
            development_hash,
            fail_closed=bool(args.resume),
        )
    payload = {
        "phase": "preflight",
        "mode": target_mode,
        "runner_mode": args.mode,
        "preflight_only": args.mode == "preflight",
        "checkpoint_namespace": str(args.checkpoint_dir),
        "development_key_hash": development_hash,
        "input_hashes": input_hashes,
        "started_at": utc_now(),
        "last_updated_at": utc_now(),
        "total_units": total_units,
        "completed_units": completed_units,
        "failed_units": 0,
        "current_unit": None,
        "processed_rows": 0,
        "expected_rows": expected_rows,
        "percent_complete": 100.0,
        "elapsed_seconds": 0.0,
        "estimated_remaining_seconds": None,
        "official_val_accessed": False,
        "last_checkpoint": None,
        "last_checkpoint_sha256": None,
        "process_state": "preflight_pass" if not missing else "preflight_fail",
        "final_verdict": None,
        "artifacts": artifacts,
        "stagea_has_maplite_columns": all(c in stage_cols for c in map_all_cols()),
        "full_has_maplite_columns": all(c in full_cols for c in map_all_cols()),
        "checkpoint_migration": migration,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "sklearn": getattr(__import__("sklearn"), "__version__", "unavailable"),
        },
        "default_is_safe_preflight": True,
    }
    write_json_atomic(args.status_file, payload)
    manifest = {
        "created_utc": utc_now(),
        "mode": target_mode,
        "runner_mode": args.mode,
        "preflight_only": args.mode == "preflight",
        "official_val_accessed": False,
        "inputs": artifacts,
        "maplite_features": MAP_LITE_FEATURES,
        "angular_features": sorted(ANGULAR_COLS),
        "checkpoint_root": str(args.checkpoint_root),
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_namespace": str(args.checkpoint_dir),
        "status_file": str(args.status_file),
        "scenario_chunk_size": args.scenario_chunk_size,
        "workers": args.workers,
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    if missing:
        raise RuntimeError(f"preflight missing required artifacts: {missing}")
    return payload


def run_forensic(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    run_preflight(args)
    status_payload = json.loads(args.status_file.read_text(encoding="utf-8"))
    input_hashes = status_payload["input_hashes"]
    dev, full, _common = load_stagea_and_full(map_only=True)
    comparison = compare_maplite(dev, full)
    root = root_cause_alignment(dev, full)
    if args.mode == "sample":
        keys = select_sample_keys(dev, full, args.sample_per_city)
    elif args.mode == "full":
        keys = dev[KEY_COLS + ["city", "is_nearest", ORDER_COL]].copy()
    else:
        raise RuntimeError(f"unexpected mode for recomputation: {args.mode}")
    raw = recompute_raw(keys, args, started, input_hashes)
    three_way = three_way_audit(dev.merge(keys[KEY_COLS], on=KEY_COLS, how="inner", validate="one_to_one"), full.merge(keys[KEY_COLS], on=KEY_COLS, how="inner", validate="one_to_one"), raw, args.mode)
    decision = decide(args.mode, three_way, root)
    write_markdown_outputs(comparison, three_way, root, decision)
    status = json.loads(args.status_file.read_text(encoding="utf-8"))
    status["phase"] = "complete"
    status["process_state"] = "complete"
    status["final_verdict"] = decision["final_decision"]
    status["elapsed_seconds"] = round(time.time() - started, 3)
    update_status(args.status_file, status)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="TRAIN-only MAP-LITE v3/full identity forensic package")
    parser.add_argument("--mode", choices=["preflight", "sample", "full"], default="preflight")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--status-file", type=Path, default=STATUS_DEFAULT)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--scenario-chunk-size", type=int, default=250)
    parser.add_argument("--sample-per-city", type=int, default=100)
    parser.add_argument("--preflight-target-mode", choices=["sample", "full"], help="validate a sample/full launch without raw recomputation")
    args = parser.parse_args()
    args.target_mode = args.preflight_target_mode if args.mode == "preflight" and args.preflight_target_mode else args.mode
    args.checkpoint_root = args.checkpoint_dir
    args.checkpoint_dir = checkpoint_namespace(args.checkpoint_root, args.target_mode)
    configure_logging(args.log_file)
    logging.info("startup config=%s", {k: str(v) for k, v in vars(args).items()})
    logging.info("environment python=%s platform=%s", sys.version.split()[0], platform.platform())
    try:
        if args.mode == "preflight":
            payload = run_preflight(args)
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            decision = run_forensic(args)
            print(json.dumps(decision, indent=2, sort_keys=True))
    except Exception:
        logging.exception("forensic_maplite_identity_v4 failed")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
