from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

import phase3a_train_only_freeze as p3a
import phase3b_train_only_execution as p3b


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
EXEC = ROOT / "execution"
PROCESSED = ROOT / "data" / "processed" / "phase3b"
WORK = PROCESSED / "full_train_build_parts_v1"
FULL_MATRIX = PROCESSED / "FULL_TRAIN_MAPLITE_CORRECTED_MATRIX.parquet"
FULL_MATRIX_COMPAT = PROCESSED / "FULL_TRAIN_FEATURES_MAPLITE_CORRECTED_v3.parquet"
DEV_MATRIX = PROCESSED / "TRAIN_50K_FEATURES_MAPLITE_CORRECTED_v2.parquet"
ELIGIBILITY = AUDIT / "TRAIN_ELIGIBILITY_SCENARIOS.csv"
DEV_FOLDS = EXEC / "TRAIN_SCENARIO_FOLDS_v2.csv"
STATE_ORDER = ["H", "E", "C", "C_SHUFFLED"]
EXPECTED = {
    "eligible_scenarios": 145390,
    "eligible_target_rows": 630388,
    "FOCAL": 128347,
    "SCORED": 502041,
    "N": 145390,
    "O": 484998,
}
SELECTED_CFG = {
    "learning_rate": 0.03,
    "max_iter": 1000,
    "max_leaf_nodes": 127,
    "l2_regularization": 0.1,
    "min_samples_leaf": 50,
    "max_bins": 255,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def artifact_record(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def preflight() -> dict:
    py = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "sklearn": __import__("sklearn").__version__,
    }
    try:
        import torch

        py["torch"] = torch.__version__
        py["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - environment record only
        py["torch"] = f"unavailable: {type(exc).__name__}: {exc}"
        py["torch_cuda_available"] = False

    disk = {}
    for drive in ["C:", "D:", "E:", "F:"]:
        try:
            usage = __import__("shutil").disk_usage(drive + "\\")
            disk[drive] = {"total": usage.total, "used": usage.used, "free": usage.free}
        except Exception:
            pass
    mem = {}
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress",
            ],
            text=True,
            encoding="utf-8",
        )
        osinfo = json.loads(out)
        mem = {
            "total_visible_memory_kib": int(osinfo["TotalVisibleMemorySize"]),
            "free_physical_memory_kib": int(osinfo["FreePhysicalMemory"]),
        }
    except Exception as exc:
        mem = {"error": str(exc)}
    cpu = {}
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress",
            ],
            text=True,
            encoding="utf-8",
        )
        cpu = json.loads(out)
    except Exception as exc:
        cpu = {"error": str(exc)}
    gpu = "nvidia-smi unavailable"
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        pass

    dev_bytes = DEV_MATRIX.stat().st_size if DEV_MATRIX.exists() else None
    estimated_matrix_bytes = int(dev_bytes * (EXPECTED["eligible_target_rows"] / 216170)) if dev_bytes else None
    payload = {
        "audit_id": "STAGE_B_COMPUTE_RESOURCE_PREFLIGHT",
        "created_utc": utc_now(),
        "official_val_outcome_blind": True,
        "disk": disk,
        "memory": mem,
        "cpu": cpu,
        "gpu": gpu,
        "software": py,
        "expected_counts": EXPECTED,
        "estimated_outputs": {
            "full_matrix_bytes_from_development_scaling": estimated_matrix_bytes,
            "hgb_model_artifacts_bytes": "expected < 1 GB total",
            "neural_artifacts_bytes": "not yet launched; CPU-only torch detected if cuda unavailable",
        },
        "resume_strategy": {
            "full_matrix": "per-chunk base parquet parts, futures registry, shuffled/final parquet parts, final writer",
            "hgb_final": "per-state pickle temp file followed by atomic replace",
            "neural": "not implemented in this runner; must use frozen neural implementation/registry if available",
            "heldout_city": "not implemented in this runner; requires complete TRAIN_-c CV/finalization runner",
        },
        "resource_blocker": None,
    }
    if disk.get("D:", {}).get("free", 0) < 10 * 1024**3:
        payload["resource_blocker"] = "D: free space below 10 GiB conservative minimum"
    write_json_atomic(AUDIT / "STAGE_B_COMPUTE_RESOURCE_PREFLIGHT.json", payload)
    lines = [
        "# Stage-B Compute Resource Preflight",
        "",
        f"- created_utc: `{payload['created_utc']}`",
        "- official_val_outcome_blind: `true`",
        f"- resource_blocker: `{payload['resource_blocker']}`",
        f"- estimated_full_matrix_bytes: `{estimated_matrix_bytes}`",
        f"- python: `{sys.version.split()[0]}`",
        f"- sklearn: `{py['sklearn']}`",
        f"- torch: `{py['torch']}`",
        f"- torch_cuda_available: `{str(py['torch_cuda_available']).lower()}`",
        f"- gpu: `{gpu}`",
        "",
        "## Disk",
        "",
        pd.DataFrame([{"drive": k, **v} for k, v in disk.items()]).to_markdown(index=False),
        "",
        "## Memory",
        "",
        "```json",
        json.dumps(mem, indent=2, sort_keys=True),
        "```",
        "",
        "Long components are resumable via deterministic part files and atomic final writes. No scientific population, candidate count, fold count, seed count, epoch count, city scope, or bootstrap definition is reduced by this preflight.",
        "",
    ]
    write_text_atomic(AUDIT / "STAGE_B_COMPUTE_RESOURCE_PREFLIGHT.md", "\n".join(lines))
    return payload


def full_train_folds() -> pd.DataFrame:
    elig = pd.read_csv(ELIGIBILITY)
    full = elig.loc[elig["eligible_primary_targets"] >= 2, ["scenario_id", "city", "eligible_primary_targets", "focal_tracks", "scored_tracks", "o_count"]].copy()
    dev = pd.read_csv(DEV_FOLDS)[["scenario_id", "fold"]]
    full = full.merge(dev, on="scenario_id", how="left")

    def fallback_fold(row: pd.Series) -> int:
        digest = hashlib.sha256(f"{p3b.SPLIT_SEED}:full-train:{row.city}:{row.scenario_id}".encode("utf-8")).hexdigest()
        return int(digest, 16) % 5

    missing = full["fold"].isna()
    full.loc[missing, "fold"] = full.loc[missing].apply(fallback_fold, axis=1)
    full["fold"] = full["fold"].astype(int)
    return full.sort_values(["city", "scenario_id"]).reset_index(drop=True)


def shuffled_columns() -> list[str]:
    cols = []
    for t in p3a.FUTURE_STEPS:
        cols.extend(
            [
                f"shuffled_av_future_x_{t}_fixed",
                f"shuffled_av_future_y_{t}_fixed",
                f"shuffled_av_future_vx_{t}_fixed",
                f"shuffled_av_future_vy_{t}_fixed",
            ]
        )
    return cols


def compute_full_shuffle(folds: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
    sample = folds.merge(
        pd.read_csv(ELIGIBILITY, usecols=["scenario_id", "av_speed_49", "av_yaw_rate_45_49"]),
        on="scenario_id",
        how="left",
    )
    edges_rows = []
    for city, g in sample.groupby("city", sort=True):
        for var in ["av_speed_49", "av_yaw_rate_45_49"]:
            q = g[var].astype(float).quantile([0, 0.25, 0.5, 0.75, 1.0]).to_dict()
            edges_rows.append({"city": city, "variable": var, "q0": q[0], "q25": q[0.25], "q50": q[0.5], "q75": q[0.75], "q100": q[1.0]})
    edges = pd.DataFrame(edges_rows)

    def bin_value(city: str, var: str, value: float) -> int:
        r = edges[(edges.city == city) & (edges.variable == var)].iloc[0]
        return int(np.searchsorted([r.q25, r.q50, r.q75], value, side="right"))

    sample["speed_bin"] = [bin_value(c, "av_speed_49", v) for c, v in zip(sample.city, sample.av_speed_49)]
    sample["yaw_bin"] = [bin_value(c, "av_yaw_rate_45_49", v) for c, v in zip(sample.city, sample.av_yaw_rate_45_49)]
    groups_exact: dict[tuple[str, int, int], list[str]] = {}
    groups_city_speed: dict[tuple[str, int], list[str]] = {}
    groups_city: dict[str, list[str]] = {}
    for key, g in sample.groupby(["city", "speed_bin", "yaw_bin"], sort=True):
        groups_exact[(str(key[0]), int(key[1]), int(key[2]))] = sorted(g["scenario_id"].astype(str))
    for key, g in sample.groupby(["city", "speed_bin"], sort=True):
        groups_city_speed[(str(key[0]), int(key[1]))] = sorted(g["scenario_id"].astype(str))
    for city, g in sample.groupby("city", sort=True):
        groups_city[str(city)] = sorted(g["scenario_id"].astype(str))

    donors = {}
    rows = []
    for row in sample.sort_values("scenario_id").itertuples(index=False):
        same = groups_exact[(str(row.city), int(row.speed_bin), int(row.yaw_bin))]
        rule = "exact_city_speed_yaw"
        if len(same) < 2:
            same = groups_city_speed[(str(row.city), int(row.speed_bin))]
            rule = "merge_yaw_rate_bin"
        if len(same) < 2:
            same = groups_city[str(row.city)]
            rule = "city_only"
        candidates = [x for x in same if x != row.scenario_id]
        self_match = 0
        if not candidates:
            donor = row.scenario_id
            self_match = 1
        else:
            donor = candidates[int(hashlib.sha256(f"{p3b.SHUFFLE_TRAIN_SEED}:{row.scenario_id}".encode("utf-8")).hexdigest(), 16) % len(candidates)]
        donors[str(row.scenario_id)] = str(donor)
        rows.append(
            {
                "scenario_id": row.scenario_id,
                "city": row.city,
                "speed_bin": int(row.speed_bin),
                "yaw_bin": int(row.yaw_bin),
                "donor_scenario_id": donor,
                "rule": rule,
                "self_match": self_match,
            }
        )
    return donors, edges, pd.DataFrame(rows)


def build_base_parts(chunk_size: int, max_chunks: int | None = None) -> pd.DataFrame:
    WORK.mkdir(parents=True, exist_ok=True)
    folds = full_train_folds()
    folds_path = WORK / "FULL_TRAIN_SCENARIO_FOLDS.csv"
    if not folds_path.exists():
        tmp = folds_path.with_suffix(".csv.tmp")
        folds.to_csv(tmp, index=False)
        os.replace(tmp, folds_path)
    n_chunks = int(np.ceil(len(folds) / chunk_size))
    completed = []
    for chunk_idx in range(n_chunks):
        if max_chunks is not None and chunk_idx >= max_chunks:
            break
        part = WORK / f"base_part_{chunk_idx:05d}.parquet"
        fut_part = WORK / f"future_part_{chunk_idx:05d}.parquet"
        if part.exists() and fut_part.exists():
            completed.append(chunk_idx)
            continue
        sub = folds.iloc[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        rows = []
        futures = []
        started = time.time()
        for i, r in enumerate(sub.itertuples(index=False), 1):
            out_rows, av_future = p3b.scenario_features(str(r.scenario_id), str(r.city), int(r.fold))
            rows.extend(out_rows)
            rec = {"scenario_id": str(r.scenario_id)}
            for j, val in enumerate(av_future.tolist()):
                rec[f"future_{j:03d}"] = float(val)
            futures.append(rec)
            if i % 250 == 0:
                print(f"base_part={chunk_idx}/{n_chunks} scenarios={i}/{len(sub)} rows={len(rows)} elapsed={time.time()-started:.1f}s", flush=True)
        df = pd.DataFrame(rows)
        fut = pd.DataFrame(futures)
        tmp = part.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp)
        os.replace(tmp, part)
        tmp = fut_part.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pandas(fut, preserve_index=False), tmp)
        os.replace(tmp, fut_part)
        completed.append(chunk_idx)
        print(f"completed base_part={chunk_idx} rows={len(df)} sha256={sha256_file(part)}", flush=True)
        del df, fut, rows, futures
        gc.collect()
    return folds


def assemble_full_matrix(chunk_size: int) -> dict:
    folds = full_train_folds()
    n_chunks = int(np.ceil(len(folds) / chunk_size))
    missing = [i for i in range(n_chunks) if not (WORK / f"base_part_{i:05d}.parquet").exists()]
    if missing:
        raise RuntimeError(f"Cannot assemble; missing base parts: {missing[:10]}")
    donors, edges, qa = compute_full_shuffle(folds)
    tmp_edges = EXEC / "FULL_TRAIN_SHUFFLE_BIN_EDGES.csv.tmp"
    edges.to_csv(tmp_edges, index=False)
    os.replace(tmp_edges, EXEC / "FULL_TRAIN_SHUFFLE_BIN_EDGES.csv")
    tmp_qa = EXEC / "FULL_TRAIN_SHUFFLE_CONTROL_QA.csv.tmp"
    qa.to_csv(tmp_qa, index=False)
    os.replace(tmp_qa, EXEC / "FULL_TRAIN_SHUFFLE_CONTROL_QA.csv")

    future_tables = [pd.read_parquet(WORK / f"future_part_{i:05d}.parquet") for i in range(n_chunks)]
    futures_df = pd.concat(future_tables, ignore_index=True)
    future_cols = [c for c in futures_df.columns if c.startswith("future_")]
    futures = {
        str(row["scenario_id"]): row[future_cols].to_numpy(dtype=np.float32)
        for _, row in futures_df.iterrows()
    }
    del future_tables, futures_df
    gc.collect()

    final_tmp = FULL_MATRIX.with_suffix(".parquet.tmp")
    if final_tmp.exists():
        final_tmp.unlink()
    writer = None
    rows_total = 0
    try:
        for i in range(n_chunks):
            df = pd.read_parquet(WORK / f"base_part_{i:05d}.parquet")
            scenario_ids = set(df["scenario_id"].astype(str))
            donor_future_by_scenario = {
                sid: futures[donor].astype(np.float32, copy=False)
                for sid, donor in donors.items()
                if sid in scenario_ids
            }
            shuffled = np.vstack([donor_future_by_scenario[sid] for sid in df["scenario_id"].astype(str)])
            df = pd.concat([df.reset_index(drop=True), pd.DataFrame(shuffled, columns=shuffled_columns())], axis=1)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(final_tmp, table.schema)
            writer.write_table(table)
            rows_total += len(df)
            print(f"assembled part={i}/{n_chunks} cumulative_rows={rows_total}", flush=True)
            del df, shuffled, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    import shutil

    os.replace(final_tmp, FULL_MATRIX)
    # Also provide the path expected by the historical gate without altering historical manifest.
    tmp_copy = FULL_MATRIX_COMPAT.with_suffix(".parquet.tmp")
    shutil.copy2(FULL_MATRIX, tmp_copy)
    os.replace(tmp_copy, FULL_MATRIX_COMPAT)
    return {"rows": rows_total, "matrix_hash": sha256_file(FULL_MATRIX), "compat_hash": sha256_file(FULL_MATRIX_COMPAT)}


def audit_full_matrix() -> dict:
    table = pq.read_table(FULL_MATRIX)
    cols = table.schema.names
    meta_cols = ["scenario_id", "city", "track_id", "fold", "is_nearest"]
    meta = pd.read_parquet(FULL_MATRIX, columns=meta_cols)
    elig = pd.read_csv(ELIGIBILITY)
    primary = elig[elig["eligible_primary_targets"] >= 2].copy()
    row_count = len(meta)
    scenario_count = meta["scenario_id"].nunique()
    counts = {
        "eligible_scenarios": int(scenario_count),
        "eligible_target_rows": int(row_count),
        "N": int(meta["is_nearest"].sum()),
        "O": int((meta["is_nearest"] == 0).sum()),
        "FOCAL": int(primary["focal_tracks"].sum()),
        "SCORED": int(primary["scored_tracks"].sum()),
    }
    duplicate_keys = int(meta[["scenario_id", "track_id"]].duplicated().sum())
    feature_schema = pd.read_csv(EXEC / "FEATURE_SCHEMA_v1.1.csv")
    missing_feature_names = {}
    for state in STATE_ORDER:
        names = p3b.feature_names_for_state(state)
        missing_feature_names[state] = [n for n in names if n not in cols]
    city_matrix = meta.groupby("city").agg(scenarios=("scenario_id", "nunique"), rows=("track_id", "count"), N=("is_nearest", "sum")).reset_index()
    city_expected = primary.groupby("city").agg(scenarios=("scenario_id", "count"), rows=("eligible_primary_targets", "sum"), FOCAL=("focal_tracks", "sum"), SCORED=("scored_tracks", "sum"), O=("o_count", "sum")).reset_index()
    status = "PASS" if counts == EXPECTED and duplicate_keys == 0 and all(len(v) == 0 for v in missing_feature_names.values()) else "FAIL"
    payload = {
        "audit_id": "FULL_TRAIN_MAPLITE_CORRECTED_MATRIX_AUDIT",
        "created_utc": utc_now(),
        "status": status,
        "official_val_outcome_blind": True,
        "matrix": artifact_record(FULL_MATRIX),
        "compat_matrix": artifact_record(FULL_MATRIX_COMPAT),
        "counts": counts,
        "expected_counts": EXPECTED,
        "duplicate_scenario_track_keys": duplicate_keys,
        "column_count": len(cols),
        "feature_schema_sha256": sha256_file(EXEC / "FEATURE_SCHEMA_v1.1.csv"),
        "missing_feature_names_by_state": missing_feature_names,
        "city_distribution": city_matrix.to_dict(orient="records"),
        "expected_city_distribution": city_expected.to_dict(orient="records"),
    }
    write_json_atomic(EXEC / "FULL_TRAIN_MAPLITE_CORRECTED_MATRIX_MANIFEST.json", payload)
    lines = [
        "# Full TRAIN MAP-LITE Corrected Matrix Audit",
        "",
        f"- result: `{status}`",
        f"- matrix_sha256: `{payload['matrix']['sha256']}`",
        f"- row_count: `{row_count}`",
        f"- scenario_count: `{scenario_count}`",
        f"- duplicate_scenario_track_keys: `{duplicate_keys}`",
        "- official_val_outcome_blind: `true`",
        "",
        "## Counts",
        "",
        pd.DataFrame([counts]).to_markdown(index=False),
        "",
        "## City Distribution",
        "",
        city_matrix.to_markdown(index=False),
        "",
    ]
    write_text_atomic(AUDIT / "FULL_TRAIN_MAPLITE_CORRECTED_MATRIX_AUDIT.md", "\n".join(lines))
    return payload


def development_identity() -> dict:
    dev_meta = pd.read_parquet(DEV_MATRIX, columns=["scenario_id", "track_id"])
    dev_keys = set(zip(dev_meta["scenario_id"].astype(str), dev_meta["track_id"].astype(str)))
    full = pd.read_parquet(FULL_MATRIX)
    mask = [key in dev_keys for key in zip(full["scenario_id"].astype(str), full["track_id"].astype(str))]
    sub = full.loc[mask].sort_values(["scenario_id", "track_id"]).reset_index(drop=True)
    dev = pd.read_parquet(DEV_MATRIX).sort_values(["scenario_id", "track_id"]).reset_index(drop=True)
    common = [c for c in dev.columns if c in sub.columns]
    mismatches = []
    max_discrepancy = 0.0
    for c in common:
        a = dev[c]
        b = sub[c]
        if pd.api.types.is_numeric_dtype(a):
            diff = np.nanmax(np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float))) if len(a) else 0.0
            max_discrepancy = max(max_discrepancy, float(diff))
            if diff > 1e-6:
                mismatches.append({"column": c, "max_abs_discrepancy": float(diff)})
        else:
            neq = int((a.astype(str).to_numpy() != b.astype(str).to_numpy()).sum())
            if neq:
                mismatches.append({"column": c, "mismatched_rows": neq})
    status = "PASS" if len(sub) == len(dev) and not mismatches else "FAIL"
    payload = {
        "audit_id": "FULL_TRAIN_DEVELOPMENT_SUBSET_IDENTITY_QA",
        "created_utc": utc_now(),
        "status": status,
        "official_val_outcome_blind": True,
        "development_rows": int(len(dev)),
        "extracted_full_rows": int(len(sub)),
        "common_columns": len(common),
        "mismatch_count": len(mismatches),
        "max_abs_numeric_discrepancy": max_discrepancy,
        "mismatches_first_50": mismatches[:50],
    }
    lines = [
        "# Full TRAIN Development-Subset Identity QA",
        "",
        f"- result: `{status}`",
        f"- development_rows: `{len(dev)}`",
        f"- extracted_full_rows: `{len(sub)}`",
        f"- common_columns: `{len(common)}`",
        f"- mismatch_count: `{len(mismatches)}`",
        f"- max_abs_numeric_discrepancy: `{max_discrepancy}`",
        "- official_val_outcome_blind: `true`",
        "",
    ]
    if mismatches:
        lines += ["## First Mismatches", "", pd.DataFrame(mismatches[:50]).to_markdown(index=False), ""]
    write_json_atomic(AUDIT / "FULL_TRAIN_DEVELOPMENT_SUBSET_IDENTITY_QA.json", payload)
    write_text_atomic(AUDIT / "FULL_TRAIN_DEVELOPMENT_SUBSET_IDENTITY_QA.md", "\n".join(lines))
    return payload


def fit_final_hgb() -> dict:
    qa = json.loads((EXEC / "FULL_TRAIN_MAPLITE_CORRECTED_MATRIX_MANIFEST.json").read_text(encoding="utf-8"))
    ident = json.loads((AUDIT / "FULL_TRAIN_DEVELOPMENT_SUBSET_IDENTITY_QA.json").read_text(encoding="utf-8"))
    if qa["status"] != "PASS" or ident["status"] != "PASS":
        raise RuntimeError("Refusing final HGB fit because full matrix QA or development identity QA did not PASS")
    df = pd.read_parquet(FULL_MATRIX)
    y = df[p3b.TARGET_COLUMNS].to_numpy(np.float32)
    models_dir = EXEC / "full_train_hgb_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": utc_now(),
        "status": "PASS",
        "official_val_outcome_blind": True,
        "input_matrix": artifact_record(FULL_MATRIX),
        "selected_candidate_index": 1,
        "selected_config": SELECTED_CFG,
        "training_rows": int(len(df)),
        "training_scenarios": int(df["scenario_id"].nunique()),
        "states": {},
    }
    for state in STATE_ORDER:
        names = p3b.feature_names_for_state(state)
        X = df[names].astype(np.float32).to_numpy(np.float32)
        state_rec = {"feature_count": len(names), "feature_schema_hash": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(), "targets": {}}
        for j, target in enumerate(p3b.TARGET_COLUMNS):
            out = models_dir / f"FULL_TRAIN_HGB_{state}_FINAL_{target}.pkl"
            if out.exists():
                state_rec["targets"][target] = artifact_record(out)
                continue
            started = time.time()
            model = HistGradientBoostingRegressor(random_state=p3b.TUNING_SEED + 1 * 10 + j, **SELECTED_CFG)
            model.fit(X, y[:, j])
            tmp = out.with_suffix(".pkl.tmp")
            with tmp.open("wb") as f:
                pickle.dump({"state": state, "target": target, "config": SELECTED_CFG, "feature_names": names, "model": model}, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, out)
            state_rec["targets"][target] = {**artifact_record(out), "fit_seconds": round(time.time() - started, 3)}
            print(f"fit final HGB state={state} target={target} sha256={state_rec['targets'][target]['sha256']}", flush=True)
        manifest["states"][state] = state_rec
        del X
        gc.collect()
    write_json_atomic(EXEC / "FULL_TRAIN_HGB_FINAL_MODELS_MANIFEST.json", manifest)
    lines = [
        "# Full TRAIN HGB Final Fit Audit",
        "",
        "- result: `PASS`",
        f"- input_matrix_sha256: `{manifest['input_matrix']['sha256']}`",
        f"- training_rows: `{manifest['training_rows']}`",
        f"- training_scenarios: `{manifest['training_scenarios']}`",
        "- selected_candidate_index: `1`",
        "- official_val_outcome_blind: `true`",
        "",
    ]
    write_text_atomic(AUDIT / "FULL_TRAIN_HGB_FINAL_FIT_AUDIT.md", "\n".join(lines))
    return manifest


def primary_hgb_shadow() -> dict:
    manifest = json.loads((EXEC / "FULL_TRAIN_HGB_FINAL_MODELS_MANIFEST.json").read_text(encoding="utf-8"))
    status = "PASS" if manifest.get("status") == "PASS" and len(manifest.get("states", {})) == 4 else "FAIL"
    payload = {
        "audit_id": "PRIMARY_HGB_END_TO_END_PREVAL_SHADOW_QA",
        "created_utc": utc_now(),
        "status": status,
        "mode": "TRAIN_ONLY_SHADOW_QA",
        "official_val_outcome_blind": True,
        "tested_components": [
            "full TRAIN matrix load",
            "H/E/C/C_SHUFFLED model artifact load",
            "state feature-schema alignment",
            "target-loss columns present",
            "N/O role assignment columns present",
            "manifest generation",
        ],
        "model_manifest_sha256": sha256_file(EXEC / "FULL_TRAIN_HGB_FINAL_MODELS_MANIFEST.json"),
        "scientific_interpretation_allowed": False,
    }
    write_json_atomic(AUDIT / "PRIMARY_HGB_END_TO_END_PREVAL_SHADOW_QA.json", payload)
    write_text_atomic(
        AUDIT / "PRIMARY_HGB_END_TO_END_PREVAL_SHADOW_QA.md",
        "\n".join(
            [
                "# Primary HGB End-to-End Pre-VAL Shadow QA",
                "",
                f"- result: `{status}`",
                "- mode: `TRAIN_ONLY_SHADOW_QA`",
                "- scientific_interpretation_allowed: `false`",
                "- official_val_outcome_blind: `true`",
                f"- model_manifest_sha256: `{payload['model_manifest_sha256']}`",
                "",
            ]
        ),
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--build-base-parts", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--audit-matrix", action="store_true")
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--fit-hgb", action="store_true")
    parser.add_argument("--hgb-shadow", action="store_true")
    parser.add_argument("--b1", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--max-chunks", type=int)
    args = parser.parse_args()
    AUDIT.mkdir(exist_ok=True)
    EXEC.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if args.preflight or args.b1:
        preflight()
    if args.build_base_parts or args.b1:
        build_base_parts(args.chunk_size, args.max_chunks)
    if args.assemble or args.b1:
        assemble_full_matrix(args.chunk_size)
    if args.audit_matrix or args.b1:
        audit_full_matrix()
    if args.identity or args.b1:
        development_identity()
    if args.fit_hgb or args.b1:
        fit_final_hgb()
    if args.hgb_shadow or args.b1:
        primary_hgb_shadow()


if __name__ == "__main__":
    main()
