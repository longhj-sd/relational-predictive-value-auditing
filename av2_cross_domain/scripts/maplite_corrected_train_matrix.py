from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from maplite_features import MAP_LITE_FEATURES, compute_maplite_features, load_map_index, nearest_lane


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
EXEC = ROOT / "execution"
TRAIN_ROOT = ROOT / "data" / "raw" / "av2_motion_forecasting" / "official_s3" / "train"
PROCESSED = ROOT / "data" / "processed" / "phase3b"
OLD_MATRIX = PROCESSED / "target_rows.parquet"
NEW_MATRIX = PROCESSED / "TRAIN_50K_FEATURES_MAPLITE_CORRECTED_v2.parquet"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_frame_hash(df: pd.DataFrame, columns: list[str]) -> str:
    h = hashlib.sha256()
    for value in pd.util.hash_pandas_object(df[columns], index=False).to_numpy(dtype=np.uint64):
        h.update(int(value).to_bytes(8, "little", signed=False))
    return h.hexdigest()


def scenario_paths(sid: str) -> tuple[Path, Path]:
    folder = TRAIN_ROOT / sid
    return folder / f"scenario_{sid}.parquet", folder / f"log_map_archive_{sid}.json"


def t49_lookup(sid: str) -> tuple[pd.Series, dict[str, pd.Series]]:
    scen_path, _map_path = scenario_paths(sid)
    df = pq.read_table(
        scen_path,
        columns=["track_id", "object_type", "object_category", "timestep", "position_x", "position_y", "heading"],
    ).to_pandas()
    t49 = df[df["timestep"] == 49].copy()
    t49["track_id_str"] = t49["track_id"].astype(str)
    av = t49[t49["track_id_str"] == "AV"].iloc[0]
    by_track = {str(r.track_id_str): r for r in t49.itertuples(index=False)}
    return av, by_track


def compute_for_rows(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    grouped = list(df.groupby("scenario_id", sort=True))
    start = time.time()
    for i, (sid, group) in enumerate(grouped, 1):
        av, by_track = t49_lookup(str(sid))
        _scen_path, map_path = scenario_paths(str(sid))
        for row in group.itertuples(index=False):
            track = str(row.track_id)
            t = by_track[track]
            values = compute_maplite_features(
                map_path,
                target_xy_49=(float(t.position_x), float(t.position_y)),
                target_heading_49=float(t.heading),
                av_xy_49=(float(av.position_x), float(av.position_y)),
                av_heading_49=float(av.heading),
            )
            values.update({"scenario_id": str(sid), "track_id": track})
            out_rows.append(values)
        if i % 2500 == 0:
            print(f"maplite extraction {i}/{len(grouped)} scenarios, rows={len(out_rows)}, elapsed={time.time()-start:.1f}s", flush=True)
    return pd.DataFrame(out_rows)


def deterministic_1k(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    cities = sorted(df["city"].unique())
    base = 1000 // len(cities)
    remainder = 1000 % len(cities)
    for i, city in enumerate(cities):
        n = base + (1 if i < remainder else 0)
        pieces.append(df[df["city"] == city].sort_values(["scenario_id", "track_id"]).head(n))
    return pd.concat(pieces, ignore_index=True)


def feature_qa(df: pd.DataFrame, path_csv: Path, path_md: Path, title: str) -> pd.DataFrame:
    rows = []
    for name in MAP_LITE_FEATURES:
        missing_col = f"{name}_missing"
        nonmissing = int((df[missing_col] == 0).sum())
        missing = int((df[missing_col] == 1).sum())
        vals = df.loc[df[missing_col] == 0, name]
        row = {
            "feature": name,
            "rows": int(len(df)),
            "nonmissing_count": nonmissing,
            "missing_count": missing,
            "missing_percent": 100.0 * missing / len(df),
            "unique_value_count": int(vals.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(vals) and len(vals):
            row.update({"min": float(vals.min()), "median": float(vals.median()), "max": float(vals.max())})
        else:
            row.update({"min": np.nan, "median": np.nan, "max": np.nan})
        for city, cg in df.groupby("city"):
            row[f"city_{city}_nonmissing"] = int((cg[missing_col] == 0).sum())
            row[f"city_{city}_missing_percent"] = 100.0 * float((cg[missing_col] == 1).mean())
        rows.append(row)
    qa = pd.DataFrame(rows)
    qa.to_csv(path_csv, index=False)
    lines = [
        f"# {title}",
        "",
        f"- created_utc: `{datetime.now(timezone.utc).isoformat()}`",
        f"- rows: `{len(df)}`",
        f"- features: `{len(MAP_LITE_FEATURES)}`",
        f"- all_features_have_nonmissing_values: `{str(bool((qa['nonmissing_count'] > 0).all())).lower()}`",
        "",
        qa.to_markdown(index=False),
        "",
    ]
    path_md.write_text("\n".join(lines), encoding="utf-8")
    return qa


def write_trace(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample_scenarios = (
        df[["scenario_id", "city"]]
        .drop_duplicates()
        .sort_values(["city", "scenario_id"])
        .groupby("city")
        .head(3)
    )
    for scen in sample_scenarios.itertuples(index=False):
        sid = str(scen.scenario_id)
        scen_rows = df[df["scenario_id"] == sid].sort_values("track_id")
        target_track = str(scen_rows.iloc[0]["track_id"])
        av, by_track = t49_lookup(sid)
        t = by_track[target_track]
        _scen_path, map_path = scenario_paths(sid)
        index = load_map_index(str(map_path))
        lane = nearest_lane(index, (float(t.position_x), float(t.position_y)))
        values = compute_maplite_features(
            map_path,
            (float(t.position_x), float(t.position_y)),
            float(t.heading),
            (float(av.position_x), float(av.position_y)),
            float(av.heading),
        )
        rows.append(
            {
                "scenario_id": sid,
                "city": scen.city,
                "track_id": target_track,
                "map_path": str(map_path),
                "map_exists": map_path.exists(),
                "schema_keys": ",".join(index["schema_keys"]),
                "raw_lane_segments": index["lane_count_raw"],
                "vehicle_lane_segments": index["vehicle_lane_count"],
                "drivable_areas": index["drivable_area_count"],
                "target_x_t49_raw": float(t.position_x),
                "target_y_t49_raw": float(t.position_y),
                "target_heading_t49_raw": float(t.heading),
                "av_x_t49_raw": float(av.position_x),
                "av_y_t49_raw": float(av.position_y),
                "av_heading_t49_raw": float(av.heading),
                "nearest_lane_id": lane["lane"]["id"] if lane else "",
                "nearest_lane_distance_m": float(lane["projection"]["distance"]) if lane else math.nan,
                **{name: values[name] for name in MAP_LITE_FEATURES},
                **{f"{name}_missing": values[f"{name}_missing"] for name in MAP_LITE_FEATURES},
            }
        )
    trace = pd.DataFrame(rows)
    trace.to_csv(AUDIT / "MAPLITE_ROOT_CAUSE_TRACE.csv", index=False)
    return trace


def write_root_cause(raw_cov: pd.DataFrame, trace: pd.DataFrame) -> None:
    lines = [
        "# MAP-LITE Root-Cause Analysis",
        "",
        f"- created_utc: `{datetime.now(timezone.utc).isoformat()}`",
        "- scope: TRAIN-only; official VAL not read, predicted, or evaluated",
        "- classification: `IMPLEMENTATION_DEFECT`",
        "",
        "## Exact Root Cause",
        "",
        "`tools/phase3b_train_only_execution.py` previously filled every frozen MAP-LITE base feature with `0.0` and set every corresponding missing indicator to `1` inside `scenario_features()`. The feature builder did not load `log_map_archive_*.json`, did not parse lane segments or drivable areas, and did not execute lane/drivable lookup logic. Raw TRAIN maps were present.",
        "",
        "## Raw TRAIN Map Coverage",
        "",
        raw_cov.to_markdown(index=False),
        "",
        "## Local AV2 Map Schema",
        "",
        "Representative TRAIN map JSON files expose top-level keys `drivable_areas`, `lane_segments`, and `pedestrian_crossings`. Vehicle lane segments expose `centerline`, `is_intersection`, `lane_type`, left/right neighbor ids, predecessor/successor lists, and lane boundaries. Drivable areas expose `area_boundary` polygons.",
        "",
        "## Reconstructability",
        "",
        "All 16 frozen MAP-LITE variables are reconstructable from local TRAIN map JSON and t49 TRAIN scenario state. `turn_direction` is reconstructed from centerline heading change because the local schema does not contain an explicit turn label. No new map variables were added.",
        "",
        "## Trace Summary",
        "",
        trace[["city", "scenario_id", "track_id", "vehicle_lane_segments", "drivable_areas", "nearest_lane_id", "nearest_lane_distance_m"]].to_markdown(index=False),
        "",
        "## Geometric Definitions",
        "",
        "- nearest lane: deterministic nearest vehicle-lane centerline segment by squared distance, then lane id, then segment index",
        "- map_nearest_lane_heading: nearest segment heading in the AV-fixed t49 frame, radians",
        "- map_target_to_lane_heading: target t49 heading minus nearest-lane heading in the AV-fixed t49 frame, wrapped to [-pi, pi]",
        "- map_lateral_lane_coord: signed perpendicular offset from nearest centerline segment, meters",
        "- map_longitudinal_lane_coord: arc length from lane centerline start to target projection, meters",
        "- intersection/neighbor/predecessor/successor: direct lane-segment attributes from the nearest vehicle lane",
        "- turn direction: left/right/straight from total centerline heading change with 20 degree threshold",
        "- same lane and lane-heading difference: target nearest lane compared with AV nearest vehicle lane at t49",
        "- drivable membership: point-in-polygon over local drivable-area boundaries",
        "- boundary distance: minimum Euclidean distance to any drivable-area polygon boundary, meters",
        "",
    ]
    (AUDIT / "MAPLITE_ROOT_CAUSE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(OLD_MATRIX)
    raw = pd.read_csv(AUDIT / "TRAIN_ELIGIBILITY_SCENARIOS.csv", usecols=["scenario_id", "city", "map_json_present", "map_lite_missing"])
    raw_cov = raw.groupby("city", as_index=False).agg(
        train_scenarios=("scenario_id", "count"),
        map_json_present=("map_json_present", "sum"),
        map_lite_missing=("map_lite_missing", "sum"),
    )
    raw_cov["map_json_coverage_percent"] = 100.0 * raw_cov["map_json_present"] / raw_cov["train_scenarios"]

    trace = write_trace(old)
    write_root_cause(raw_cov, trace)

    sample_1k = deterministic_1k(old)
    map_1k = compute_for_rows(sample_1k)
    corrected_1k = sample_1k.copy().reset_index(drop=True)
    corrected_1k.update(map_1k.drop(columns=["scenario_id", "track_id"]))
    feature_qa(corrected_1k, AUDIT / "MAPLITE_CORRECTED_1K_QA.csv", AUDIT / "MAPLITE_CORRECTED_1K_QA.md", "MAP-LITE Corrected 1K QA")
    if corrected_1k[[f"{name}_missing" for name in MAP_LITE_FEATURES]].mean().max() >= 0.95:
        raise RuntimeError("1k MAP-LITE QA failed: at least one feature remains near-all missing")

    map_all = compute_for_rows(old)
    corrected = old.copy().reset_index(drop=True)
    corrected.update(map_all.drop(columns=["scenario_id", "track_id"]))
    pq.write_table(pa.Table.from_pandas(corrected, preserve_index=False), NEW_MATRIX)
    feature_qa(
        corrected,
        AUDIT / "MAPLITE_FEATURE_MISSINGNESS_CORRECTED_v2.csv",
        AUDIT / "MAPLITE_FEATURE_MISSINGNESS_CORRECTED_v2.md",
        "MAP-LITE Feature Missingness Corrected v2",
    )

    map_cols = MAP_LITE_FEATURES + [f"{name}_missing" for name in MAP_LITE_FEATURES]
    non_map_cols = [c for c in old.columns if c not in map_cols]
    key_cols = ["scenario_id", "city", "track_id", "fold", "is_nearest"]
    comparison = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "old_matrix": str(OLD_MATRIX),
        "new_matrix": str(NEW_MATRIX),
        "old_matrix_sha256": sha256_file(OLD_MATRIX),
        "new_matrix_sha256": sha256_file(NEW_MATRIX),
        "row_count_old": int(len(old)),
        "row_count_new": int(len(corrected)),
        "key_columns_identical": bool(old[key_cols].equals(corrected[key_cols])),
        "folds_identical": bool(old["fold"].equals(corrected["fold"])),
        "non_map_columns_identical": bool(old[non_map_cols].equals(corrected[non_map_cols])),
        "non_map_hash_old": stable_frame_hash(old, non_map_cols),
        "non_map_hash_new": stable_frame_hash(corrected, non_map_cols),
        "map_columns_changed": bool(not old[map_cols].equals(corrected[map_cols])),
        "old_cv_status": "superseded-for-final-selection",
        "cv_rerun_required": True,
    }
    (AUDIT / "MAPLITE_PRE_POST_COMPARISON_v2.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# MAP-LITE Pre/Post Comparison v2",
        "",
        f"- old_matrix_sha256: `{comparison['old_matrix_sha256']}`",
        f"- new_matrix_sha256: `{comparison['new_matrix_sha256']}`",
        f"- rows identical: `{comparison['row_count_old'] == comparison['row_count_new']}`",
        f"- key columns identical: `{str(comparison['key_columns_identical']).lower()}`",
        f"- folds identical: `{str(comparison['folds_identical']).lower()}`",
        f"- non-map columns identical: `{str(comparison['non_map_columns_identical']).lower()}`",
        f"- non-map hash old: `{comparison['non_map_hash_old']}`",
        f"- non-map hash new: `{comparison['non_map_hash_new']}`",
        f"- map columns changed: `{str(comparison['map_columns_changed']).lower()}`",
        f"- old CV status: `{comparison['old_cv_status']}`",
        f"- CV rerun required: `{str(comparison['cv_rerun_required']).lower()}`",
        "",
        "The corrected matrix preserves scenario IDs, target rows, fold assignments, seeds, targets, and all non-map feature values. Only the frozen MAP-LITE base features and their missingness indicators changed.",
        "",
    ]
    (AUDIT / "MAPLITE_PRE_POST_COMPARISON_v2.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
