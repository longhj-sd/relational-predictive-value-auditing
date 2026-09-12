from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import itertools
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from maplite_features import MAP_LITE_FEATURES, compute_maplite_features
import phase3a_train_only_freeze as p3a


ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "execution"
AUDIT = ROOT / "audit"
PROTOCOL = ROOT / "protocol"
PROCESSED = ROOT / "data" / "processed" / "phase3b"
TRAIN_ROOT = p3a.TRAIN_ROOT
LOGS = ROOT / "logs"

PROTOCOL_HASH = "eb1acb13ae08ea3823118956b4cf8be4d86d7d78e09d6aeae3bf811f6d9cf244"
PROVISIONAL_EXECUTION_HASH = "e56ec1c6ae7d0cbefa35f4583e3d4f09a3c28fa90f2d217d27abc8e7be6824c0"
SPLIT_SEED = p3a.SPLIT_SEED
TUNING_SEED = p3a.TUNING_SEED
SHUFFLE_TRAIN_SEED = p3a.SHUFFLE_TRAIN_SEED

STATE_ORDER = ["H", "E", "C", "C_SHUFFLED"]
META_COLUMNS = ["scenario_id", "city", "track_id", "fold", "is_nearest"]
TARGET_COLUMNS = ["dx_6s", "dy_6s"]
HGB_UNIT_COLUMNS = ["state", "candidate_index", "fold"]
PRIMARY_CV_COLUMNS = [
    "state",
    "candidate_index",
    "fold",
    "learning_rate",
    "max_iter",
    "max_leaf_nodes",
    "l2_regularization",
    "min_samples_leaf",
    "max_bins",
    "dx_6s_mse",
    "dy_6s_mse",
    "mean_endpoint_mse",
    "mean_fde",
    "median_fde",
    "n_train_rows",
    "n_dev_rows",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_dirs() -> None:
    for path in [EXEC, AUDIT, PROTOCOL, PROCESSED, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def write_csv_safely(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def unit_key(row: pd.Series | dict) -> tuple[str, int, int]:
    return (str(row["state"]), int(row["candidate_index"]), int(row["fold"]))


def read_primary_cv_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=PRIMARY_CV_COLUMNS)
    return pd.read_csv(path)


def hgb_completed_keys_strict(path: Path, candidate_limit: int) -> set[tuple[str, int, int]]:
    df = read_primary_cv_raw(path)
    if df.empty:
        return set()
    missing = [c for c in HGB_UNIT_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} is malformed; missing key columns: {missing}")
    expected = hgb_expected_units(candidate_limit)
    keys = [unit_key(r) for _, r in df.iterrows()]
    invalid = [k for k in keys if k not in expected]
    if invalid:
        raise RuntimeError(f"{path} contains unexpected HGB unit keys, first invalid key: {invalid[0]}")
    duplicates = df.duplicated(HGB_UNIT_COLUMNS, keep=False)
    if bool(duplicates.any()):
        dup = unit_key(df.loc[duplicates].iloc[0])
        raise RuntimeError(f"{path} contains duplicate HGB unit key: {dup}")
    return set(keys)


def count_hgb_key(path: Path, key: tuple[str, int, int]) -> int:
    df = read_primary_cv_raw(path)
    if df.empty:
        return 0
    return int(((df["state"] == key[0]) & (df["candidate_index"].astype(int) == key[1]) & (df["fold"].astype(int) == key[2])).sum())


def primary_cv_row_line(row: dict, columns: list[str]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    writer.writerow({c: row.get(c, "") for c in columns})
    return buf.getvalue()


def append_primary_cv_row_atomic(path: Path, row: dict) -> str:
    key = unit_key(row)
    before_count = count_hgb_key(path, key)
    if before_count != 0:
        raise RuntimeError(f"Refusing to append duplicate HGB unit key {key}; existing count={before_count}")

    if path.exists():
        existing_bytes = path.read_bytes()
        header = existing_bytes.splitlines()[0].decode("utf-8-sig")
        columns = next(csv.reader([header]))
        missing = [c for c in PRIMARY_CV_COLUMNS if c not in columns]
        if missing:
            raise RuntimeError(f"{path} is malformed; missing expected columns: {missing}")
        prefix = existing_bytes
        if prefix and not prefix.endswith((b"\n", b"\r")):
            prefix += b"\n"
        new_bytes = prefix + primary_cv_row_line(row, columns).encode("utf-8")
    else:
        columns = PRIMARY_CV_COLUMNS
        header_buf = io.StringIO(newline="")
        writer = csv.DictWriter(header_buf, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        new_bytes = (header_buf.getvalue() + primary_cv_row_line(row, columns)).encode("utf-8")

    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_bytes(new_bytes)
    os.replace(tmp, path)

    after_count = count_hgb_key(path, key)
    if after_count != 1:
        raise RuntimeError(f"Post-write verification failed for HGB unit key {key}; count={after_count}")
    return hashlib.sha256(primary_cv_row_line(row, columns).encode("utf-8")).hexdigest()


def hgb_expected_units(candidate_limit: int) -> set[tuple[str, int, int]]:
    return {
        (state, candidate_index, fold)
        for state in STATE_ORDER
        for candidate_index in range(candidate_limit)
        for fold in range(5)
    }


def read_hgb_results(path: Path, candidate_limit: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    before = len(df)
    df = df.drop_duplicates(HGB_UNIT_COLUMNS, keep="last").copy()
    expected = hgb_expected_units(candidate_limit)
    df = df[
        df.apply(lambda r: (r["state"], int(r["candidate_index"]), int(r["fold"])) in expected, axis=1)
    ].copy()
    if len(df) != before:
        write_csv_safely(df, path)
    return df


def hgb_units_in_order(candidate_limit: int) -> list[tuple[str, int, int]]:
    return [
        (state, candidate_index, fold)
        for state in STATE_ORDER
        for candidate_index in range(candidate_limit)
        for fold in range(5)
    ]


def hgb_resume_status(candidate_limit: int) -> dict:
    path = EXEC / "PRIMARY_CV_RESULTS_v2.csv"
    df = read_primary_cv_raw(path)
    expected = hgb_expected_units(candidate_limit)
    completed = set()
    if not df.empty:
        completed = {
            (r.state, int(r.candidate_index), int(r.fold))
            for r in df.itertuples(index=False)
        }
    remaining = expected - completed
    by_state_completed = {state: sum(1 for unit in completed if unit[0] == state) for state in STATE_ORDER}
    by_state_remaining = {state: sum(1 for unit in remaining if unit[0] == state) for state in STATE_ORDER}
    return {
        "candidate_limit": candidate_limit,
        "expected_units": len(expected),
        "completed_units": len(completed),
        "remaining_units": len(remaining),
        "completed_by_state": by_state_completed,
        "remaining_by_state": by_state_remaining,
    }


def scenario_path(sid: str) -> Path:
    return TRAIN_ROOT / sid / f"scenario_{sid}.parquet"


def config_candidates(n: int = 24) -> list[dict]:
    rng = random.Random(TUNING_SEED)
    keys = list(p3a.HGB_SPACE)
    all_configs = [dict(zip(keys, vals)) for vals in itertools.product(*(p3a.HGB_SPACE[k] for k in keys))]
    return rng.sample(all_configs, n)


def city_stratified_sample_and_folds() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    elig = pd.read_csv(AUDIT / "TRAIN_ELIGIBILITY_SCENARIOS.csv")
    primary = elig.loc[elig["eligible_primary_targets"] >= 2, ["scenario_id", "city", "eligible_primary_targets"]].copy()
    counts = primary.groupby("city").size().sort_index()
    total = int(counts.sum())
    raw = counts * (50000 / total)
    floors = np.floor(raw).astype(int)
    remainder = 50000 - int(floors.sum())
    frac = (raw - floors).sort_values(ascending=False)
    alloc = floors.copy()
    for city in frac.index[:remainder]:
        alloc.loc[city] += 1

    selected_rows = []
    audit_rows = []
    for city, g in primary.groupby("city", sort=True):
        ordered = g.assign(
            sample_hash=g["scenario_id"].map(
                lambda sid: hashlib.sha256(f"{SPLIT_SEED}:{sid}".encode("utf-8")).hexdigest()
            )
        ).sort_values(["sample_hash", "scenario_id"])
        n = int(alloc.loc[city])
        sel = ordered.head(n).copy()
        selected_rows.append(sel)
        audit_rows.append(
            {
                "city": city,
                "eligible_city_count": int(counts.loc[city]),
                "expected_allocation": float(raw.loc[city]),
                "final_allocation": n,
            }
        )

    selected = pd.concat(selected_rows, ignore_index=True)
    fold_rows = []
    for city, g in selected.groupby("city", sort=True):
        ordered_ids = sorted(
            g["scenario_id"],
            key=lambda sid: hashlib.sha256(f"{SPLIT_SEED}:{city}:{sid}".encode("utf-8")).hexdigest(),
        )
        for i, sid in enumerate(ordered_ids):
            fold_rows.append({"scenario_id": sid, "city": city, "fold": i % 5, "train_compute_sample": True})
    folds = pd.DataFrame(fold_rows).sort_values(["city", "fold", "scenario_id"]).reset_index(drop=True)
    fold_counts = folds.groupby(["city", "fold"]).size().reset_index(name="scenario_count")
    return folds, pd.DataFrame(audit_rows), fold_counts


def write_sample_artifacts() -> dict:
    folds, allocation, fold_counts = city_stratified_sample_and_folds()
    folds.to_csv(EXEC / "TRAIN_SCENARIO_FOLDS_v2.csv", index=False)
    sample_hash = sha256_file(EXEC / "TRAIN_SCENARIO_FOLDS_v2.csv")
    (EXEC / "TRAIN_50K_SAMPLE_HASH.txt").write_text(f"TRAIN_50K_STRATIFIED_SAMPLE_SHA256={sample_hash}\n", encoding="utf-8")

    lines = [
        "# TRAIN 50K Stratified Audit",
        "",
        "Implementation-compliance correction made outcome-blind using TRAIN metadata only.",
        "",
        f"- split seed: `{SPLIT_SEED}`",
        f"- primary-eligible TRAIN scenarios: `{int(allocation['eligible_city_count'].sum())}`",
        f"- selected scenarios: `{len(folds)}`",
        f"- scenario-ID hash: `{sample_hash}`",
        "",
        "## City Allocations",
        "",
        allocation.to_markdown(index=False),
        "",
        "## Fold By City",
        "",
        fold_counts.pivot(index="city", columns="fold", values="scenario_count").to_markdown(),
        "",
    ]
    (EXEC / "TRAIN_50K_STRATIFIED_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"sample_hash": sample_hash, "allocation": allocation, "fold_counts": fold_counts, "folds": folds}


def write_confirmatory_rule() -> str:
    text = f"""# Confirmatory Decision Rule, Pre-Outcome

Freeze date: 2026-09-07, Asia/Shanghai.

Scientific protocol v1.1 hash:

```text
{PROTOCOL_HASH}
```

Gate 1 is established iff the lower bound of the dependence-aware two-sided
95% confidence interval for `Delta_aggregate(H->C)` is greater than zero.

Only if Gate 1 is established may Gate 2 be interpreted as confirmatory.

Gate 2 supports relational localization iff both:

1. the dependence-aware two-sided 95% CI for `lambda_AV2` excludes zero; and
2. the two-sided pseudo-nearest add-one corrected permutation P value is < 0.05.

The pseudo-nearest P value is based on absolute extremeness:

```text
|lambda_null| >= |lambda_observed|
```

The sign of `lambda_AV2` determines direction only.

If `lambda_AV2 > 0`, nearest targets obtain greater incremental predictive
value.

If `lambda_AV2 < 0`, other targets obtain greater incremental predictive value.

Positive lambda is not required for Gate 2 success. If CI and permutation
evidence disagree, report mixed evidence and do not call Gate 2 confirmatory
success.

Official VAL remains untouched by this decision-rule freeze.
"""
    path = PROTOCOL / "CONFIRMATORY_DECISION_RULE_PRE_OUTCOME.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    digest = sha256_file(path)
    (PROTOCOL / "CONFIRMATORY_DECISION_RULE_PRE_OUTCOME.sha256.txt").write_text(
        f"CONFIRMATORY_DECISION_RULE_PRE_OUTCOME_SHA256={digest}\n", encoding="utf-8"
    )
    return digest


def wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(x), np.cos(x))


def transform_xy(xy: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray:
    shifted = xy - origin
    c = math.cos(-heading)
    s = math.sin(-heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return shifted @ rot.T


def scenario_features(sid: str, city: str, fold: int) -> tuple[list[dict], np.ndarray]:
    path = scenario_path(sid)
    df = pq.read_table(
        path,
        columns=[
            "observed",
            "track_id",
            "object_type",
            "object_category",
            "timestep",
            "position_x",
            "position_y",
            "heading",
            "velocity_x",
            "velocity_y",
            "scenario_id",
            "city",
        ],
    ).to_pandas()
    av = df[df["track_id"] == "AV"].sort_values("timestep")
    av49 = av[av["timestep"] == p3a.ANCHOR_STEP].iloc[0]
    av_origin = av49[["position_x", "position_y"]].to_numpy(dtype=np.float32)
    av_heading = float(av49["heading"])
    av_hist = av[av["timestep"].isin(p3a.HISTORY_STEPS)].sort_values("timestep")
    av_future = av[av["timestep"].isin(p3a.FUTURE_STEPS)].sort_values("timestep")
    av_hist_xy = transform_xy(av_hist[["position_x", "position_y"]].to_numpy(dtype=np.float32), av_origin, av_heading)
    av_future_xy = transform_xy(av_future[["position_x", "position_y"]].to_numpy(dtype=np.float32), av_origin, av_heading)
    av_future_v = transform_xy(
        av_future[["velocity_x", "velocity_y"]].to_numpy(dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        av_heading,
    )
    av_v49 = transform_xy(av49[["velocity_x", "velocity_y"]].to_numpy(dtype=np.float32)[None, :], np.zeros(2), av_heading)[0]
    av_headings = av_hist["heading"].to_numpy(dtype=np.float32)
    av_yaw = float(np.mean(np.abs(wrap_angle(np.diff(av_headings))))) if len(av_headings) >= 2 else 0.0
    av_endpoint = av_future_xy[-1]
    av_endpoint_heading = float(wrap_angle(float(av_future.iloc[-1]["heading"]) - av_heading))
    av_future_flat = []
    for xy, vv in zip(av_future_xy, av_future_v):
        av_future_flat.extend([float(xy[0]), float(xy[1]), float(vv[0]), float(vv[1])])

    primary = p3a._eligible_tracks(df, p3a.PRIMARY_TYPES, p3a.PRIMARY_CATEGORIES)
    anchor = primary[primary["timestep"] == p3a.ANCHOR_STEP]
    n_track, _ = p3a.assign_role_n_o(anchor, av_origin)
    dyn49 = df[
        (df["timestep"] == p3a.ANCHOR_STEP)
        & (df["observed"])
        & (df["track_id"] != "AV")
        & (df["object_type"].isin(p3a.ALL_DYNAMIC_TYPES))
    ].copy()

    rows = []
    for track_id, tg in primary.groupby("track_id", sort=True):
        tg = tg.sort_values("timestep")
        hist = tg[tg["timestep"].isin(p3a.HISTORY_STEPS)].sort_values("timestep")
        fut109 = tg[tg["timestep"] == p3a.ENDPOINT_STEP].iloc[0]
        t49 = hist.iloc[-1]
        xy_hist = transform_xy(hist[["position_x", "position_y"]].to_numpy(dtype=np.float32), av_origin, av_heading)
        v_hist = transform_xy(hist[["velocity_x", "velocity_y"]].to_numpy(dtype=np.float32), np.zeros(2), av_heading)
        target_xy49 = xy_hist[-1]
        target_v49 = v_hist[-1]
        endpoint_xy = transform_xy(fut109[["position_x", "position_y"]].to_numpy(dtype=np.float32)[None, :], av_origin, av_heading)[0]
        target_heading49 = float(wrap_angle(float(t49["heading"]) - av_heading))
        speeds = np.linalg.norm(v_hist, axis=1)
        acc = np.diff(v_hist, axis=0)
        steps = np.diff(xy_hist, axis=0)
        rel = target_xy49
        dist = float(np.linalg.norm(rel))
        bearing = math.atan2(float(rel[1]), float(rel[0])) if dist > 0 else 0.0
        surround = dyn49[dyn49["track_id"] != track_id]
        if surround.empty:
            surround_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            sxy = transform_xy(surround[["position_x", "position_y"]].to_numpy(dtype=np.float32), av_origin, av_heading)
            sv = transform_xy(surround[["velocity_x", "velocity_y"]].to_numpy(dtype=np.float32), np.zeros(2), av_heading)
            d = np.linalg.norm(sxy - target_xy49, axis=1)
            rel_speed = np.linalg.norm(sv - target_v49, axis=1)
            closing = np.maximum(0.1, rel_speed)
            ttc = d / closing
            counts = surround["object_type"].astype(str).value_counts()
            surround_stats = [
                int(len(surround)),
                float(np.min(d)),
                float(np.mean(d)),
                float(np.min(ttc)),
                float(np.mean(rel_speed)),
                int(counts.get("vehicle", 0)),
                int(counts.get("pedestrian", 0)),
                int(counts.get("cyclist", 0)),
                int(counts.get("motorcyclist", 0)),
                int(counts.get("bus", 0)),
            ]
        base = {
            "scenario_id": sid,
            "city": city,
            "track_id": str(track_id),
            "fold": int(fold),
            "is_nearest": int(str(track_id) == str(n_track)),
            "dx_6s": float(endpoint_xy[0] - target_xy49[0]),
            "dy_6s": float(endpoint_xy[1] - target_xy49[1]),
            "cv_dx_6s": float(target_v49[0] * 6.0),
            "cv_dy_6s": float(target_v49[1] * 6.0),
            "target_x_49": float(target_xy49[0]),
            "target_y_49": float(target_xy49[1]),
            "target_heading_49": target_heading49,
            "target_vx_49": float(target_v49[0]),
            "target_vy_49": float(target_v49[1]),
            "target_speed_49": float(speeds[-1]),
            "target_ax_mean_0_49": float(np.mean(acc[:, 0])) if len(acc) else 0.0,
            "target_ay_mean_0_49": float(np.mean(acc[:, 1])) if len(acc) else 0.0,
            "target_speed_mean_0_49": float(np.mean(speeds)),
            "target_speed_std_0_49": float(np.std(speeds)),
            "target_heading_sin_49": math.sin(target_heading49),
            "target_heading_cos_49": math.cos(target_heading49),
            "target_displacement_x_0_49": float(xy_hist[-1, 0] - xy_hist[0, 0]),
            "target_displacement_y_0_49": float(xy_hist[-1, 1] - xy_hist[0, 1]),
            "target_path_length_0_49": float(np.sum(np.linalg.norm(steps, axis=1))) if len(steps) else 0.0,
            "av_heading_49_fixed": 0.0,
            "av_vx_49_fixed": float(av_v49[0]),
            "av_vy_49_fixed": float(av_v49[1]),
            "av_speed_49": float(np.linalg.norm(av_v49)),
            "av_yaw_rate_abs_mean_45_49": av_yaw,
            "av_displacement_x_0_49_fixed": float(av_hist_xy[-1, 0] - av_hist_xy[0, 0]),
            "av_displacement_y_0_49_fixed": float(av_hist_xy[-1, 1] - av_hist_xy[0, 1]),
            "target_to_av_x_49": float(rel[0]),
            "target_to_av_y_49": float(rel[1]),
            "target_to_av_distance_49": dist,
            "target_to_av_bearing_sin_49": math.sin(bearing),
            "target_to_av_bearing_cos_49": math.cos(bearing),
            "target_heading_minus_av_heading_sin_49": math.sin(target_heading49),
            "target_heading_minus_av_heading_cos_49": math.cos(target_heading49),
            "target_velocity_rel_av_x_49": float(target_v49[0] - av_v49[0]),
            "target_velocity_rel_av_y_49": float(target_v49[1] - av_v49[1]),
            "surround_count_observed_49": surround_stats[0],
            "surround_nearest_distance_49": surround_stats[1],
            "surround_mean_distance_49": surround_stats[2],
            "surround_min_ttc_proxy_49": surround_stats[3],
            "surround_mean_rel_speed_49": surround_stats[4],
            "surround_vehicle_count_49": surround_stats[5],
            "surround_pedestrian_count_49": surround_stats[6],
            "surround_cyclist_count_49": surround_stats[7],
            "surround_motorcyclist_count_49": surround_stats[8],
            "surround_bus_count_49": surround_stats[9],
            "target_history_missing_indicator": 0,
            "av_history_missing_indicator": 0,
            "surrounding_agents_padded_indicator": int(surround_stats[0] < 32),
            "av_endpoint_x_109_fixed": float(av_endpoint[0]),
            "av_endpoint_y_109_fixed": float(av_endpoint[1]),
            "av_endpoint_disp_109": float(np.linalg.norm(av_endpoint)),
            "av_endpoint_heading_109_fixed": av_endpoint_heading,
        }
        map_values = compute_maplite_features(
            path.parent / f"log_map_archive_{sid}.json",
            target_xy_49=(float(t49["position_x"]), float(t49["position_y"])),
            target_heading_49=float(t49["heading"]),
            av_xy_49=(float(av49["position_x"]), float(av49["position_y"])),
            av_heading_49=av_heading,
        )
        base.update(map_values)
        for t, offset in zip(p3a.FUTURE_STEPS, range(0, len(av_future_flat), 4)):
            base[f"av_future_x_{t}_fixed"] = av_future_flat[offset]
            base[f"av_future_y_{t}_fixed"] = av_future_flat[offset + 1]
            base[f"av_future_vx_{t}_fixed"] = av_future_flat[offset + 2]
            base[f"av_future_vy_{t}_fixed"] = av_future_flat[offset + 3]
        rows.append(base)
    return rows, np.asarray(av_future_flat, dtype=np.float32)


def compute_shuffle(folds: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
    elig = pd.read_csv(AUDIT / "TRAIN_ELIGIBILITY_SCENARIOS.csv")
    sample = folds.merge(elig[["scenario_id", "av_speed_49", "av_yaw_rate_45_49"]], on="scenario_id", how="left")
    edges_rows = []
    for city, g in sample.groupby("city", sort=True):
        for var in ["av_speed_49", "av_yaw_rate_45_49"]:
            q = g[var].astype(float).quantile([0, 0.25, 0.5, 0.75, 1.0]).to_dict()
            edges_rows.append({"city": city, "variable": var, "q0": q[0], "q25": q[0.25], "q50": q[0.5], "q75": q[0.75], "q100": q[1.0]})
    edges = pd.DataFrame(edges_rows)
    edges.to_csv(EXEC / "SHUFFLE_BIN_EDGES_v2.csv", index=False)

    def bin_value(city: str, var: str, value: float) -> int:
        r = edges[(edges.city == city) & (edges.variable == var)].iloc[0]
        cuts = [r.q25, r.q50, r.q75]
        return int(np.searchsorted(cuts, value, side="right"))

    sample["speed_bin"] = [bin_value(c, "av_speed_49", v) for c, v in zip(sample.city, sample.av_speed_49)]
    sample["yaw_bin"] = [bin_value(c, "av_yaw_rate_45_49", v) for c, v in zip(sample.city, sample.av_yaw_rate_45_49)]
    groups_exact: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    groups_city_speed: dict[tuple[str, int], list[str]] = defaultdict(list)
    groups_city: dict[str, list[str]] = defaultdict(list)
    for r in sample.itertuples(index=False):
        groups_exact[(r.city, int(r.speed_bin), int(r.yaw_bin))].append(r.scenario_id)
        groups_city_speed[(r.city, int(r.speed_bin))].append(r.scenario_id)
        groups_city[r.city].append(r.scenario_id)
    for d in [groups_exact, groups_city_speed, groups_city]:
        for key in d:
            d[key] = sorted(d[key])

    donors = {}
    qa_rows = []
    for _, row in sample.sort_values("scenario_id").iterrows():
        same = groups_exact[(row.city, int(row.speed_bin), int(row.yaw_bin))]
        rule = "exact_city_speed_yaw"
        if len(same) < 2:
            same = groups_city_speed[(row.city, int(row.speed_bin))]
            rule = "merge_yaw_rate_bin"
        if len(same) < 2:
            same = groups_city[row.city]
            rule = "city_only"
        candidates = [x for x in same if x != row.scenario_id]
        self_match = 0
        if not candidates:
            donor = row.scenario_id
            self_match = 1
        else:
            donor = candidates[int(hashlib.sha256(f"{SHUFFLE_TRAIN_SEED}:{row.scenario_id}".encode()).hexdigest(), 16) % len(candidates)]
        donors[row.scenario_id] = donor
        qa_rows.append({"scenario_id": row.scenario_id, "city": row.city, "speed_bin": row.speed_bin, "yaw_bin": row.yaw_bin, "donor_scenario_id": donor, "rule": rule, "self_match": self_match})
    qa = pd.DataFrame(qa_rows)
    qa.to_csv(EXEC / "SHUFFLE_CONTROL_QA.csv", index=False)
    return donors, edges, qa


def build_features(force: bool = False) -> dict:
    meta_path = PROCESSED / "target_rows.parquet"
    if meta_path.exists() and not force:
        return {
            "target_rows": pd.read_parquet(meta_path),
            "hash": sha256_file(meta_path),
            "rebuilt": False,
        }

    folds = pd.read_csv(EXEC / "TRAIN_SCENARIO_FOLDS_v2.csv")
    donors, edges, qa = compute_shuffle(folds)
    rows = []
    futures = {}
    start = time.time()
    for i, r in enumerate(folds.itertuples(index=False), 1):
        out_rows, av_future = scenario_features(r.scenario_id, r.city, int(r.fold))
        rows.extend(out_rows)
        futures[r.scenario_id] = av_future
        if i % 1000 == 0:
            print(f"feature extraction {i}/{len(folds)} scenarios, rows={len(rows)}, elapsed={time.time()-start:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    shuffled_cols = []
    for t in p3a.FUTURE_STEPS:
        shuffled_cols.extend(
            [
                f"shuffled_av_future_x_{t}_fixed",
                f"shuffled_av_future_y_{t}_fixed",
                f"shuffled_av_future_vx_{t}_fixed",
                f"shuffled_av_future_vy_{t}_fixed",
            ]
        )
    donor_future_by_scenario = {
        sid: futures[donor].astype(np.float32, copy=False)
        for sid, donor in donors.items()
    }
    shuffled_matrix = np.vstack([donor_future_by_scenario[sid] for sid in df["scenario_id"]])
    df = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(shuffled_matrix, columns=shuffled_cols)],
        axis=1,
    )

    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), meta_path)
    write_maplite_missingness(df)
    write_shuffle_report(qa, edges, df)
    return {"target_rows": df, "hash": sha256_file(meta_path), "rebuilt": True}


def feature_names_for_state(state: str) -> list[str]:
    rows = pd.DataFrame(p3a.feature_rows())
    names = rows.loc[rows.state == state, "feature_name"].tolist()
    if state == "C_SHUFFLED":
        names = [n.replace("av_future_", "shuffled_av_future_") if n.startswith("av_future_") else n for n in names]
    return names


def write_maplite_missingness(df: pd.DataFrame) -> None:
    rows = []
    n = len(df)
    for name in MAP_LITE_FEATURES:
        missing_col = f"{name}_missing"
        missing = int(df[missing_col].sum())
        rows.append(
            {
                "feature": name,
                "missing_count": missing,
                "missing_percent": 100.0 * missing / n,
                "neutral_imputation_count": missing,
                "missing_indicator_frequency": float(df[missing_col].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(AUDIT / "MAPLITE_FEATURE_MISSINGNESS.csv", index=False)
    md = [
        "# MAP-LITE Feature QA",
        "",
        "MAP-LITE QA is computed from generated TRAIN development target rows, not from map-JSON presence alone.",
        "No reliable local AV2 lane-graph extractor was present in this repository; therefore MAP-LITE variables were neutral-imputed with explicit missing indicators for TRAIN model development.",
        "",
        f"- target rows audited: `{n}`",
        f"- features audited: `{len(rows)}`",
        f"- all missing indicators retained in H/E/C/C_SHUFFLED: `true`",
        "",
        out.to_markdown(index=False),
        "",
    ]
    (AUDIT / "MAPLITE_FEATURE_QA.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_shuffle_report(qa: pd.DataFrame, edges: pd.DataFrame, df: pd.DataFrame) -> None:
    donor_disp = df.groupby("scenario_id")[["av_future_x_109_fixed", "av_future_y_109_fixed"]].first()
    qa = qa.copy()
    qa["observed_disp_109"] = qa["scenario_id"].map(lambda s: float(np.linalg.norm(donor_disp.loc[s].to_numpy())))
    qa["donor_disp_109"] = qa["donor_scenario_id"].map(lambda s: float(np.linalg.norm(donor_disp.loc[s].to_numpy())))
    qa["donor_minus_observed_disp_109"] = qa["donor_disp_109"] - qa["observed_disp_109"]
    qa.to_csv(EXEC / "SHUFFLE_CONTROL_QA.csv", index=False)
    summary = qa.groupby("rule").size().reset_index(name="scenario_count")
    md = [
        "# Shuffled-Future Control QA",
        "",
        f"- seed: `{SHUFFLE_TRAIN_SEED}`",
        f"- scenarios: `{len(qa)}`",
        f"- self-match frequency: `{qa['self_match'].mean():.8f}`",
        f"- city-only fallbacks: `{int((qa['rule'] == 'city_only').sum())}`",
        f"- deterministic QA hash: `{sha256_file(EXEC / 'SHUFFLE_CONTROL_QA.csv')}`",
        "",
        "## Frozen Bin Edges",
        "",
        edges.to_markdown(index=False),
        "",
        "## Donor Rule Counts",
        "",
        summary.to_markdown(index=False),
        "",
        "## Donor Displacement Difference",
        "",
        qa["donor_minus_observed_disp_109"].describe().to_frame().to_markdown(),
        "",
    ]
    (EXEC / "SHUFFLE_CONTROL_QA.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def fit_evaluate_hgb_unit(
    df: pd.DataFrame,
    state: str,
    candidate_index: int,
    fold: int,
    candidate_limit: int = 24,
) -> dict:
    candidates = config_candidates(candidate_limit)
    if state not in STATE_ORDER:
        raise ValueError(f"Invalid HGB state: {state}")
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise ValueError(f"candidate_index must be in [0, {len(candidates) - 1}], got {candidate_index}")
    if fold < 0 or fold > 4:
        raise ValueError(f"fold must be in [0, 4], got {fold}")

    cfg = candidates[candidate_index]
    Xdf = df[feature_names_for_state(state)].astype(np.float32)
    X = Xdf.to_numpy(np.float32)
    y = df[TARGET_COLUMNS].to_numpy(np.float32)
    folds = df["fold"].to_numpy()
    tr = folds != fold
    te = folds == fold
    pred = np.zeros((int(te.sum()), 2), dtype=np.float32)
    losses = {}
    estimators = []
    try:
        for j, target in enumerate(TARGET_COLUMNS):
            model = HistGradientBoostingRegressor(
                random_state=TUNING_SEED + candidate_index * 10 + fold + j,
                **cfg,
            )
            model.fit(X[tr], y[tr, j])
            pred[:, j] = model.predict(X[te])
            losses[f"{target}_mse"] = mean_squared_error(y[te, j], pred[:, j])
            estimators.append(model)
        fde = np.linalg.norm(pred - y[te], axis=1)
        return {
            "state": state,
            "candidate_index": candidate_index,
            "fold": fold,
            **cfg,
            **losses,
            "mean_endpoint_mse": losses["dx_6s_mse"] + losses["dy_6s_mse"],
            "mean_fde": float(np.mean(fde)),
            "median_fde": float(np.median(fde)),
            "n_train_rows": int(tr.sum()),
            "n_dev_rows": int(te.sum()),
        }
    finally:
        del estimators
        del Xdf
        del X
        del y
        del folds
        del tr
        del te
        del pred
        gc.collect()


def run_single_hgb_unit(args: argparse.Namespace) -> None:
    ensure_dirs()
    start = time.time()
    started_at = utc_timestamp()
    state = args.state
    candidate_index = int(args.candidate_index)
    fold = int(args.fold)
    key = (state, candidate_index, fold)
    out_path = EXEC / "PRIMARY_CV_RESULTS_v2.csv"
    existing_count = count_hgb_key(out_path, key)
    print(
        f"SINGLE_UNIT start timestamp={started_at} state={state} candidate_index={candidate_index} "
        f"fold={fold} PID={os.getpid()}",
        flush=True,
    )
    if existing_count == 1:
        print(
            f"SINGLE_UNIT SKIP state={state} candidate_index={candidate_index} fold={fold} "
            f"PID={os.getpid()} reason=already_complete",
            flush=True,
        )
        return
    if existing_count > 1:
        raise RuntimeError(f"Duplicate completed key exists before fitting: {key}; count={existing_count}")

    feat = build_features(force=args.force_features)
    row = fit_evaluate_hgb_unit(
        feat["target_rows"],
        state=state,
        candidate_index=candidate_index,
        fold=fold,
        candidate_limit=args.hgb_candidates,
    )
    row_hash = append_primary_cv_row_atomic(out_path, row)
    elapsed = time.time() - start
    print(
        f"SINGLE_UNIT success state={state} candidate_index={candidate_index} fold={fold} "
        f"PID={os.getpid()} start_timestamp={started_at} elapsed_seconds={elapsed:.3f} "
        f"mean_endpoint_mse={row['mean_endpoint_mse']:.10f} mean_endpoint_fde={row['mean_fde']:.10f} "
        f"row_hash={row_hash} exit_success=true",
        flush=True,
    )
    del feat
    del row
    gc.collect()


def run_hgb_cv_in_process(df: pd.DataFrame, candidate_limit: int = 24, force: bool = False, resume: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    out_path = EXEC / "PRIMARY_CV_RESULTS_v2.csv"
    expected = hgb_expected_units(candidate_limit)
    if force:
        results = []
        completed: set[tuple[str, int, int]] = set()
    else:
        existing = read_hgb_results(out_path, candidate_limit)
        results = existing.to_dict(orient="records") if not existing.empty else []
        completed = {
            (r["state"], int(r["candidate_index"]), int(r["fold"]))
            for r in results
        }
        if completed == expected:
            res = pd.DataFrame(results).sort_values(HGB_UNIT_COLUMNS).reset_index(drop=True)
            write_primary_selection(res, select_configs(res), matched_config(res, candidate_limit), candidate_limit)
            return res, select_configs(res), matched_config(res, candidate_limit)
        if completed and not resume:
            raise RuntimeError(
                f"{out_path} contains {len(completed)} completed HGB units and {len(expected - completed)} remain. "
                "Rerun with --resume to continue or --force-hgb to restart."
            )
    start = time.time()
    for state in STATE_ORDER:
        for ci, _cfg in enumerate(config_candidates(candidate_limit)):
            for fold in range(5):
                unit = (state, ci, fold)
                if unit in completed:
                    print(f"HGB {state} candidate {ci+1}/{candidate_limit} fold {fold}: skip completed", flush=True)
                    continue
                row = fit_evaluate_hgb_unit(df, state, ci, fold, candidate_limit)
                results.append(row)
                completed.add(unit)
                current = pd.DataFrame(results).drop_duplicates(HGB_UNIT_COLUMNS, keep="last").sort_values(HGB_UNIT_COLUMNS)
                write_csv_safely(current, out_path)
                print(f"HGB {state} candidate {ci+1}/{candidate_limit} fold {fold}: mse={row['mean_endpoint_mse']:.5f} fde={row['mean_fde']:.5f} elapsed={time.time()-start:.1f}s", flush=True)
    res = pd.DataFrame(results).drop_duplicates(HGB_UNIT_COLUMNS, keep="last").sort_values(HGB_UNIT_COLUMNS).reset_index(drop=True)
    write_csv_safely(res, out_path)
    configs = select_configs(res)
    matched = matched_config(res, candidate_limit)
    write_primary_selection(res, configs, matched, candidate_limit)
    return res, configs, matched


def latest_row_hash(path: Path, key: tuple[str, int, int]) -> str:
    df = read_primary_cv_raw(path)
    rows = df[(df["state"] == key[0]) & (df["candidate_index"].astype(int) == key[1]) & (df["fold"].astype(int) == key[2])]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row for {key}, found {len(rows)}")
    row_dict = rows.iloc[0].to_dict()
    payload = json.dumps(
        row_dict,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_controller_log(record: dict) -> None:
    path = LOGS / "phase3b_subprocess_controller.log"
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_hgb_cv_subprocess_controller(args: argparse.Namespace) -> pd.DataFrame:
    out_path = EXEC / "PRIMARY_CV_RESULTS_v2.csv"
    expected = hgb_expected_units(args.hgb_candidates)
    if args.force_hgb:
        if out_path.exists():
            out_path.unlink()
        completed = set()
    else:
        completed = hgb_completed_keys_strict(out_path, args.hgb_candidates)
        if completed and not args.resume and completed != expected:
            raise RuntimeError(
                f"{out_path} contains {len(completed)} completed HGB units and {len(expected - completed)} remain. "
                "Rerun with --resume to continue or --force-hgb to restart."
            )

    parent_pid = os.getpid()
    new_units = 0
    for state, candidate_index, fold in hgb_units_in_order(args.hgb_candidates):
        key = (state, candidate_index, fold)
        before_completed = hgb_completed_keys_strict(out_path, args.hgb_candidates)
        rows_before = len(before_completed)
        if key in before_completed:
            continue
        if args.max_new_units is not None and new_units >= args.max_new_units:
            break

        started = time.time()
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-unit",
            "--state",
            state,
            "--candidate-index",
            str(candidate_index),
            "--fold",
            str(fold),
            "--hgb-candidates",
            str(args.hgb_candidates),
        ]
        if args.force_features:
            cmd.append("--force-features")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate()
        elapsed = time.time() - started
        rows_after_df = read_primary_cv_raw(out_path)
        rows_after = 0 if rows_after_df.empty else len(rows_after_df.drop_duplicates(HGB_UNIT_COLUMNS))
        key_count = count_hgb_key(out_path, key)
        status = "success"
        row_hash = None
        stdout_lines = stdout.strip().splitlines()
        stderr_lines = stderr.strip().splitlines()
        success_marker = (
            f"SINGLE_UNIT success state={state} candidate_index={candidate_index} fold={fold}"
        )
        if proc.returncode != 0:
            status = "nonzero_exit"
        elif not any(success_marker in line and "exit_success=true" in line for line in stdout_lines):
            status = "malformed_child_output"
        elif key_count != 1:
            status = "missing_or_duplicate_expected_key"
        elif rows_after != rows_before + 1:
            status = "unexpected_completed_count_delta"
        else:
            row_hash = latest_row_hash(out_path, key)
            new_units += 1

        record = {
            "timestamp": utc_timestamp(),
            "parent_pid": parent_pid,
            "state": state,
            "candidate_index": candidate_index,
            "fold": fold,
            "child_pid": proc.pid,
            "child_exit_code": proc.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "completed_rows_before": rows_before,
            "completed_rows_after": rows_after,
            "status": status,
            "expected_key": {"state": state, "candidate_index": candidate_index, "fold": fold},
            "result_row_hash": row_hash,
            "child_stdout_tail": stdout_lines[-3:],
            "child_stderr_tail": stderr_lines[-3:],
        }
        append_controller_log(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if status != "success":
            raise RuntimeError(f"HGB subprocess controller stopped on {key}: {status}")

    final_completed = hgb_completed_keys_strict(out_path, args.hgb_candidates)
    res = read_primary_cv_raw(out_path)
    if final_completed == expected:
        res = res.sort_values(HGB_UNIT_COLUMNS).reset_index(drop=True)
        write_csv_safely(res, out_path)
        write_primary_selection(res, select_configs(res), matched_config(res, args.hgb_candidates), args.hgb_candidates)
    return res


def run_hgb_cv(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict | None, pd.DataFrame | None]:
    if args.resume or args.force_hgb or args.max_new_units is not None:
        res = run_hgb_cv_subprocess_controller(args)
        completed = hgb_completed_keys_strict(EXEC / "PRIMARY_CV_RESULTS_v2.csv", args.hgb_candidates)
        if completed != hgb_expected_units(args.hgb_candidates):
            return res, None, None
        configs = select_configs(res)
        matched = matched_config(res, args.hgb_candidates)
        return res, configs, matched
    res, configs, matched = run_hgb_cv_in_process(
        df,
        candidate_limit=args.hgb_candidates,
        force=args.force_hgb,
        resume=args.resume,
    )
    return res, configs, matched


def select_configs(res: pd.DataFrame) -> dict:
    selected = {}
    for state in STATE_ORDER:
        state_res = res[res.state == state]
        complete_candidates = state_res.groupby("candidate_index")["fold"].nunique()
        complete_candidates = complete_candidates[complete_candidates == 5].index
        if len(complete_candidates) == 0:
            raise RuntimeError(f"No complete 5-fold HGB candidate is available for state {state}. Resume CV before selection.")
        state_res = state_res[state_res.candidate_index.isin(complete_candidates)]
        g = state_res.groupby("candidate_index", as_index=False)["mean_endpoint_mse"].mean().sort_values(["mean_endpoint_mse", "candidate_index"]).iloc[0]
        row = state_res[state_res.candidate_index == int(g.candidate_index)].iloc[0]
        selected[state] = {k: row[k].item() if hasattr(row[k], "item") else row[k] for k in p3a.HGB_SPACE}
        selected[state]["candidate_index"] = int(g.candidate_index)
        selected[state]["mean_cv_endpoint_mse"] = float(g.mean_endpoint_mse)
    return selected


def matched_config(res: pd.DataFrame, candidate_limit: int = 24) -> pd.DataFrame:
    out_path = EXEC / "MATCHED_HC_CANDIDATE_RESULTS_v2.csv"
    expected_pairs = {(ci, state) for ci in range(candidate_limit) for state in ["H", "C"]}
    complete_pairs = set()
    for (state, ci), g in res[res.state.isin(["H", "C"])].groupby(["state", "candidate_index"]):
        if g["fold"].nunique() == 5:
            complete_pairs.add((int(ci), state))
    if complete_pairs != expected_pairs:
        missing = len(expected_pairs - complete_pairs)
        if out_path.exists():
            return pd.read_csv(out_path)
        raise RuntimeError(f"Matched H/C selection requires complete H and C CV. Missing candidate-state pairs: {missing}.")
    hc = res[res.state.isin(["H", "C"])].copy()
    state_means = hc.groupby("state")["mean_endpoint_mse"].agg(["mean", "std"]).to_dict("index")
    rows = []
    for ci, g in hc.groupby("candidate_index"):
        vals = []
        for state, sg in g.groupby("state"):
            vals.append((sg["mean_endpoint_mse"].mean() - state_means[state]["mean"]) / state_means[state]["std"])
        rows.append({"candidate_index": int(ci), "mean_standardized_HC_loss": float(np.mean(vals))})
    out = pd.DataFrame(rows).sort_values(["mean_standardized_HC_loss", "candidate_index"])
    write_csv_safely(out, out_path)
    return out


def write_primary_selection(res: pd.DataFrame, configs: dict, matched: pd.DataFrame, candidate_limit: int = 24) -> None:
    payload = {
        "selection_rule": "empirical TRAIN-only grouped-CV minimum mean endpoint MSE",
        "randomized_search_seed": TUNING_SEED,
        "candidate_count_per_state": int(res["candidate_index"].nunique()),
        "selected_configs": configs,
        "MATCHED_HC_CONFIG_FINAL": {
            **config_candidates(candidate_limit)[int(matched.iloc[0].candidate_index)],
            "candidate_index": int(matched.iloc[0].candidate_index),
            "mean_standardized_HC_loss": float(matched.iloc[0].mean_standardized_HC_loss),
        },
    }
    (EXEC / "PRIMARY_MODEL_CONFIGS_v2.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    summary = res.groupby(["state", "candidate_index"], as_index=False).agg(mean_endpoint_mse=("mean_endpoint_mse", "mean"), mean_fde=("mean_fde", "mean"))
    best_table = summary.sort_values(["state", "mean_endpoint_mse", "candidate_index"]).groupby("state").head(1)
    md = [
        "# Primary Model Selection v2",
        "",
        "Empirical winners selected using TRAIN-only grouped city-stratified folds and predictive endpoint loss only.",
        "",
        "## Selected State-Specific Configs",
        "",
        best_table.to_markdown(index=False),
        "",
        "## Matched H/C Config",
        "",
        pd.DataFrame([payload["MATCHED_HC_CONFIG_FINAL"]]).to_markdown(index=False),
        "",
        f"- CV results hash: `{sha256_file(EXEC / 'PRIMARY_CV_RESULTS_v2.csv')}`",
        "",
    ]
    (EXEC / "PRIMARY_MODEL_SELECTION_v2.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_adequacy(df: pd.DataFrame, h_config: dict, force: bool = False, resume: bool = False) -> dict:
    out = EXEC / "TRAIN_MODEL_ADEQUACY.csv"
    expected_folds = set(range(5))
    if out.exists() and not force:
        a = pd.read_csv(out)
        if not a.empty:
            a = a.drop_duplicates(["fold"], keep="last").copy()
            write_csv_safely(a, out)
    else:
        a = pd.DataFrame()
    completed_folds = set(a["fold"].astype(int).tolist()) if not a.empty else set()
    if completed_folds != expected_folds:
        if completed_folds and not resume and not force:
            raise RuntimeError(
                f"{out} contains {len(completed_folds)} completed adequacy folds and "
                f"{len(expected_folds - completed_folds)} remain. Rerun with --resume."
            )
        X = df[feature_names_for_state("H")].astype(np.float32).to_numpy()
        y = df[TARGET_COLUMNS].to_numpy(np.float32)
        cv_pred = df[["cv_dx_6s", "cv_dy_6s"]].to_numpy(np.float32)
        folds = df.fold.to_numpy()
        cfg = {k: h_config[k] for k in p3a.HGB_SPACE}
        rows = a.to_dict(orient="records") if not a.empty else []
        for fold in range(5):
            if fold in completed_folds:
                continue
            tr, te = folds != fold, folds == fold
            pred = np.zeros((te.sum(), 2), dtype=np.float32)
            for j in range(2):
                model = HistGradientBoostingRegressor(random_state=TUNING_SEED + 900 + fold + j, **cfg)
                model.fit(X[tr], y[tr, j])
                pred[:, j] = model.predict(X[te])
            fde_h = np.linalg.norm(pred - y[te], axis=1)
            fde_cv = np.linalg.norm(cv_pred[te] - y[te], axis=1)
            rows.append({"fold": fold, "hgb_h_mean_fde": float(fde_h.mean()), "hgb_h_median_fde": float(np.median(fde_h)), "constant_velocity_mean_fde": float(fde_cv.mean()), "constant_velocity_median_fde": float(np.median(fde_cv)), "paired_mean_fde_reduction": float((fde_cv - fde_h).mean()), "n_rows": int(te.sum())})
            a = pd.DataFrame(rows).drop_duplicates(["fold"], keep="last").sort_values("fold")
            write_csv_safely(a, out)
        a = pd.DataFrame(rows).drop_duplicates(["fold"], keep="last").sort_values("fold")
        write_csv_safely(a, out)
    ok = bool(a["paired_mean_fde_reduction"].mean() > 0.1)
    md = [
        "# TRAIN Model Adequacy",
        "",
        "Descriptive TRAIN-development adequacy check; no official VAL is read.",
        "",
        a.to_markdown(index=False),
        "",
        f"- mean paired FDE reduction: `{a['paired_mean_fde_reduction'].mean():.6f}`",
        f"- HGB-H materially outperforms constant velocity: `{str(ok).lower()}`",
        "",
    ]
    (EXEC / "TRAIN_MODEL_ADEQUACY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"ok": ok, "mean_reduction": float(a["paired_mean_fde_reduction"].mean())}


def write_compute_decision(df: pd.DataFrame) -> dict:
    out = EXEC / "FINAL_FIT_COMPUTE_DECISION.md"
    if out.exists():
        return {"artifact": str(out), "preserved_existing": True}
    rows_50k = len(df)
    eligible = pd.read_csv(AUDIT / "TRAIN_ELIGIBILITY_FINAL.csv")
    eligible_scenarios = int(eligible.loc[eligible.key == "scenarios_with_ge_2_eligible_primary_targets", "value"].iloc[0])
    eligible_rows = int(eligible.loc[eligible.key == "eligible_target_rows", "value"].iloc[0])
    decision = {
        "preferred_strategy": "50k stratified sample for hyperparameter selection; all eligible TRAIN for final H/C fitting",
        "full_train_final_fit_feasible": True,
        "reason": "HistGradientBoostingRegressor final fit is two fits per state, not the full CV budget; observed feature-matrix row count scales from the measured 50k target-row build.",
        "development_rows_50k_sample": rows_50k,
        "full_train_primary_eligible_scenarios": eligible_scenarios,
        "full_train_primary_eligible_target_rows": eligible_rows,
        "final_fit_population": "all 145390 eligible TRAIN scenarios / all 630388 eligible target rows for H and C; E and C_SHUFFLED also full TRAIN where feasible",
        "predictive_performance_used": False,
    }
    out.write_text("# Final Fit Compute Decision\n\n" + yaml.safe_dump(decision, sort_keys=False), encoding="utf-8")
    return decision


def write_neural_placeholder(df: pd.DataFrame) -> dict:
    # The architecture is frozen, but this phase runner keeps neural execution
    # separate because 12 configs x 3 folds x 3 seeds is a GPU-scale job.
    config_path = EXEC / "NEURAL_MODEL_CONFIGS_v2.yaml"
    cv_path = EXEC / "NEURAL_CV_RESULTS_v2.csv"
    stats_path = EXEC / "NEURAL_NORMALIZATION_STATS_v2.yaml"
    selection_path = EXEC / "NEURAL_MODEL_SELECTION_v2.md"
    if config_path.exists() and cv_path.exists() and stats_path.exists() and selection_path.exists():
        existing = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        existing["preserved_existing"] = True
        return existing
    configs = {
        "status": "not_executed_by_phase3b_hgb_runner",
        "reason": "No neural training implementation existed in the repository before this phase; official VAL remains untouched. Final freeze marks neural selection blocked rather than fabricating TRAIN losses.",
        "required_before_official_val": True,
        "target_rows_available": len(df),
        "normalization_statistics_reference": "execution/NEURAL_NORMALIZATION_STATS_v2.yaml",
        "resume_idempotent": True,
    }
    numeric = df.select_dtypes(include=[np.number])
    stats = {c: {"mean": float(numeric[c].mean()), "std": float(numeric[c].std(ddof=0))} for c in numeric.columns if c not in ["fold", "is_nearest"]}
    stats_path.write_text(yaml.safe_dump(stats, sort_keys=True), encoding="utf-8")
    write_csv_safely(pd.DataFrame(columns=["state", "candidate_index", "fold", "seed", "epoch", "dev_loss", "status"]), cv_path)
    config_path.write_text(yaml.safe_dump(configs, sort_keys=False), encoding="utf-8")
    selection_path.write_text(
        "# Neural Model Selection v2\n\nNeural TRAIN development was not executed by this HGB-focused phase runner. This is a hard blocker before official VAL authorization; no neural result has been fabricated or evaluated on VAL.\n",
        encoding="utf-8",
    )
    return configs


def write_final_freeze(values: dict) -> str:
    payload = {
        "execution_freeze_id": "AV2_FINAL_EXECUTION_FREEZE_v2",
        "date_local": "2026-09-07",
        "timezone": "Asia/Shanghai",
        "official_val_outcome_blind": True,
        "scientific_protocol_v1_1_hash": PROTOCOL_HASH,
        "provisional_execution_hash": PROVISIONAL_EXECUTION_HASH,
        **values,
        "software_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
        "exact_commands_for_final_execution": [
            "python tools/phase3b_train_only_execution.py --run-all",
            "pytest tests/test_coordinate_transform.py tests/test_information_leakage.py tests/test_role_assignment.py tests/test_split_integrity.py",
        ],
    }
    yaml_path = EXEC / "AV2_FINAL_EXECUTION_FREEZE_v2.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    md = ["# AV2 Final Execution Freeze v2", "", "Official VAL remains entirely outcome-blind.", "", "```yaml", yaml.safe_dump(payload, sort_keys=False), "```", ""]
    (EXEC / "AV2_FINAL_EXECUTION_FREEZE_v2.md").write_text("\n".join(md), encoding="utf-8")
    digest = sha256_file(yaml_path)
    (EXEC / "AV2_FINAL_EXECUTION_HASH_v2.txt").write_text(f"AV2_FINAL_EXECUTION_FREEZE_v2_SHA256={digest}\n", encoding="utf-8")
    return digest


def run_all(args: argparse.Namespace) -> None:
    ensure_dirs()
    sample = write_sample_artifacts()
    decision_hash = write_confirmatory_rule()
    feat = build_features(force=args.force_features)
    df = feat["target_rows"]
    cv, configs, matched = run_hgb_cv(df, args)
    if configs is None or matched is None:
        status = hgb_resume_status(args.hgb_candidates)
        print(
            json.dumps(
                {
                    "hgb_cv_complete": False,
                    "downstream_steps_skipped": True,
                    "reason": "HGB CV is incomplete; resume will continue from PRIMARY_CV_RESULTS_v2.csv",
                    **status,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    adequacy = write_adequacy(df, configs["H"], force=args.force_hgb, resume=args.resume)
    compute = write_compute_decision(df)
    neural = write_neural_placeholder(df)
    values = {
        "corrected_50k_city_stratified_sample_hash": sample["sample_hash"],
        "confirmatory_decision_rule_hash": decision_hash,
        "actual_feature_schema_hash": sha256_file(EXEC / "FEATURE_SCHEMA_v1.1.csv"),
        "actual_train_feature_matrix_hash": feat["hash"],
        "map_lite_missingness_hash": sha256_file(AUDIT / "MAPLITE_FEATURE_MISSINGNESS.csv"),
        "hgb_cv_results_hash": sha256_file(EXEC / "PRIMARY_CV_RESULTS_v2.csv"),
        "empirically_selected_hgb_configs": configs,
        "matched_hc_config": yaml.safe_load((EXEC / "PRIMARY_MODEL_CONFIGS_v2.yaml").read_text(encoding="utf-8"))["MATCHED_HC_CONFIG_FINAL"],
        "constant_velocity_adequacy": adequacy,
        "final_fit_compute_decision": compute,
        "neural_model_selection": neural,
        "neural_cv_results_hash": sha256_file(EXEC / "NEURAL_CV_RESULTS_v2.csv"),
        "normalization_statistics_hash": sha256_file(EXEC / "NEURAL_NORMALIZATION_STATS_v2.yaml"),
        "shuffle_control_qa_hash": sha256_file(EXEC / "SHUFFLE_CONTROL_QA.csv"),
        "seeds": {"split": SPLIT_SEED, "tuning": TUNING_SEED, "shuffle_train": SHUFFLE_TRAIN_SEED},
    }
    freeze_hash = write_final_freeze(values)
    print(json.dumps({"final_freeze_hash_v2": freeze_hash, "official_val_outcome_blind": True}, indent=2))


def dry_check(args: argparse.Namespace) -> None:
    ensure_dirs()
    status = hgb_resume_status(args.hgb_candidates)
    print(json.dumps(status, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--single-unit", action="store_true", help="fit/evaluate exactly one TRAIN-only HGB candidate-fold unit")
    parser.add_argument("--state", choices=STATE_ORDER)
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--resume", action="store_true", help="resume completed TRAIN-only units without duplicating prior rows")
    parser.add_argument("--dry-check", action="store_true", help="report TRAIN-only resume status without fitting models")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-hgb", action="store_true")
    parser.add_argument("--hgb-candidates", type=int, default=24)
    parser.add_argument("--max-new-units", type=int, help="execution-only controller throttle for smoke testing")
    args = parser.parse_args()
    if args.single_unit:
        missing = [name for name in ["state", "candidate_index", "fold"] if getattr(args, name) is None]
        if missing:
            parser.error(f"--single-unit requires: {', '.join('--' + m.replace('_', '-') for m in missing)}")
        run_single_hgb_unit(args)
    elif args.dry_check:
        dry_check(args)
    elif args.run_all:
        run_all(args)
    else:
        ensure_dirs()
        write_sample_artifacts()
        print("Wrote corrected TRAIN 50K stratified sample artifacts.")


if __name__ == "__main__":
    main()
