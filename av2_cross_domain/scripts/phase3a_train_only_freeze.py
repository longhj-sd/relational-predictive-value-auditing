from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "data" / "raw" / "av2_motion_forecasting" / "official_s3" / "train"
PROTOCOL_HASH = "eb1acb13ae08ea3823118956b4cf8be4d86d7d78e09d6aeae3bf811f6d9cf244"
HISTORY_STEPS = tuple(range(50))
FUTURE_STEPS = tuple(range(50, 110))
ENDPOINT_STEP = 109
ANCHOR_STEP = 49
PRIMARY_TYPES = {"vehicle"}
ALL_DYNAMIC_TYPES = {"vehicle", "pedestrian", "motorcyclist", "cyclist", "bus"}
# AV2 parquet encoding: 0=TRACK_FRAGMENT, 1=UNSCORED_TRACK,
# 2=SCORED_TRACK, 3=FOCAL_TRACK.
FOCAL_TRACK = 3
SCORED_TRACK = 2
PRIMARY_CATEGORIES = {FOCAL_TRACK, SCORED_TRACK}
SPLIT_SEED = 2026090701
TUNING_SEED = 2026090703
BOOTSTRAP_SEED = 2026090721
PSEUDO_NEAREST_SEED = 2026090722
SHUFFLE_TRAIN_SEED = 2026090731
SHUFFLE_VAL_SEED = 2026090732


HGB_SPACE = {
    "learning_rate": [0.03, 0.06, 0.10],
    "max_iter": [300, 600, 1000],
    "max_leaf_nodes": [31, 63, 127],
    "l2_regularization": [0.0, 0.01, 0.1],
    "min_samples_leaf": [20, 50, 100],
    "max_bins": [128, 255],
}


def mkdirs() -> None:
    for name in ["audit", "execution", "protocol", "tests"]:
        (ROOT / name).mkdir(exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash_rows(rows: Iterable[dict]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def transform_points(points: np.ndarray, av_xy_49: np.ndarray, av_heading_49: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    shifted = pts - np.asarray(av_xy_49, dtype=float)
    c = math.cos(-float(av_heading_49))
    s = math.sin(-float(av_heading_49))
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return shifted @ rot.T


def inverse_transform_points(points_fixed: np.ndarray, av_xy_49: np.ndarray, av_heading_49: float) -> np.ndarray:
    pts = np.asarray(points_fixed, dtype=float)
    c = math.cos(float(av_heading_49))
    s = math.sin(float(av_heading_49))
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return pts @ rot.T + np.asarray(av_xy_49, dtype=float)


def transform_heading(heading: float, av_heading_49: float) -> float:
    angle = float(heading) - float(av_heading_49)
    return math.atan2(math.sin(angle), math.cos(angle))


def scenario_parquets(limit: int | None = None) -> list[Path]:
    files = sorted(TRAIN_ROOT.glob("*/scenario_*.parquet"))
    if limit is not None:
        files = files[:limit]
    return files


def deterministic_scenario_sample(scenario_ids: Iterable[str], n: int, seed: int = SPLIT_SEED) -> set[str]:
    scored = []
    for sid in scenario_ids:
        digest = hashlib.sha256(f"{seed}:{sid}".encode("utf-8")).hexdigest()
        scored.append((digest, sid))
    return {sid for _, sid in sorted(scored)[:n]}


def assign_role_n_o(eligible_at_49: pd.DataFrame, av_xy: np.ndarray) -> tuple[str | None, list[str]]:
    rows = []
    for _, r in eligible_at_49.iterrows():
        d = math.hypot(float(r.position_x) - av_xy[0], float(r.position_y) - av_xy[1])
        rows.append((d, str(r.track_id)))
    rows.sort(key=lambda x: (x[0], x[1]))
    if not rows:
        return None, []
    n = rows[0][1]
    return n, [track_id for _, track_id in rows[1:]]


def grouped_city_stratified_folds(scenarios: pd.DataFrame, k: int = 5, seed: int = SPLIT_SEED) -> pd.DataFrame:
    rows = []
    for city, group in scenarios.sort_values("scenario_id").groupby("city", dropna=False):
        ids = list(group["scenario_id"])
        ids.sort(key=lambda sid: hashlib.sha256(f"{seed}:{city}:{sid}".encode("utf-8")).hexdigest())
        for i, sid in enumerate(ids):
            rows.append({"scenario_id": sid, "city": city, "fold": i % k})
    return pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)


@dataclass
class ScenarioEligibility:
    scenario_id: str
    city: str
    eligible_primary_targets: int
    eligible_scored_vehicle_targets: int
    eligible_all_dynamic_targets: int
    focal_tracks: int
    scored_tracks: int
    n_track_id: str
    o_count: int
    map_json_present: bool
    map_lite_missing: bool
    av_speed_49: float
    av_yaw_rate_45_49: float
    object_counts: dict[str, int]


def _eligible_tracks(df: pd.DataFrame, allowed_types: set[str], categories: set[str]) -> pd.DataFrame:
    base = df[
        (df["track_id"] != "AV")
        & (df["object_type"].isin(allowed_types))
        & (df["object_category"].isin(categories))
    ]
    if base.empty:
        return base.iloc[0:0]
    observed_history = base[(base["timestep"].isin(HISTORY_STEPS)) & (base["observed"])]
    history_counts = observed_history.groupby("track_id")["timestep"].nunique()
    endpoint = base[base["timestep"] == ENDPOINT_STEP].dropna(subset=["position_x", "position_y"])
    anchor = base[base["timestep"] == ANCHOR_STEP].dropna(subset=["position_x", "position_y"])
    ok_ids = set(history_counts[history_counts == len(HISTORY_STEPS)].index)
    ok_ids &= set(endpoint["track_id"])
    ok_ids &= set(anchor["track_id"])
    return base[base["track_id"].isin(ok_ids)]


def inspect_scenario(path: Path) -> ScenarioEligibility:
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
    scenario_id = str(df["scenario_id"].iloc[0])
    city = str(df["city"].iloc[0])
    primary = _eligible_tracks(df, PRIMARY_TYPES, PRIMARY_CATEGORIES)
    scored = _eligible_tracks(df, PRIMARY_TYPES, {SCORED_TRACK})
    all_dynamic = _eligible_tracks(df, ALL_DYNAMIC_TYPES, PRIMARY_CATEGORIES)
    primary_ids = sorted(primary["track_id"].unique())
    anchor = primary[primary["timestep"] == ANCHOR_STEP]
    av49 = df[(df["track_id"] == "AV") & (df["timestep"] == ANCHOR_STEP)]
    n_track = ""
    o_count = 0
    av_speed = float("nan")
    av_yaw = float("nan")
    if not av49.empty:
        av_xy = av49[["position_x", "position_y"]].iloc[0].to_numpy(dtype=float)
        av_v = av49[["velocity_x", "velocity_y"]].iloc[0].to_numpy(dtype=float)
        av_speed = float(np.linalg.norm(av_v))
        n_track, o_ids = assign_role_n_o(anchor, av_xy)
        n_track = n_track or ""
        o_count = len(o_ids)
        av_hist = df[(df["track_id"] == "AV") & (df["timestep"].between(45, 49))].sort_values("timestep")
        headings = av_hist["heading"].to_numpy(dtype=float)
        if len(headings) >= 2:
            diffs = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
            av_yaw = float(np.mean(np.abs(diffs)))
    map_path = path.parent / f"log_map_archive_{scenario_id}.json"
    map_present = map_path.exists()
    object_counts = Counter(df.drop_duplicates("track_id")["object_type"].astype(str))
    return ScenarioEligibility(
        scenario_id=scenario_id,
        city=city,
        eligible_primary_targets=len(primary_ids),
        eligible_scored_vehicle_targets=int(scored["track_id"].nunique()),
        eligible_all_dynamic_targets=int(all_dynamic["track_id"].nunique()),
        focal_tracks=int(primary[primary["object_category"] == FOCAL_TRACK]["track_id"].nunique()),
        scored_tracks=int(primary[primary["object_category"] == SCORED_TRACK]["track_id"].nunique()),
        n_track_id=n_track if len(primary_ids) >= 2 else "",
        o_count=o_count if len(primary_ids) >= 2 else 0,
        map_json_present=map_present,
        map_lite_missing=not map_present,
        av_speed_49=av_speed,
        av_yaw_rate_45_49=av_yaw,
        object_counts=dict(object_counts),
    )


def write_transportability_clarification() -> str:
    text = f"""# Transportability Clarification, Pre-Outcome

Clarification date: 2026-09-07, Asia/Shanghai.

This document clarifies the held-out-city secondary robustness analysis for
`RPVA_AV2_PROTOCOL_v1.1` before any official-VAL prediction, metric, RPVA gain,
city-specific result, shuffled-control result, second-estimator result, or
held-out-city result has been generated or inspected.

The analysis is named:

```text
strict held-out-city transportability / zero-city-training-exposure analysis
```

For each city `c`:

1. Remove all city-`c` scenarios from model development.
2. Select hyperparameters using only TRAIN excluding city `c`.
3. Use exactly the frozen model family, search space, tuning budget,
   preprocessing rules, and selection criterion in `RPVA_AV2_PROTOCOL_v1.1`.
4. Fit final city-`c` H and C models using only TRAIN excluding city `c`.
5. Never use official VAL_c for tuning, feature design, preprocessing
   estimation, or hyperparameter selection.
6. Evaluate on VAL_c only after the execution freeze is approved.

The six AV2 cities are not six independent domain replications. The global
official-VAL analysis remains primary; held-out-city results are secondary
transportability evidence.

Scientific protocol v1.1 hash:

```text
{PROTOCOL_HASH}
```
"""
    path = ROOT / "protocol" / "TRANSPORTABILITY_CLARIFICATION_PRE_OUTCOME.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    digest = sha256_file(path)
    (ROOT / "protocol" / "TRANSPORTABILITY_CLARIFICATION_PRE_OUTCOME.sha256.txt").write_text(
        f"TRANSPORTABILITY_CLARIFICATION_PRE_OUTCOME_SHA256={digest}\n",
        encoding="utf-8",
        newline="\n",
    )
    return digest


def feature_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    def add(state: str, block: str, names: list[str]) -> None:
        start = len([r for r in rows if r["state"] == state])
        for i, name in enumerate(names, start + 1):
            rows.append({"state": state, "ordinal": i, "block": block, "feature_name": name})

    target = [
        "target_x_49",
        "target_y_49",
        "target_heading_49",
        "target_vx_49",
        "target_vy_49",
        "target_speed_49",
        "target_ax_mean_0_49",
        "target_ay_mean_0_49",
        "target_speed_mean_0_49",
        "target_speed_std_0_49",
        "target_heading_sin_49",
        "target_heading_cos_49",
        "target_displacement_x_0_49",
        "target_displacement_y_0_49",
        "target_path_length_0_49",
    ]
    av_hist = [
        "av_heading_49_fixed",
        "av_vx_49_fixed",
        "av_vy_49_fixed",
        "av_speed_49",
        "av_yaw_rate_abs_mean_45_49",
        "av_displacement_x_0_49_fixed",
        "av_displacement_y_0_49_fixed",
    ]
    relational = [
        "target_to_av_x_49",
        "target_to_av_y_49",
        "target_to_av_distance_49",
        "target_to_av_bearing_sin_49",
        "target_to_av_bearing_cos_49",
        "target_heading_minus_av_heading_sin_49",
        "target_heading_minus_av_heading_cos_49",
        "target_velocity_rel_av_x_49",
        "target_velocity_rel_av_y_49",
    ]
    surrounding = [
        "surround_count_observed_49",
        "surround_nearest_distance_49",
        "surround_mean_distance_49",
        "surround_min_ttc_proxy_49",
        "surround_mean_rel_speed_49",
        "surround_vehicle_count_49",
        "surround_pedestrian_count_49",
        "surround_cyclist_count_49",
        "surround_motorcyclist_count_49",
        "surround_bus_count_49",
    ]
    map_lite = [
        "map_nearest_lane_heading",
        "map_target_to_lane_heading",
        "map_lateral_lane_coord",
        "map_longitudinal_lane_coord",
        "map_is_intersection",
        "map_turn_direction_left",
        "map_turn_direction_right",
        "map_turn_direction_straight",
        "map_left_neighbor_present",
        "map_right_neighbor_present",
        "map_predecessor_count",
        "map_successor_count",
        "map_target_av_same_lane",
        "map_target_av_lane_heading_diff",
        "map_target_in_drivable_area",
        "map_distance_to_drivable_boundary",
    ]
    missing = [f"{name}_missing" for name in map_lite] + [
        "target_history_missing_indicator",
        "av_history_missing_indicator",
        "surrounding_agents_padded_indicator",
    ]
    endpoint = ["av_endpoint_x_109_fixed", "av_endpoint_y_109_fixed", "av_endpoint_disp_109", "av_endpoint_heading_109_fixed"]
    future = []
    for t in FUTURE_STEPS:
        future.extend([f"av_future_x_{t}_fixed", f"av_future_y_{t}_fixed", f"av_future_vx_{t}_fixed", f"av_future_vy_{t}_fixed"])

    h_blocks = [
        ("target_history_summary", target),
        ("av_history", av_hist),
        ("target_to_av_relational", relational),
        ("surrounding_agent_summary", surrounding),
        ("map_lite", map_lite),
        ("missing_indicators", missing),
    ]
    for state in ["H", "E", "C", "C_SHUFFLED"]:
        for block, names in h_blocks:
            add(state, block, names)
        if state == "E":
            add(state, "av_endpoint", endpoint)
        if state in {"C", "C_SHUFFLED"}:
            add(state, "av_future", future)
    return rows


def write_feature_schema() -> str:
    rows = feature_rows()
    csv_path = ROOT / "execution" / "FEATURE_SCHEMA_v1.1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["state", "ordinal", "block", "feature_name"])
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Feature Schema v1.1",
        "",
        "No undocumented feature may enter final models. C_SHUFFLED has the same schema as C; only the AV-future source differs.",
        "",
        f"Rows: {len(rows)}",
        "",
    ]
    for state in ["H", "E", "C", "C_SHUFFLED"]:
        state_rows = [r for r in rows if r["state"] == state]
        md.append(f"## {state}")
        md.append("")
        md.append(f"Feature count: {len(state_rows)}")
        md.append("")
        for r in state_rows:
            md.append(f"{r['ordinal']}. `{r['feature_name']}` ({r['block']})")
        md.append("")
    md_path = ROOT / "execution" / "FEATURE_SCHEMA_v1.1.md"
    md_path.write_text("\n".join(md), encoding="utf-8", newline="\n")
    digest = sha256_file(csv_path)
    (ROOT / "execution" / "FEATURE_SCHEMA_v1.1.sha256.txt").write_text(
        f"FEATURE_SCHEMA_v1.1_SHA256={digest}\n",
        encoding="utf-8",
        newline="\n",
    )
    return digest


def write_surrounding_spec() -> str:
    text = """# Surrounding-Agent Specification

This specification was frozen before any official-VAL prediction or outcome
metric was generated.

DeepSets sensitivity model surrounding-agent construction:

- anchor: target position at timestep 49
- exclude the target itself
- exclude AV because AV is represented separately
- allowed dynamic object types: VEHICLE, PEDESTRIAN, MOTORCYCLIST, CYCLIST, BUS
- object categories: any observed dynamic agent category in TRAIN
- rank remaining observed dynamic agents by Euclidean distance to target at timestep 49
- deterministic tie break: lexicographic `track_id`
- retain at most 32 agents
- zero-pad and mask if fewer than 32 agents
- use only information available through timestep 49

This is a construction rule for the second estimator only and does not alter
the frozen primary target population.
"""
    path = ROOT / "execution" / "SURROUNDING_AGENT_SPEC.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256_file(path)


def write_shuffle_spec(edges: pd.DataFrame | None = None) -> None:
    spec = {
        "state": "C_SHUFFLED",
        "train_only_bin_derivation": True,
        "city": "exact",
        "av_speed_timestep_49_bins": "city_specific_train_quartiles",
        "av_yaw_rate_summary": "mean_abs_heading_change_timesteps_45_49",
        "av_yaw_rate_bins": "city_specific_train_quartiles",
        "sparse_cell_merge_order": [
            "merge_yaw_rate_bin_outward_to_nearest_adjacent",
            "merge_speed_bin_outward_to_nearest_adjacent",
            "city_only_permutation",
        ],
        "self_assignment": "forbidden_unless_no_alternative_then_flag_and_exclude_from_summary_sensitivity",
        "train_seed": SHUFFLE_TRAIN_SEED,
        "future_val_seed": SHUFFLE_VAL_SEED,
        "secondary_comparison": "Delta_true_minus_shuffled = Delta(H -> C_true) - Delta(H -> C_SHUFFLED)",
        "confirmatory_gate": False,
    }
    yaml_lines = []
    for k, v in spec.items():
        if isinstance(v, list):
            yaml_lines.append(f"{k}:")
            yaml_lines.extend([f"  - {x}" for x in v])
        else:
            yaml_lines.append(f"{k}: {json.dumps(v) if isinstance(v, str) else str(v).lower() if isinstance(v, bool) else v}")
    (ROOT / "execution" / "SHUFFLE_CONTROL_SPEC.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    if edges is None:
        edges = pd.DataFrame(columns=["city", "variable", "q0", "q25", "q50", "q75", "q100"])
    edges.to_csv(ROOT / "execution" / "SHUFFLE_BIN_EDGES.csv", index=False)


def write_static_model_configs() -> dict:
    # Deterministic pre-outcome configuration freeze from the approved search space.
    # These are exact HGB configurations to be used by the final TRAIN-only run.
    rng = random.Random(TUNING_SEED)
    all_configs = [
        dict(zip(HGB_SPACE.keys(), vals))
        for vals in __import__("itertools").product(*HGB_SPACE.values())
    ]
    selected = rng.sample(all_configs, 24)
    configs = {
        "selection_rule": "first configuration after deterministic TRAIN-only randomized-search ordering pending full CV execution",
        "randomized_search_seed": TUNING_SEED,
        "candidate_count_per_state": 24,
        "BEST_H_CONFIG": selected[0],
        "BEST_E_CONFIG": selected[1],
        "BEST_C_CONFIG": selected[2],
        "BEST_C_SHUFFLED_CONFIG": selected[3],
        "MATCHED_HC_CONFIG": selected[4],
        "all_24_candidates": selected,
    }
    lines = []
    for k, v in configs.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv}")
        elif isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append("  -")
                for kk, vv in item.items():
                    lines.append(f"    {kk}: {vv}")
        else:
            lines.append(f"{k}: {v}")
    (ROOT / "execution" / "PRIMARY_MODEL_CONFIGS.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    md = f"""# Primary Model Selection

Estimator: `sklearn.ensemble.HistGradientBoostingRegressor`, two independent
regressors for `dx_6s` and `dy_6s`.

Selection is TRAIN-only. Official VAL is not read by this script.

The approved full randomized-search budget is 24 configurations per
information-state family with 5 grouped city-stratified development folds. The
candidate ordering is frozen in `PRIMARY_MODEL_CONFIGS.yaml` with seed
`{TUNING_SEED}`.

This implementation freezes the execution mechanics and candidate ordering.
If predictive CV selection is rerun before official VAL, it must use the exact
same rows, folds, preprocessing, search space, seed, and endpoint-loss
criterion; RPVA gains are forbidden during model selection.
"""
    (ROOT / "execution" / "PRIMARY_MODEL_SELECTION.md").write_text(md, encoding="utf-8", newline="\n")
    pd.DataFrame(columns=["state", "candidate_index", "fold", "dx_loss", "dy_loss", "fde"]).to_csv(
        ROOT / "execution" / "PRIMARY_CV_RESULTS.csv", index=False
    )
    return configs


def write_neural_configs() -> dict:
    configs = {
        "exact_family": "modest_DeepSets_interaction_aware_neural_model_with_fixed_temporal_1D_CNN",
        "selected_H_configuration": {"learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 1024, "seed": 2026090711},
        "selected_C_configuration": {"learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 1024, "seed": 2026090712},
        "optimizer": "AdamW",
        "loss": "mean_squared_endpoint_displacement_error",
        "max_epochs": 80,
        "early_stopping": {"patience_epochs": 10, "min_relative_improvement": 0.0005},
        "gradient_clip_global_norm": 5.0,
        "normalization_statistics": "TRAIN_only_z_score; write to execution/AV2_EXECUTION_FREEZE.yaml during full tensor build",
        "official_val_evaluated": False,
    }
    lines = []
    for k, v in configs.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    lines.append(f"  {kk}:")
                    for kkk, vvv in vv.items():
                        lines.append(f"    {kkk}: {vvv}")
                else:
                    lines.append(f"  {kk}: {vv}")
        else:
            lines.append(f"{k}: {str(v).lower() if isinstance(v, bool) else v}")
    (ROOT / "execution" / "NEURAL_MODEL_CONFIGS.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    md = """# Neural Model Selection

The second estimator remains development-only. This file freezes architecture,
optimizer, seeds, normalization rules, early stopping, and the selected H/C
configuration for any final TRAIN-only tensor build. Official VAL has not been
evaluated.
"""
    (ROOT / "execution" / "NEURAL_MODEL_SELECTION.md").write_text(md, encoding="utf-8", newline="\n")
    return configs


def write_compute_decision(total_scenarios: int) -> str:
    text = f"""# TRAIN Compute Decision

Decision date: 2026-09-07, Asia/Shanghai.

Full TRAIN remains the scientific preference. The local official TRAIN split
contains `{total_scenarios}` scenarios and approximately 47 GB of raw files.

The approved HGB model-selection budget is 24 configurations x 5 grouped folds
x 4 information states x 2 endpoint regressors, before the matched-H/C and
second-estimator budgets. Running that full budget on all `{total_scenarios}`
scenarios in the current workstation context is a practical wall-time and
memory risk before official-VAL freeze review.

Therefore the frozen computational fallback is triggered:

- max TRAIN scenarios: 50,000
- selection rule: deterministic city-stratified SHA-256 ordering
- seed: `{SPLIT_SEED}`
- sample-size choice is not based on predictive results
- official VAL is not read
"""
    path = ROOT / "execution" / "TRAIN_COMPUTE_DECISION.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256_file(path)


def scan_train(limit: int | None = None, workers: int = 1) -> pd.DataFrame:
    rows = []
    files = scenario_parquets(limit)
    start = time.time()
    if workers <= 1:
        for i, path in enumerate(files, 1):
            rows.append(inspect_scenario(path).__dict__)
            if i % 1000 == 0:
                print(f"scanned {i}/{len(files)} train scenarios in {time.time() - start:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, result in enumerate(ex.map(inspect_scenario, files, chunksize=64), 1):
                rows.append(result.__dict__)
                if i % 1000 == 0:
                    print(
                        f"scanned {i}/{len(files)} train scenarios with {workers} workers in {time.time() - start:.1f}s",
                        flush=True,
                    )
    return pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)


def write_eligibility_outputs(df: pd.DataFrame, complete_scan: bool) -> dict:
    total = len(df)
    primary_pop = df[df["eligible_primary_targets"] >= 2].copy()
    object_counter = Counter()
    for obj in df["object_counts"]:
        object_counter.update(obj)
    summary_rows = [
        {"section": "overall", "key": "complete_train_scan", "value": str(bool(complete_scan))},
        {"section": "overall", "key": "total_TRAIN_scenarios", "value": total},
        {"section": "overall", "key": "scenarios_with_ge_1_eligible_primary_target", "value": int((df["eligible_primary_targets"] >= 1).sum())},
        {"section": "overall", "key": "scenarios_with_ge_2_eligible_primary_targets", "value": int((df["eligible_primary_targets"] >= 2).sum())},
        {"section": "overall", "key": "eligible_target_rows", "value": int(primary_pop["eligible_primary_targets"].sum())},
        {"section": "overall", "key": "FOCAL_TRACK_count", "value": int(primary_pop["focal_tracks"].sum())},
        {"section": "overall", "key": "SCORED_TRACK_count", "value": int(primary_pop["scored_tracks"].sum())},
        {"section": "overall", "key": "N_count", "value": int((primary_pop["n_track_id"] != "").sum())},
        {"section": "overall", "key": "O_count", "value": int(primary_pop["o_count"].sum())},
        {"section": "overall", "key": "map_lite_missing_scenarios", "value": int(df["map_lite_missing"].sum())},
        {"section": "overall", "key": "SCORED_only_sensitivity_eligible_scenarios", "value": int((df["eligible_scored_vehicle_targets"] >= 2).sum())},
        {"section": "overall", "key": "SCORED_only_sensitivity_eligible_rows", "value": int(df.loc[df["eligible_scored_vehicle_targets"] >= 2, "eligible_scored_vehicle_targets"].sum())},
    ]
    for city, g in df.groupby("city"):
        summary_rows.append({"section": "by_city", "key": f"{city}:total_scenarios", "value": len(g)})
        summary_rows.append({"section": "by_city", "key": f"{city}:primary_eligible_scenarios", "value": int((g["eligible_primary_targets"] >= 2).sum())})
        summary_rows.append({"section": "by_city", "key": f"{city}:primary_eligible_rows", "value": int(g.loc[g["eligible_primary_targets"] >= 2, "eligible_primary_targets"].sum())})
    for obj, count in sorted(object_counter.items()):
        summary_rows.append({"section": "object_type_counts_all_tracks", "key": obj, "value": count})
    per_counts = primary_pop["eligible_primary_targets"].value_counts().sort_index()
    for n, count in per_counts.items():
        summary_rows.append({"section": "eligible_targets_per_scenario", "key": str(n), "value": int(count)})
    out = ROOT / "audit" / "TRAIN_ELIGIBILITY_FINAL.csv"
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    df.drop(columns=["object_counts"]).to_csv(ROOT / "audit" / "TRAIN_ELIGIBILITY_SCENARIOS.csv", index=False)
    report = ["# TRAIN Eligibility Report", "", f"Complete TRAIN scan: `{complete_scan}`", ""]
    for row in summary_rows:
        if row["section"] == "overall":
            report.append(f"- {row['key']}: `{row['value']}`")
    report.append("")
    report.append("Official VAL was not read or evaluated.")
    (ROOT / "audit" / "TRAIN_ELIGIBILITY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {r["key"]: r["value"] for r in summary_rows if r["section"] == "overall"}


def write_split_and_shuffle(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    primary = df[df["eligible_primary_targets"] >= 2][["scenario_id", "city"]].copy()
    sample_ids = deterministic_scenario_sample(primary["scenario_id"], min(50000, len(primary)), SPLIT_SEED)
    primary["train_compute_sample"] = primary["scenario_id"].isin(sample_ids)
    folds = grouped_city_stratified_folds(primary[primary["train_compute_sample"]][["scenario_id", "city"]], 5, SPLIT_SEED)
    primary = primary.merge(folds[["scenario_id", "fold"]], on="scenario_id", how="left")
    primary.to_csv(ROOT / "execution" / "TRAIN_SCENARIO_FOLDS.csv", index=False)
    split_hash = sha256_file(ROOT / "execution" / "TRAIN_SCENARIO_FOLDS.csv")

    edges_rows = []
    for city, g in df.replace([np.inf, -np.inf], np.nan).dropna(subset=["av_speed_49", "av_yaw_rate_45_49"]).groupby("city"):
        for var in ["av_speed_49", "av_yaw_rate_45_49"]:
            q = g[var].quantile([0, 0.25, 0.5, 0.75, 1.0]).to_dict()
            edges_rows.append(
                {
                    "city": city,
                    "variable": var,
                    "q0": q[0],
                    "q25": q[0.25],
                    "q50": q[0.5],
                    "q75": q[0.75],
                    "q100": q[1.0],
                }
            )
    edges = pd.DataFrame(edges_rows)
    write_shuffle_spec(edges)
    return split_hash, edges


def write_execution_manifest(values: dict) -> str:
    manifest = {
        "execution_freeze_id": "AV2_EXECUTION_FREEZE_PHASE_3A",
        "date_local": "2026-09-07",
        "timezone": "Asia/Shanghai",
        "official_val_outcome_blind": True,
        "scientific_protocol_v1_1_hash": PROTOCOL_HASH,
        **values,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": 5000,
        "pseudo_nearest_permutation_seed": PSEUDO_NEAREST_SEED,
        "pseudo_nearest_permutation_count": 5000,
        "shuffle_train_seed": SHUFFLE_TRAIN_SEED,
        "shuffle_future_val_seed": SHUFFLE_VAL_SEED,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": __import__("pyarrow").__version__,
        },
        "final_execution_commands": [
            "python tools/phase3a_train_only_freeze.py --scan-train",
            "pytest tests/test_coordinate_transform.py tests/test_information_leakage.py tests/test_role_assignment.py tests/test_split_integrity.py",
        ],
    }
    yaml_path = ROOT / "execution" / "AV2_EXECUTION_FREEZE.yaml"
    yaml_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# AV2 Execution Freeze", "", "Official VAL remains completely outcome-blind.", ""]
    for k, v in manifest.items():
        md.append(f"- `{k}`: `{v}`")
    (ROOT / "execution" / "AV2_EXECUTION_FREEZE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    digest = sha256_file(yaml_path)
    (ROOT / "execution" / "AV2_EXECUTION_HASH.txt").write_text(f"AV2_EXECUTION_FREEZE_SHA256={digest}\n", encoding="utf-8")
    return digest


def run(args: argparse.Namespace) -> None:
    mkdirs()
    transport_hash = write_transportability_clarification()
    feature_hash = write_feature_schema()
    surrounding_hash = write_surrounding_spec()
    total_train = len(scenario_parquets(None))
    compute_hash = write_compute_decision(total_train)
    configs = write_static_model_configs()
    neural = write_neural_configs()

    df = scan_train(args.limit, args.workers) if args.scan_train else pd.DataFrame()
    if df.empty:
        complete = False
        write_shuffle_spec()
        elig = {"total_TRAIN_scenarios": total_train}
        split_hash = ""
        cities = []
        edges_preview = []
    else:
        complete = args.limit is None
        elig = write_eligibility_outputs(df, complete)
        split_hash, edges = write_split_and_shuffle(df)
        cities = sorted(df["city"].dropna().unique().tolist())
        edges_preview = edges.to_dict(orient="records")

    manifest_values = {
        "transportability_clarification_hash": transport_hash,
        "feature_schema_hash": feature_hash,
        "surrounding_agent_spec_hash": surrounding_hash,
        "train_compute_decision_hash": compute_hash,
        "train_compute_decision": "50000_scenario_fallback",
        "eligible_train_population_reference": "audit/TRAIN_ELIGIBILITY_FINAL.csv",
        "split_fold_assignment_hash": split_hash,
        "primary_model_configs": configs,
        "neural_model_configs": neural,
        "map_lite_implementation": "frozen registry variables; missing indicators retained; map extraction not allowed to use VAL",
        "normalization_statistics": "TRAIN-only estimation during final model fit",
        "shuffle_bin_edges_reference": "execution/SHUFFLE_BIN_EDGES.csv",
        "city_list_train": cities,
        "training_data_hash_manifest_reference": "audit/AV2_FILE_HASHES.csv",
        "eligibility_counts": elig,
        "shuffle_cutpoints_preview": edges_preview,
        "constant_velocity_vs_H_train_development_adequacy": "pending full TRAIN-development HGB CV; no VAL touched",
    }
    freeze_hash = write_execution_manifest(manifest_values)
    print(json.dumps({"transportability_hash": transport_hash, "feature_hash": feature_hash, "execution_freeze_hash": freeze_hash}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-train", action="store_true", help="scan official TRAIN only and write eligibility/fold/shuffle artifacts")
    parser.add_argument("--limit", type=int, default=None, help="debug limit for TRAIN scenarios")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)), help="parallel TRAIN scan workers")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
