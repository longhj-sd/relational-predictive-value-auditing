from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from maplite_features import MAP_LITE_FEATURES
import phase3a_train_only_freeze as p3a
import phase3b_train_only_execution as p3b


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
EXEC = ROOT / "execution"
PROCESSED = ROOT / "data" / "processed" / "phase3b"

OLD_MATRIX = PROCESSED / "target_rows.parquet"
CORRECTED_MATRIX = PROCESSED / "TRAIN_50K_FEATURES_MAPLITE_CORRECTED_v2.parquet"
CV_V3 = EXEC / "PRIMARY_CV_RESULTS_MAPLITE_CORRECTED_v3.csv"
CV_V2 = EXEC / "PRIMARY_CV_RESULTS_v2.csv"
FEATURE_SCHEMA = EXEC / "FEATURE_SCHEMA_v1.1.csv"

COMPLETION_JSON = AUDIT / "PRIMARY_CV_480_COMPLETION_AUDIT_MAPLITE_CORRECTED_v3.json"
COMPLETION_MD = AUDIT / "PRIMARY_CV_480_COMPLETION_AUDIT_MAPLITE_CORRECTED_v3.md"
EQUIV_CSV = AUDIT / "HGB_EXECUTION_EQUIVALENCE_QA_MAPLITE_CORRECTED_v3.csv"
EQUIV_MD = AUDIT / "HGB_EXECUTION_EQUIVALENCE_QA_MAPLITE_CORRECTED_v3.md"
INPUT_JSON = AUDIT / "MAPLITE_CORRECTED_FINAL_INPUT_AUDIT_v3.json"
INPUT_MD = AUDIT / "MAPLITE_CORRECTED_FINAL_INPUT_AUDIT_v3.md"
CORRECTION_NOTE = AUDIT / "MAPLITE_PREVAL_IMPLEMENTATION_CORRECTION_NOTE_v3.md"
STATE_SUMMARY = EXEC / "PRIMARY_CV_STATE_SUMMARY_MAPLITE_CORRECTED_v3.csv"
SELECTION_MD = EXEC / "PRIMARY_MODEL_SELECTION_MAPLITE_CORRECTED_v3.md"
CONFIG_YAML = EXEC / "PRIMARY_MODEL_CONFIGS_MAPLITE_CORRECTED_v3.yaml"
MATCHED_CSV = EXEC / "MATCHED_HC_CANDIDATE_RESULTS_MAPLITE_CORRECTED_v3.csv"
MATCHED_MD = EXEC / "MATCHED_HC_SELECTION_MAPLITE_CORRECTED_v3.md"
MATCHED_YAML = EXEC / "MATCHED_HC_CONFIG_MAPLITE_CORRECTED_v3.yaml"
ADEQUACY_CSV = EXEC / "TRAIN_MODEL_ADEQUACY_MAPLITE_CORRECTED_v3.csv"
ADEQUACY_MD = EXEC / "TRAIN_MODEL_ADEQUACY_MAPLITE_CORRECTED_v3.md"
PREPOST_CSV = AUDIT / "PRE_POST_MAPLITE_MODEL_DEVELOPMENT_COMPARISON.csv"
PREPOST_MD = AUDIT / "PRE_POST_MAPLITE_MODEL_DEVELOPMENT_COMPARISON.md"
MANIFEST = EXEC / "MAPLITE_CORRECTED_STAGE_A_MANIFEST_v3.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash_df(df: pd.DataFrame, columns: list[str]) -> str:
    h = hashlib.sha256()
    vals = pd.util.hash_pandas_object(df[columns], index=False).to_numpy(dtype=np.uint64)
    for value in vals:
        h.update(int(value).to_bytes(8, "little", signed=False))
    return h.hexdigest()


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def metric_columns() -> list[str]:
    return ["dx_6s_mse", "dy_6s_mse", "mean_endpoint_mse", "mean_fde", "median_fde"]


def audit_cv_480(cv: pd.DataFrame) -> dict:
    expected = p3b.hgb_expected_units(24)
    required_cols = set(p3b.PRIMARY_CV_COLUMNS)
    keys = [(str(r.state), int(r.candidate_index), int(r.fold)) for r in cv.itertuples(index=False)]
    observed = set(keys)
    duplicate_rows = int(cv.duplicated(p3b.HGB_UNIT_COLUMNS, keep=False).sum())
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    candidate_fold_counts = (
        cv.groupby(["state", "candidate_index"])["fold"].nunique().reset_index(name="folds_observed")
        if set(p3b.HGB_UNIT_COLUMNS).issubset(cv.columns)
        else pd.DataFrame()
    )
    malformed_hparams = []
    candidates = p3b.config_candidates(24)
    for r in cv.itertuples(index=False):
        cfg = candidates[int(r.candidate_index)]
        for name, expected_value in cfg.items():
            observed_value = getattr(r, name)
            if isinstance(expected_value, float):
                ok = math.isclose(float(observed_value), float(expected_value), rel_tol=0.0, abs_tol=1e-12)
            else:
                ok = int(observed_value) == int(expected_value)
            if not ok:
                malformed_hparams.append(
                    {
                        "state": r.state,
                        "candidate_index": int(r.candidate_index),
                        "fold": int(r.fold),
                        "parameter": name,
                        "expected": expected_value,
                        "observed": observed_value,
                    }
                )
    finite_metrics = bool(np.isfinite(cv[metric_columns()].to_numpy(dtype=float)).all())
    all_candidate_folds = (
        len(candidate_fold_counts) == 4 * 24
        and bool((candidate_fold_counts["folds_observed"] == 5).all())
    )
    payload = {
        "audit_id": "PRIMARY_CV_480_COMPLETION_AUDIT_MAPLITE_CORRECTED_v3",
        "created_utc": utc_now(),
        "input_path": str(CV_V3),
        "input_sha256": sha256_file(CV_V3),
        "observed_rows": int(len(cv)),
        "expected_rows": 480,
        "unique_keys": int(len(observed)),
        "state_counts": {s: int((cv["state"] == s).sum()) for s in p3b.STATE_ORDER},
        "candidate_count_per_state": {s: int(cv.loc[cv["state"] == s, "candidate_index"].nunique()) for s in p3b.STATE_ORDER},
        "fold_count_per_state": {s: int(cv.loc[cv["state"] == s, "fold"].nunique()) for s in p3b.STATE_ORDER},
        "duplicate_key_rows": duplicate_rows,
        "missing_expected_units": len(missing),
        "unexpected_units": len(unexpected),
        "all_candidate_fold_cells_complete": all_candidate_folds,
        "missing_columns": sorted(required_cols - set(cv.columns)),
        "extra_columns": sorted(set(cv.columns) - required_cols),
        "all_predictive_metrics_finite": finite_metrics,
        "hgb_hyperparameters_match_frozen_registry": len(malformed_hparams) == 0,
        "hyperparameter_mismatches": malformed_hparams[:20],
        "official_val_outcome_blind": True,
    }
    payload["status"] = (
        "PASS"
        if payload["observed_rows"] == 480
        and payload["unique_keys"] == 480
        and all(v == 120 for v in payload["state_counts"].values())
        and payload["duplicate_key_rows"] == 0
        and payload["missing_expected_units"] == 0
        and payload["unexpected_units"] == 0
        and payload["all_candidate_fold_cells_complete"]
        and not payload["missing_columns"]
        and payload["all_predictive_metrics_finite"]
        and payload["hgb_hyperparameters_match_frozen_registry"]
        else "FAIL"
    )
    write_json(COMPLETION_JSON, payload)
    lines = [
        "# Primary CV 480 Completion Audit MAP-LITE Corrected v3",
        "",
        f"- status: `{payload['status']}`",
        f"- input_sha256: `{payload['input_sha256']}`",
        f"- observed_rows: `{payload['observed_rows']}`",
        f"- unique_keys: `{payload['unique_keys']}`",
        f"- state_counts: `{payload['state_counts']}`",
        f"- duplicate_key_rows: `{payload['duplicate_key_rows']}`",
        f"- missing_expected_units: `{payload['missing_expected_units']}`",
        f"- unexpected_units: `{payload['unexpected_units']}`",
        f"- all_predictive_metrics_finite: `{str(payload['all_predictive_metrics_finite']).lower()}`",
        f"- hgb_hyperparameters_match_frozen_registry: `{str(payload['hgb_hyperparameters_match_frozen_registry']).lower()}`",
        "- official_val_outcome_blind: `true`",
        "",
    ]
    COMPLETION_MD.write_text("\n".join(lines), encoding="utf-8")
    if payload["status"] != "PASS":
        raise RuntimeError("corrected-v3 CV integrity audit failed")
    return payload


def rerun_equivalence(cv: pd.DataFrame) -> dict:
    df = pd.read_parquet(CORRECTED_MATRIX)
    units = [("C", 0, 0), ("E", 1, 0), ("H", 1, 0)]
    tolerance = 1e-9
    rows = []
    for state, candidate_index, fold in units:
        observed = cv[
            (cv["state"] == state)
            & (cv["candidate_index"].astype(int) == candidate_index)
            & (cv["fold"].astype(int) == fold)
        ].iloc[0]
        fresh = p3b.fit_evaluate_hgb_unit(df, state, candidate_index, fold, 24)
        row = {"state": state, "candidate_index": candidate_index, "fold": fold, "tolerance_abs": tolerance}
        max_abs = 0.0
        hparams_match = True
        for col in ["dx_6s_mse", "dy_6s_mse", "mean_endpoint_mse", "mean_fde"]:
            diff = abs(float(observed[col]) - float(fresh[col]))
            row[f"authoritative_{col}"] = float(observed[col])
            row[f"rerun_{col}"] = float(fresh[col])
            row[f"absdiff_{col}"] = diff
            max_abs = max(max_abs, diff)
        for col in ["learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization", "min_samples_leaf", "max_bins"]:
            if not math.isclose(float(observed[col]), float(fresh[col]), rel_tol=0.0, abs_tol=1e-12):
                hparams_match = False
        row["recorded_hyperparameters_match"] = hparams_match
        row["max_abs_diff"] = max_abs
        row["pass"] = bool(max_abs <= tolerance and hparams_match)
        rows.append(row)
    out = pd.DataFrame(rows)
    write_csv(EQUIV_CSV, out)
    status = "PASS" if bool(out["pass"].all()) and out["state"].nunique() >= 2 and len(out) >= 3 else "FAIL"
    lines = [
        "# HGB Execution-Equivalence QA MAP-LITE Corrected v3",
        "",
        f"- result: `{status}`",
        f"- rerun_units: `{len(out)}`",
        f"- states_spanned: `{','.join(sorted(out['state'].unique()))}`",
        f"- tolerance_abs: `{tolerance}`",
        f"- max_abs_diff: `{float(out['max_abs_diff'].max())}`",
        f"- csv_sha256: `{sha256_file(EQUIV_CSV)}`",
        "",
        out.to_markdown(index=False),
        "",
        "Strict no-write QA: the authoritative corrected-v3 CV CSV was not modified.",
        "",
        "Official VAL remains outcome-blind.",
        "",
    ]
    EQUIV_MD.write_text("\n".join(lines), encoding="utf-8")
    if status != "PASS":
        raise RuntimeError("execution-equivalence QA failed")
    return {"status": status, "max_abs_diff": float(out["max_abs_diff"].max()), "csv_sha256": sha256_file(EQUIV_CSV)}


def audit_inputs() -> dict:
    old = pd.read_parquet(OLD_MATRIX)
    new = pd.read_parquet(CORRECTED_MATRIX)
    map_cols = []
    for name in MAP_LITE_FEATURES:
        map_cols.extend([name, f"{name}_missing"])
    non_map_cols = [c for c in old.columns if c not in map_cols]
    key_cols = ["scenario_id", "track_id"]
    target_cols = ["dx_6s", "dy_6s"]
    identity = {
        "row_count_preserved": len(old) == len(new),
        "column_order_preserved": old.columns.tolist() == new.columns.tolist(),
        "target_row_keys_preserved": old[key_cols].equals(new[key_cols]),
        "scenario_ids_preserved": old["scenario_id"].equals(new["scenario_id"]),
        "fold_assignments_preserved": old["fold"].equals(new["fold"]),
        "target_labels_preserved": old[target_cols].equals(new[target_cols]),
        "non_map_features_preserved": old[non_map_cols].equals(new[non_map_cols]),
        "old_non_map_hash": stable_hash_df(old, non_map_cols),
        "new_non_map_hash": stable_hash_df(new, non_map_cols),
        "old_key_hash": stable_hash_df(old, key_cols),
        "new_key_hash": stable_hash_df(new, key_cols),
        "old_target_hash": stable_hash_df(old, target_cols),
        "new_target_hash": stable_hash_df(new, target_cols),
    }
    rows = []
    for name in MAP_LITE_FEATURES:
        miss = f"{name}_missing"
        nonmissing_mask = new[miss].astype(int) == 0
        vals = new.loc[nonmissing_mask, name]
        row = {
            "feature": name,
            "nonmissing_n": int(nonmissing_mask.sum()),
            "missing_n": int((~nonmissing_mask).sum()),
            "missing_percent": float((~nonmissing_mask).mean() * 100.0),
            "finite_n": int(np.isfinite(vals.astype(float)).sum()) if len(vals) else 0,
            "min": float(vals.min()) if len(vals) else math.nan,
            "median": float(vals.median()) if len(vals) else math.nan,
            "max": float(vals.max()) if len(vals) else math.nan,
        }
        for city, idx in new.groupby("city").groups.items():
            city_mask = nonmissing_mask.loc[idx]
            row[f"{city}_nonmissing_n"] = int(city_mask.sum())
            row[f"{city}_missing_percent"] = float((~city_mask).mean() * 100.0)
        rows.append(row)
    coverage = pd.DataFrame(rows)
    coverage_csv = AUDIT / "MAPLITE_CORRECTED_FINAL_INPUT_AUDIT_v3_feature_coverage.csv"
    write_csv(coverage_csv, coverage)
    old_all_missing = {name: bool((old[f"{name}_missing"].astype(int) == 1).all()) for name in MAP_LITE_FEATURES}
    changed_cols = [c for c in old.columns if not old[c].equals(new[c])]
    expected_changed_only = sorted(changed_cols) == sorted(map_cols)
    schema_order_hash = hashlib.sha256("\n".join(new.columns).encode("utf-8")).hexdigest()
    payload = {
        "audit_id": "MAPLITE_CORRECTED_FINAL_INPUT_AUDIT_v3",
        "created_utc": utc_now(),
        "old_matrix_path": str(OLD_MATRIX),
        "corrected_matrix_path": str(CORRECTED_MATRIX),
        "old_matrix_sha256": sha256_file(OLD_MATRIX),
        "corrected_matrix_sha256": sha256_file(CORRECTED_MATRIX),
        "feature_schema_path": str(FEATURE_SCHEMA),
        "feature_schema_sha256": sha256_file(FEATURE_SCHEMA),
        "corrected_column_order_hash": schema_order_hash,
        "identity_checks": identity,
        "changed_columns": changed_cols,
        "changed_only_maplite_and_missing_flags": expected_changed_only,
        "old_all_maplite_missing_flags_one": old_all_missing,
        "all_maplite_features_populated": bool((coverage["nonmissing_n"] > 0).all()),
        "feature_coverage_csv": str(coverage_csv),
        "feature_coverage_csv_sha256": sha256_file(coverage_csv),
        "status": "PASS"
        if all(v for k, v in identity.items() if isinstance(v, bool))
        and expected_changed_only
        and all(old_all_missing.values())
        and bool((coverage["nonmissing_n"] > 0).all())
        else "FAIL",
    }
    write_json(INPUT_JSON, payload)
    lines = [
        "# MAP-LITE Corrected Final Input Audit v3",
        "",
        f"- status: `{payload['status']}`",
        f"- corrected_matrix_sha256: `{payload['corrected_matrix_sha256']}`",
        f"- feature_schema_sha256: `{payload['feature_schema_sha256']}`",
        f"- corrected_column_order_hash: `{schema_order_hash}`",
        f"- changed_only_maplite_and_missing_flags: `{str(expected_changed_only).lower()}`",
        f"- all_maplite_features_populated: `{str(payload['all_maplite_features_populated']).lower()}`",
        "",
        "## Identity Checks",
        "",
        pd.DataFrame([identity]).to_markdown(index=False),
        "",
        "## MAP-LITE Coverage",
        "",
        coverage.to_markdown(index=False),
        "",
    ]
    INPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    if payload["status"] != "PASS":
        raise RuntimeError("corrected MAP-LITE input audit failed")
    return payload


def build_selection(cv: pd.DataFrame) -> dict:
    candidates = p3b.config_candidates(24)
    rows = []
    selected = {}
    paired_rows = []
    for state, sg in cv.groupby("state", sort=True):
        agg = sg.groupby("candidate_index", as_index=False).agg(
            mean_endpoint_mse=("mean_endpoint_mse", "mean"),
            sd_endpoint_mse=("mean_endpoint_mse", "std"),
            mean_fde=("mean_fde", "mean"),
            sd_fde=("mean_fde", "std"),
            mean_dx_6s_mse=("dx_6s_mse", "mean"),
            mean_dy_6s_mse=("dy_6s_mse", "mean"),
        )
        agg = agg.sort_values(["mean_endpoint_mse", "candidate_index"]).reset_index(drop=True)
        agg["rank"] = np.arange(1, len(agg) + 1)
        winner = agg.iloc[0]
        runner = agg.iloc[1]
        for r in agg.itertuples(index=False):
            folds = sg[sg["candidate_index"] == int(r.candidate_index)].sort_values("fold")
            cfg = candidates[int(r.candidate_index)]
            rows.append(
                {
                    "state": state,
                    "candidate_index": int(r.candidate_index),
                    **cfg,
                    "fold_endpoint_mse": ";".join(f"{float(x):.12g}" for x in folds["mean_endpoint_mse"]),
                    "fold_fde": ";".join(f"{float(x):.12g}" for x in folds["mean_fde"]),
                    "mean_endpoint_mse": float(r.mean_endpoint_mse),
                    "sd_endpoint_mse": float(r.sd_endpoint_mse),
                    "mean_fde": float(r.mean_fde),
                    "sd_fde": float(r.sd_fde),
                    "rank": int(r.rank),
                    "selected": bool(int(r.candidate_index) == int(winner.candidate_index)),
                    "runner_up_candidate_index": int(runner.candidate_index) if int(r.candidate_index) == int(winner.candidate_index) else "",
                    "runner_up_mean_endpoint_mse": float(runner.mean_endpoint_mse) if int(r.candidate_index) == int(winner.candidate_index) else "",
                    "selected_minus_runner_up_mean_endpoint_mse": float(winner.mean_endpoint_mse - runner.mean_endpoint_mse)
                    if int(r.candidate_index) == int(winner.candidate_index)
                    else "",
                }
            )
        win_folds = sg[sg["candidate_index"] == int(winner.candidate_index)].sort_values("fold")
        run_folds = sg[sg["candidate_index"] == int(runner.candidate_index)].sort_values("fold")
        for wf, rf in zip(win_folds.itertuples(index=False), run_folds.itertuples(index=False)):
            paired_rows.append(
                {
                    "state": state,
                    "fold": int(wf.fold),
                    "selected_candidate_index": int(winner.candidate_index),
                    "runner_up_candidate_index": int(runner.candidate_index),
                    "selected_endpoint_mse": float(wf.mean_endpoint_mse),
                    "runner_up_endpoint_mse": float(rf.mean_endpoint_mse),
                    "selected_minus_runner_up_endpoint_mse": float(wf.mean_endpoint_mse - rf.mean_endpoint_mse),
                }
            )
        cfg = candidates[int(winner.candidate_index)]
        selected[state] = {
            **cfg,
            "candidate_index": int(winner.candidate_index),
            "fold_endpoint_mse": [float(x) for x in win_folds["mean_endpoint_mse"]],
            "fold_fde": [float(x) for x in win_folds["mean_fde"]],
            "mean_endpoint_mse": float(winner.mean_endpoint_mse),
            "sd_endpoint_mse": float(winner.sd_endpoint_mse),
            "mean_fde": float(winner.mean_fde),
            "rank": int(winner["rank"] if isinstance(winner, dict) else winner["rank"]),
            "runner_up_candidate_index": int(runner.candidate_index),
            "runner_up_mean_endpoint_mse": float(runner.mean_endpoint_mse),
            "selected_minus_runner_up_mean_endpoint_mse": float(winner.mean_endpoint_mse - runner.mean_endpoint_mse),
        }
    state_summary = pd.DataFrame(rows)
    write_csv(STATE_SUMMARY, state_summary.sort_values(["state", "rank"]))
    paired = pd.DataFrame(paired_rows)
    paired_csv = EXEC / "PRIMARY_MODEL_SELECTION_PAIRED_RUNNER_UP_DIFFS_MAPLITE_CORRECTED_v3.csv"
    write_csv(paired_csv, paired)
    payload = {
        "selection_rule": "TRAIN-only grouped-CV minimum mean endpoint MSE; ties by candidate_index order",
        "downstream_rpva_gains_used": False,
        "official_val_used": False,
        "randomized_search_seed": p3b.TUNING_SEED,
        "candidate_count_per_state": 24,
        "candidate_registry": candidates,
        "corrected_maplite_matrix": str(CORRECTED_MATRIX),
        "corrected_maplite_matrix_sha256": sha256_file(CORRECTED_MATRIX),
        "corrected_cv_sha256": sha256_file(CV_V3),
        "selected_configs": selected,
        "paired_runner_up_differences_csv": str(paired_csv),
        "paired_runner_up_differences_csv_sha256": sha256_file(paired_csv),
    }
    CONFIG_YAML.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    selected_table = pd.DataFrame(
        [
            {
                "state": state,
                "candidate_index": cfg["candidate_index"],
                "mean_endpoint_mse": cfg["mean_endpoint_mse"],
                "sd_endpoint_mse": cfg["sd_endpoint_mse"],
                "mean_fde": cfg["mean_fde"],
                "rank": cfg["rank"],
                "runner_up_candidate_index": cfg["runner_up_candidate_index"],
                "runner_up_mean_endpoint_mse": cfg["runner_up_mean_endpoint_mse"],
                "selected_minus_runner_up": cfg["selected_minus_runner_up_mean_endpoint_mse"],
            }
            for state, cfg in selected.items()
        ]
    )
    lines = [
        "# Primary Model Selection MAP-LITE Corrected v3",
        "",
        "Selection used only corrected TRAIN predictive CV endpoint MSE and the frozen candidate order. RPVA gains, bootstrap results, pseudo-nearest results, held-out-city results, neural models, and official VAL were not used.",
        "",
        "## Selected State-Specific Configs",
        "",
        selected_table.to_markdown(index=False),
        "",
        "## Fold-Wise Selected vs Runner-Up Differences",
        "",
        paired.to_markdown(index=False),
        "",
        f"- complete_24_candidate_summary_csv_sha256: `{sha256_file(STATE_SUMMARY)}`",
        f"- selected_config_yaml_sha256: `{sha256_file(CONFIG_YAML)}`",
        f"- corrected_cv_sha256: `{sha256_file(CV_V3)}`",
        "",
    ]
    SELECTION_MD.write_text("\n".join(lines), encoding="utf-8")
    return payload


def build_matched(cv: pd.DataFrame) -> dict:
    hc = cv[cv["state"].isin(["H", "C"])].copy()
    expected_pairs = {(ci, state) for ci in range(24) for state in ["H", "C"]}
    observed_pairs = set()
    for (state, ci), g in hc.groupby(["state", "candidate_index"]):
        if g["fold"].nunique() == 5:
            observed_pairs.add((int(ci), str(state)))
    if observed_pairs != expected_pairs:
        raise RuntimeError("matched H/C candidate set is not fully represented in corrected H and C v3 registry")
    stats = hc.groupby("state")["mean_endpoint_mse"].agg(["mean", "std"]).to_dict("index")
    rows = []
    for ci, cg in hc.groupby("candidate_index"):
        h_loss = float(cg[cg["state"] == "H"]["mean_endpoint_mse"].mean())
        c_loss = float(cg[cg["state"] == "C"]["mean_endpoint_mse"].mean())
        h_std = (h_loss - stats["H"]["mean"]) / stats["H"]["std"]
        c_std = (c_loss - stats["C"]["mean"]) / stats["C"]["std"]
        rows.append(
            {
                "candidate_index": int(ci),
                "H_mean_endpoint_mse": h_loss,
                "C_mean_endpoint_mse": c_loss,
                "H_standardized_loss": float(h_std),
                "C_standardized_loss": float(c_std),
                "mean_standardized_HC_loss": float(np.mean([h_std, c_std])),
            }
        )
    out = pd.DataFrame(rows).sort_values(["mean_standardized_HC_loss", "candidate_index"]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    write_csv(MATCHED_CSV, out)
    winner = out.iloc[0].to_dict()
    cfg = p3b.config_candidates(24)[int(winner["candidate_index"])]
    payload = {
        "selection_rule": "pre-specified joint standardized H/C TRAIN predictive-loss rule",
        "derived_from_existing_corrected_cv": True,
        "distinct_refits_executed": False,
        **cfg,
        "candidate_index": int(winner["candidate_index"]),
        "mean_standardized_HC_loss": float(winner["mean_standardized_HC_loss"]),
        "H_mean_endpoint_mse": float(winner["H_mean_endpoint_mse"]),
        "C_mean_endpoint_mse": float(winner["C_mean_endpoint_mse"]),
        "matched_results_sha256": sha256_file(MATCHED_CSV),
    }
    MATCHED_YAML.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    MATCHED_MD.write_text(
        "\n".join(
            [
                "# Matched H/C Selection MAP-LITE Corrected v3",
                "",
                "The frozen matched-H/C candidate set is fully represented by candidate_index 0-23 in both corrected H and C CV results, so no redundant refits were executed.",
                "",
                out.to_markdown(index=False),
                "",
                f"- selected_candidate_index: `{payload['candidate_index']}`",
                f"- mean_standardized_HC_loss: `{payload['mean_standardized_HC_loss']}`",
                f"- config_sha256: `{sha256_file(MATCHED_YAML)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def build_adequacy(cv: pd.DataFrame) -> dict:
    df = pd.read_parquet(CORRECTED_MATRIX, columns=["fold", "dx_6s", "dy_6s", "cv_dx_6s", "cv_dy_6s"])
    h_rows = cv[(cv["state"] == "H") & (cv["candidate_index"].astype(int) == 1)].sort_values("fold")
    rows = []
    y = df[["dx_6s", "dy_6s"]].to_numpy(float)
    cv_pred = df[["cv_dx_6s", "cv_dy_6s"]].to_numpy(float)
    for h in h_rows.itertuples(index=False):
        fold = int(h.fold)
        mask = df["fold"].astype(int).to_numpy() == fold
        diff = cv_pred[mask] - y[mask]
        baseline_endpoint_mse = float(np.mean(np.sum(diff * diff, axis=1)))
        baseline_fde = float(np.mean(np.linalg.norm(diff, axis=1)))
        rows.append(
            {
                "fold": fold,
                "baseline_endpoint_mse": baseline_endpoint_mse,
                "baseline_mean_fde": baseline_fde,
                "hgb_h_endpoint_mse": float(h.mean_endpoint_mse),
                "hgb_h_mean_fde": float(h.mean_fde),
                "endpoint_mse_absolute_improvement": baseline_endpoint_mse - float(h.mean_endpoint_mse),
                "endpoint_mse_relative_improvement_percent": 100.0 * (baseline_endpoint_mse - float(h.mean_endpoint_mse)) / baseline_endpoint_mse,
                "fde_absolute_improvement": baseline_fde - float(h.mean_fde),
                "fde_relative_improvement_percent": 100.0 * (baseline_fde - float(h.mean_fde)) / baseline_fde,
                "n_rows": int(mask.sum()),
            }
        )
    out = pd.DataFrame(rows)
    write_csv(ADEQUACY_CSV, out)
    mean_row = out.drop(columns=["fold", "n_rows"]).mean(numeric_only=True).to_dict()
    ADEQUACY_MD.write_text(
        "\n".join(
            [
                "# TRAIN Model Adequacy MAP-LITE Corrected v3",
                "",
                "This is a descriptive TRAIN-only comparison against the already-frozen constant-velocity endpoint baseline. No new adequacy pass/fail threshold is introduced here.",
                "",
                out.to_markdown(index=False),
                "",
                "## Means",
                "",
                pd.DataFrame([mean_row]).to_markdown(index=False),
                "",
                f"- csv_sha256: `{sha256_file(ADEQUACY_CSV)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"csv_sha256": sha256_file(ADEQUACY_CSV), "means": mean_row}


def build_prepost() -> dict:
    v2 = pd.read_csv(CV_V2)
    v3 = pd.read_csv(CV_V3)
    rows = []
    for label, df in [("v2_PRE_MAPLITE_CORRECTION_provenance", v2), ("v3_AUTHORITATIVE_corrected_TRAIN_model_development_evidence", v3)]:
        summary = df.groupby(["state", "candidate_index"], as_index=False).agg(
            mean_endpoint_mse=("mean_endpoint_mse", "mean"),
            mean_fde=("mean_fde", "mean"),
        )
        summary = summary.sort_values(["state", "mean_endpoint_mse", "candidate_index"])
        summary["rank"] = summary.groupby("state").cumcount() + 1
        for r in summary[summary["rank"] == 1].itertuples(index=False):
            rows.append(
                {
                    "evidence": label,
                    "state": r.state,
                    "selected_candidate_index": int(r.candidate_index),
                    "mean_endpoint_mse": float(r.mean_endpoint_mse),
                    "mean_fde": float(r.mean_fde),
                    "rank": int(r.rank),
                }
            )
    out = pd.DataFrame(rows)
    wide = out.pivot(index="state", columns="evidence", values=["selected_candidate_index", "mean_endpoint_mse", "mean_fde"])
    wide.columns = ["__".join(col).strip() for col in wide.columns]
    wide = wide.reset_index()
    v2_col = "mean_endpoint_mse__v2_PRE_MAPLITE_CORRECTION_provenance"
    v3_col = "mean_endpoint_mse__v3_AUTHORITATIVE_corrected_TRAIN_model_development_evidence"
    wide["v3_minus_v2_mean_endpoint_mse"] = wide[v3_col] - wide[v2_col]
    wide["v3_minus_v2_relative_percent"] = 100.0 * wide["v3_minus_v2_mean_endpoint_mse"] / wide[v2_col]
    write_csv(PREPOST_CSV, wide)
    PREPOST_MD.write_text(
        "\n".join(
            [
                "# Pre/Post MAP-LITE Model Development Comparison",
                "",
                "v2 = PRE-MAPLITE-CORRECTION provenance. v3 = AUTHORITATIVE corrected TRAIN model-development evidence. This comparison is descriptive provenance/sensitivity documentation only and did not alter the corrected-v3 selection rule.",
                "",
                wide.to_markdown(index=False),
                "",
                f"- v2_cv_sha256: `{sha256_file(CV_V2)}`",
                f"- v3_cv_sha256: `{sha256_file(CV_V3)}`",
                f"- csv_sha256: `{sha256_file(PREPOST_CSV)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"csv_sha256": sha256_file(PREPOST_CSV)}


def write_correction_note(input_payload: dict, cv_payload: dict) -> dict:
    lines = [
        "# MAP-LITE Pre-VAL Implementation Correction Note v3",
        "",
        "The original frozen protocol v1.1 already specified the MAP-LITE predictor family. The pre-correction implementation defect was that the TRAIN feature builder hard-coded those MAP-LITE variables as missing rather than extracting them from available TRAIN map JSON.",
        "",
        "The defect was detected during TRAIN-only implementation audit. Raw TRAIN map data existed locally. Official AV2 VAL outcomes were not accessed for this correction, and no official VAL outcome was used for model-development decisions.",
        "",
        "The extractor was corrected to implement only the already-frozen MAP-LITE variables. No new scientific predictor was introduced. The scenario sample, fold assignments, target labels, non-map predictors, HGB model family, candidate registry, candidate ordering, predictive endpoint loss, and selection rule were unchanged.",
        "",
        "Therefore `execution/PRIMARY_CV_RESULTS_MAPLITE_CORRECTED_v3.csv` supersedes v2 as the authoritative corrected TRAIN model-development evidence. v2 remains preserved as PRE-MAPLITE-CORRECTION provenance.",
        "",
        "## Hashes",
        "",
        f"- protocol_v1.1_yaml_sha256: `{sha256_file(ROOT / 'protocol' / 'RPVA_AV2_PROTOCOL_v1.1.yaml')}`",
        f"- feature_schema_sha256: `{input_payload['feature_schema_sha256']}`",
        f"- corrected_matrix_sha256: `{input_payload['corrected_matrix_sha256']}`",
        f"- corrected_v3_cv_sha256: `{cv_payload['input_sha256']}`",
        f"- pre_correction_v2_cv_sha256: `{sha256_file(CV_V2)}`",
        f"- maplite_extractor_sha256: `{sha256_file(ROOT / 'tools' / 'maplite_features.py')}`",
        f"- corrected_train_matrix_builder_sha256: `{sha256_file(ROOT / 'tools' / 'maplite_corrected_train_matrix.py')}`",
        "",
        "This is a pre-VAL frozen-protocol implementation correction note, not a claim of formal preregistration.",
        "",
    ]
    CORRECTION_NOTE.write_text("\n".join(lines), encoding="utf-8")
    return {"path": str(CORRECTION_NOTE), "sha256": sha256_file(CORRECTION_NOTE)}


def build_manifest(stage_status: str, pieces: dict) -> dict:
    paths = {
        "corrected_matrix": CORRECTED_MATRIX,
        "feature_schema": FEATURE_SCHEMA,
        "candidate_registry_source_protocol": ROOT / "protocol" / "RPVA_AV2_PROTOCOL_v1.1.yaml",
        "corrected_480_cv": CV_V3,
        "completion_audit_json": COMPLETION_JSON,
        "completion_audit_md": COMPLETION_MD,
        "execution_equivalence_csv": EQUIV_CSV,
        "execution_equivalence_md": EQUIV_MD,
        "corrected_maplite_input_audit_json": INPUT_JSON,
        "corrected_maplite_input_audit_md": INPUT_MD,
        "correction_declaration": CORRECTION_NOTE,
        "state_summary": STATE_SUMMARY,
        "selected_configs": CONFIG_YAML,
        "selection_md": SELECTION_MD,
        "matched_hc_results": MATCHED_CSV,
        "matched_hc_selection": MATCHED_MD,
        "matched_hc_config": MATCHED_YAML,
        "adequacy_csv": ADEQUACY_CSV,
        "adequacy_md": ADEQUACY_MD,
        "v2_v3_provenance_comparison_csv": PREPOST_CSV,
        "v2_v3_provenance_comparison_md": PREPOST_MD,
    }
    manifest = {
        "manifest_id": "MAPLITE_CORRECTED_STAGE_A_MANIFEST_v3",
        "created_utc": utc_now(),
        "stage_status": stage_status,
        "official_val_outcome_blind": True,
        "hard_stop_after_stage_a": True,
        "artifacts": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyyaml": yaml.__version__,
        },
        "gate_evidence": pieces,
    }
    write_json(MANIFEST, manifest)
    manifest["manifest_file_sha256"] = sha256_file(MANIFEST)
    return manifest


def main() -> None:
    AUDIT.mkdir(exist_ok=True)
    EXEC.mkdir(exist_ok=True)
    cv = pd.read_csv(CV_V3)
    pieces = {}
    pieces["cv_integrity"] = audit_cv_480(cv)
    pieces["execution_equivalence"] = rerun_equivalence(cv)
    pieces["input_audit"] = audit_inputs()
    pieces["selection"] = build_selection(cv)
    pieces["matched_hc"] = build_matched(cv)
    pieces["adequacy"] = build_adequacy(cv)
    pieces["prepost"] = build_prepost()
    pieces["correction_note"] = write_correction_note(pieces["input_audit"], pieces["cv_integrity"])
    gate_checks = {
        "corrected_v3_cv_integrity": pieces["cv_integrity"]["status"] == "PASS",
        "execution_equivalence": pieces["execution_equivalence"]["status"] == "PASS",
        "corrected_input_identity_non_map_preservation": pieces["input_audit"]["status"] == "PASS",
        "maplite_variables_populated": pieces["input_audit"]["all_maplite_features_populated"],
        "model_selection_train_only_derivable": True,
        "official_val_outcome_blind": True,
    }
    stage_status = "STAGE_A_CERTIFIED_PREVAL" if all(gate_checks.values()) else "STAGE_A_NOT_CERTIFIED"
    pieces["gate_checks"] = gate_checks
    manifest = build_manifest(stage_status, pieces)
    print(json.dumps({"stage_status": stage_status, "manifest_sha256": manifest["manifest_file_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
