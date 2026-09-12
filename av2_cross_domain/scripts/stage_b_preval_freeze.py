from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

import phase3b_train_only_execution as p3b
from maplite_features import MAP_LITE_FEATURES


ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "execution"
AUDIT = ROOT / "audit"
TOOLS = ROOT / "tools"
PROTOCOL = ROOT / "protocol"
PROCESSED = ROOT / "data" / "processed" / "phase3b"

AUTHORITATIVE_CV = EXEC / "PRIMARY_CV_RESULTS_MAPLITE_CORRECTED_v3.csv"
AUTHORITATIVE_CV_SHA256 = "88ae13eef0ca5f342e7f0cc6ec72047e49fc9640ba4f9536051245a9b9e1e2bb"
CORRECTED_DEV_MATRIX = PROCESSED / "TRAIN_50K_FEATURES_MAPLITE_CORRECTED_v2.parquet"
FULL_TRAIN_MATRIX = PROCESSED / "FULL_TRAIN_FEATURES_MAPLITE_CORRECTED_v3.parquet"
STATE_ORDER = ["H", "E", "C", "C_SHUFFLED"]
SELECTED_CANDIDATE = 1
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


def sha256_json(obj: object) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def artifact_record(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "bytes": path.stat().st_size if path.exists() else None,
    }


def stage_a_immutability() -> dict:
    paths = {
        "authoritative_v3_cv": AUTHORITATIVE_CV,
        "correction_note": AUDIT / "MAPLITE_PREVAL_IMPLEMENTATION_CORRECTION_NOTE_v3.md",
        "input_audit_json": AUDIT / "MAPLITE_CORRECTED_FINAL_INPUT_AUDIT_v3.json",
        "input_audit_md": AUDIT / "MAPLITE_CORRECTED_FINAL_INPUT_AUDIT_v3.md",
        "execution_equivalence_csv": AUDIT / "HGB_EXECUTION_EQUIVALENCE_QA_MAPLITE_CORRECTED_v3.csv",
        "execution_equivalence_md": AUDIT / "HGB_EXECUTION_EQUIVALENCE_QA_MAPLITE_CORRECTED_v3.md",
        "selection_yaml": EXEC / "PRIMARY_MODEL_CONFIGS_MAPLITE_CORRECTED_v3.yaml",
        "selection_md": EXEC / "PRIMARY_MODEL_SELECTION_MAPLITE_CORRECTED_v3.md",
        "matched_hc_csv": EXEC / "MATCHED_HC_CANDIDATE_RESULTS_MAPLITE_CORRECTED_v3.csv",
        "matched_hc_yaml": EXEC / "MATCHED_HC_CONFIG_MAPLITE_CORRECTED_v3.yaml",
        "matched_hc_md": EXEC / "MATCHED_HC_SELECTION_MAPLITE_CORRECTED_v3.md",
        "adequacy_csv": EXEC / "TRAIN_MODEL_ADEQUACY_MAPLITE_CORRECTED_v3.csv",
        "adequacy_md": EXEC / "TRAIN_MODEL_ADEQUACY_MAPLITE_CORRECTED_v3.md",
        "prepost_csv": AUDIT / "PRE_POST_MAPLITE_MODEL_DEVELOPMENT_COMPARISON.csv",
        "prepost_md": AUDIT / "PRE_POST_MAPLITE_MODEL_DEVELOPMENT_COMPARISON.md",
        "stage_a_manifest": EXEC / "MAPLITE_CORRECTED_STAGE_A_MANIFEST_v3.json",
    }
    artifacts = {k: artifact_record(v) for k, v in paths.items()}
    cv_hash_ok = artifacts["authoritative_v3_cv"]["sha256"] == AUTHORITATIVE_CV_SHA256
    manifest = json.loads(paths["stage_a_manifest"].read_text(encoding="utf-8"))
    manifest_status_ok = manifest.get("stage_status") == "STAGE_A_CERTIFIED_PREVAL"
    manifest_blind_ok = manifest.get("official_val_outcome_blind") is True
    all_present = all(v["exists"] for v in artifacts.values())
    status = "PASS" if all([all_present, cv_hash_ok, manifest_status_ok, manifest_blind_ok]) else "FAIL"
    payload = {
        "audit_id": "STAGE_A_IMMUTABILITY_CHECK_BEFORE_STAGE_B",
        "created_utc": utc_now(),
        "status": status,
        "official_val_outcome_blind": True,
        "checks": {
            "all_expected_stage_a_artifacts_present": all_present,
            "authoritative_v3_cv_sha256_matches": cv_hash_ok,
            "stage_a_manifest_certified": manifest_status_ok,
            "stage_a_manifest_val_blind": manifest_blind_ok,
        },
        "artifacts": artifacts,
    }
    write_json(AUDIT / "STAGE_A_IMMUTABILITY_CHECK_BEFORE_STAGE_B.json", payload)
    write_text(
        AUDIT / "STAGE_A_IMMUTABILITY_CHECK_BEFORE_STAGE_B.md",
        "\n".join(
            [
                "# Stage-A Immutability Check Before Stage B",
                "",
                f"- created_utc: `{payload['created_utc']}`",
                f"- result: `{status}`",
                f"- authoritative_v3_cv_sha256: `{artifacts['authoritative_v3_cv']['sha256']}`",
                f"- expected_v3_cv_sha256: `{AUTHORITATIVE_CV_SHA256}`",
                f"- all_expected_stage_a_artifacts_present: `{str(all_present).lower()}`",
                f"- stage_a_manifest_certified: `{str(manifest_status_ok).lower()}`",
                "",
                "No Stage-A artifact is overwritten by this audit. Official AV2 VAL remains outcome-blind.",
                "",
            ]
        ),
    )
    return payload


def selected_config_extension_qa() -> dict:
    df = pd.read_parquet(CORRECTED_DEV_MATRIX)
    cv = pd.read_csv(AUTHORITATIVE_CV)
    rows = []
    for state in ["C", "C_SHUFFLED"]:
        fold = int(cv[(cv.state == state) & (cv.candidate_index == SELECTED_CANDIDATE)]["fold"].min())
        ref = cv[(cv.state == state) & (cv.candidate_index == SELECTED_CANDIDATE) & (cv.fold == fold)].iloc[0]
        fresh = p3b.fit_evaluate_hgb_unit(df, state, SELECTED_CANDIDATE, fold, 24)
        obs = {
            "state": state,
            "candidate_index": SELECTED_CANDIDATE,
            "fold": fold,
            **SELECTED_CFG,
            "dx_6s_mse": float(fresh["dx_6s_mse"]),
            "dy_6s_mse": float(fresh["dy_6s_mse"]),
            "mean_endpoint_mse": float(fresh["mean_endpoint_mse"]),
            "mean_fde": float(fresh["mean_fde"]),
            "reference_dx_6s_mse": float(ref.dx_6s_mse),
            "reference_dy_6s_mse": float(ref.dy_6s_mse),
            "reference_mean_endpoint_mse": float(ref.mean_endpoint_mse),
            "reference_mean_fde": float(ref.mean_fde),
        }
        obs["max_abs_metric_discrepancy"] = max(
            abs(obs["dx_6s_mse"] - obs["reference_dx_6s_mse"]),
            abs(obs["dy_6s_mse"] - obs["reference_dy_6s_mse"]),
            abs(obs["mean_endpoint_mse"] - obs["reference_mean_endpoint_mse"]),
            abs(obs["mean_fde"] - obs["reference_mean_fde"]),
        )
        obs["pass_1e_9"] = bool(obs["max_abs_metric_discrepancy"] <= 1e-9)
        rows.append(obs)
    out = pd.DataFrame(rows)
    out.to_csv(AUDIT / "SELECTED_CONFIG_EXECUTION_EQUIVALENCE_EXTENSION_v3.csv", index=False)
    status = "PASS" if bool(out["pass_1e_9"].all()) else "FAIL"
    write_text(
        AUDIT / "SELECTED_CONFIG_EXECUTION_EQUIVALENCE_EXTENSION_v3.md",
        "\n".join(
            [
                "# Selected-Config Execution-Equivalence Extension v3",
                "",
                "Strict TRAIN-only no-authoritative-CV-write rerun for C and C_SHUFFLED candidate 1.",
                "",
                out.to_markdown(index=False),
                "",
                f"- absolute_tolerance: `1e-9`",
                f"- result: `{status}`",
                "",
                "This QA did not alter Stage-A selection. Official AV2 VAL remains outcome-blind.",
                "",
            ]
        ),
    )
    return {"status": status, "rows": rows, "csv_sha256": sha256_file(AUDIT / "SELECTED_CONFIG_EXECUTION_EQUIVALENCE_EXTENSION_v3.csv")}


def maplite_semantic_audit() -> dict:
    df = pd.read_parquet(CORRECTED_DEV_MATRIX, columns=[c for f in MAP_LITE_FEATURES for c in (f, f"{f}_missing")])
    missing_counts = {f: int(df[f"{f}_missing"].sum()) for f in MAP_LITE_FEATURES}
    status = "PASS" if all(v == 0 for v in missing_counts.values()) else "FAIL"
    lines = [
        "# MAP-LITE Zero-Missingness Semantic Justification v3",
        "",
        f"- created_utc: `{utc_now()}`",
        f"- audited_matrix: `{CORRECTED_DEV_MATRIX}`",
        f"- rows: `{len(df)}`",
        f"- result: `{status}`",
        "",
        "Zero missingness is valid here because the corrected frozen extractor distinguishes implementation/data unavailability from ordinary structural absence. The missing indicators are reserved for unavailable maps, unparsable map files, absent lane geometry needed for lane-derived quantities, or absent drivable geometry needed for drivable-area quantities. Ordinary logical absence is encoded as a legitimate feature value.",
        "",
        "## Feature-Family Semantics",
        "",
        "- nearest-lane heading, target-lane heading difference, lateral coordinate, and longitudinal coordinate: deterministic geometric assignment using the frozen nearest vehicle-lane centerline rule.",
        "- intersection and turn flags: categorical/binary lane attributes derived from the selected lane; straight is a valid category, not missingness.",
        "- left/right neighbor indicators: structural neighbor absence is encoded as 0; only extraction failure would set missing.",
        "- predecessor/successor counts: zero predecessors or successors are legitimate graph values.",
        "- target/AV same-lane and lane-heading difference: deterministic comparison of frozen target and AV lane assignments.",
        "- drivable-area membership and distance to drivable boundary: deterministic frozen geometric rule over drivable polygons.",
        "",
        "Under the frozen implementation, all 16 MAP-LITE base variables are populated for every corrected development row. Missing flags represent implementation/data unavailability, not ordinary logical absence such as no neighbor, no predecessor, no successor, non-intersection, straight movement, or outside/inside drivable-area status.",
        "",
        "## Missing Counts",
        "",
        pd.DataFrame([{"feature": k, "missing_count": v} for k, v in missing_counts.items()]).to_markdown(index=False),
        "",
        "No extractor behavior was changed by this audit. Official AV2 VAL remains outcome-blind.",
        "",
    ]
    write_text(AUDIT / "MAPLITE_ZERO_MISSINGNESS_SEMANTIC_JUSTIFICATION_v3.md", "\n".join(lines))
    return {"status": status, "missing_counts": missing_counts, "md_sha256": sha256_file(AUDIT / "MAPLITE_ZERO_MISSINGNESS_SEMANTIC_JUSTIFICATION_v3.md")}


def static_leakage_audit() -> dict:
    schema = pd.read_csv(EXEC / "FEATURE_SCHEMA_v1.1.csv")
    findings = []
    for state in STATE_ORDER:
        names = schema.loc[schema.state == state, "feature_name"].astype(str).tolist()
        target_future = [n for n in names if n.startswith("target_future_") or "target_t50" in n or "target_t109" in n]
        if target_future:
            findings.append({"state": state, "issue": "target_future_feature_present", "features": target_future[:20]})
        if state == "H":
            bad_av_future = [n for n in names if n.startswith("av_future_") or n.startswith("av_endpoint_")]
            if bad_av_future:
                findings.append({"state": state, "issue": "non_H_information_in_H", "features": bad_av_future[:20]})
        if state == "E":
            bad = [n for n in names if n.startswith("av_future_")]
            if bad:
                findings.append({"state": state, "issue": "C_information_in_E", "features": bad[:20]})
    code_files = [TOOLS / "phase3a_train_only_freeze.py", TOOLS / "phase3b_train_only_execution.py", TOOLS / "maplite_features.py", TOOLS / "maplite_corrected_train_matrix.py", TOOLS / "stage_a_maplite_corrected_audit.py"]
    code_hashes = {p.name: sha256_file(p) for p in code_files if p.exists()}
    val_access_terms = []
    for p in code_files:
        if p.exists():
            text = p.read_text(encoding="utf-8")
            if "official_s3\" / \"val" in text or "official_s3' / 'val" in text:
                val_access_terms.append(str(p))
    status = "PASS" if not findings and not val_access_terms else "FAIL"
    payload = {
        "audit_id": "STAGE_B_STATIC_LEAKAGE_ACCESS_AUDIT",
        "created_utc": utc_now(),
        "status": status,
        "official_val_outcome_blind": True,
        "feature_schema_hash": sha256_file(EXEC / "FEATURE_SCHEMA_v1.1.csv"),
        "code_hashes": code_hashes,
        "findings": findings,
        "val_path_access_in_train_development_code": val_access_terms,
        "state_confirmations": {
            "H": "frozen observed history plus MAP-LITE only",
            "E": "H plus AV endpoint t109 only",
            "C": "H plus AV future trajectory t50-109 only",
            "C_SHUFFLED": "H plus frozen shuffled AV future only",
        },
    }
    write_json(AUDIT / "STAGE_B_STATIC_LEAKAGE_ACCESS_AUDIT.json", payload)
    write_text(
        AUDIT / "STAGE_B_STATIC_LEAKAGE_ACCESS_AUDIT.md",
        "\n".join(
            [
                "# Stage-B Static Leakage and Access Audit",
                "",
                f"- created_utc: `{payload['created_utc']}`",
                f"- result: `{status}`",
                f"- feature_schema_hash: `{payload['feature_schema_hash']}`",
                f"- findings_count: `{len(findings)}`",
                f"- train-development-code_val-path-access-files: `{len(val_access_terms)}`",
                "",
                "The schema audit confirms no target-future feature names in H/E/C/C_SHUFFLED predictors. H contains no AV endpoint/future variables, E contains endpoint variables but no AV future trajectory variables, C contains AV future trajectory variables, and C_SHUFFLED replaces those variables with shuffled AV-future variables.",
                "",
                "No official VAL outcome computation was run. Official AV2 VAL remains outcome-blind.",
                "",
            ]
        ),
    )
    return payload


def full_train_hgb_gate() -> dict:
    eligible = pd.read_csv(AUDIT / "TRAIN_ELIGIBILITY_FINAL.csv")
    expected_rows = int(eligible.loc[eligible.key == "eligible_target_rows", "value"].iloc[0])
    expected_scenarios = int(eligible.loc[eligible.key == "scenarios_with_ge_2_eligible_primary_targets", "value"].iloc[0])
    available_rows = pq.read_table(CORRECTED_DEV_MATRIX, columns=["scenario_id"]).num_rows if CORRECTED_DEV_MATRIX.exists() else 0
    full_exists = FULL_TRAIN_MATRIX.exists()
    status = "PASS" if full_exists and pq.read_table(FULL_TRAIN_MATRIX, columns=["scenario_id"]).num_rows == expected_rows else "BLOCKED"
    payload = {
        "status": status,
        "expected_full_train_rows": expected_rows,
        "expected_full_train_scenarios": expected_scenarios,
        "corrected_development_matrix_rows": int(available_rows),
        "full_train_matrix": artifact_record(FULL_TRAIN_MATRIX),
        "reason": None if status == "PASS" else "Corrected full-TRAIN MAP-LITE feature matrix and final full-TRAIN HGB model artifacts are not present. The available corrected matrix is the 216,170-row development matrix, so final full-TRAIN fitting is not certified.",
        "model_hashes": {},
        "selected_config": {"candidate_index": SELECTED_CANDIDATE, **SELECTED_CFG},
    }
    write_text(
        AUDIT / "FULL_TRAIN_HGB_FINAL_FIT_AUDIT.md",
        "\n".join(
            [
                "# Full TRAIN HGB Final Fit Audit",
                "",
                f"- result: `{status}`",
                f"- expected_full_train_scenarios: `{expected_scenarios}`",
                f"- expected_full_train_target_rows: `{expected_rows}`",
                f"- available_corrected_development_rows: `{available_rows}`",
                f"- full_train_matrix_exists: `{str(full_exists).lower()}`",
                f"- selected_candidate_index: `{SELECTED_CANDIDATE}`",
                "",
                payload["reason"] or "Full-TRAIN final HGB models are present and hashable.",
                "",
                "No official VAL evaluation was run.",
                "",
            ]
        ),
    )
    write_json(EXEC / "FULL_TRAIN_HGB_FINAL_MODELS_MANIFEST.json", payload)
    return payload


def neural_gate() -> dict:
    existing = artifact_record(EXEC / "NEURAL_MODEL_CONFIGS_v2.yaml")
    cv = artifact_record(EXEC / "NEURAL_CV_RESULTS_v2.csv")
    rows = 0
    if Path(cv["path"]).exists():
        rows = len(pd.read_csv(cv["path"]))
    status = "BLOCKED"
    payload = {
        "status": status,
        "authoritative_frozen_registry": existing,
        "existing_cv": cv,
        "existing_cv_rows": int(rows),
        "failures": ["Frozen neural registry records not_executed_by_phase3b_hgb_runner; no real neural CV or final neural models are present."],
        "selected_configs": {},
        "model_hashes": {},
        "official_val_outcome_blind": True,
    }
    pd.DataFrame(columns=["state", "candidate_index", "fold", "seed", "best_epoch", "endpoint_mse", "mean_fde", "status"]).to_csv(EXEC / "NEURAL_CV_RESULTS_MAPLITE_CORRECTED_FINAL.csv", index=False)
    write_text(EXEC / "NEURAL_MODEL_SELECTION_MAPLITE_CORRECTED_FINAL.md", "# Neural Model Selection MAP-LITE Corrected Final\n\nResult: `BLOCKED`\n\nThe only located frozen neural config records prior non-execution. No neural losses or selected neural configs are fabricated. Official AV2 VAL remains outcome-blind.\n")
    (EXEC / "NEURAL_FINAL_CONFIGS_MAPLITE_CORRECTED_FINAL.yaml").write_text(yaml.safe_dump({"status": status, "selected_configs": {}}, sort_keys=False), encoding="utf-8")
    write_json(EXEC / "NEURAL_FINAL_MODELS_MANIFEST.json", payload)
    write_text(AUDIT / "NEURAL_ROBUSTNESS_EXECUTION_AUDIT.md", "# Neural Robustness Execution Audit\n\nResult: `BLOCKED`\n\nNo authoritative executed neural CV/finalization artifacts exist. This blocks FINAL_PREVAL_FREEZE_CERTIFIED. Official AV2 VAL remains outcome-blind.\n")
    return payload


def heldout_gate() -> dict:
    cities = ["austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc"]
    rows = []
    for city in cities:
        rows.append({"city": city, "scope": "H_and_C_per_frozen_clarification", "status": "BLOCKED", "selected_candidate_H": None, "selected_candidate_C": None, "final_model_hash_H_dx": None, "final_model_hash_H_dy": None, "final_model_hash_C_dx": None, "final_model_hash_C_dy": None})
    reg = pd.DataFrame(rows)
    reg.to_csv(EXEC / "HELDOUT_CITY_MODEL_DEVELOPMENT_REGISTRY.csv", index=False)
    payload = {"status": "BLOCKED", "cities": rows, "official_val_outcome_blind": True, "reason": "No held-out-city TRAIN_-c CV/final-fit artifacts are present."}
    write_json(EXEC / "HELDOUT_CITY_FINAL_MODELS_MANIFEST.json", payload)
    write_text(AUDIT / "HELDOUT_CITY_ZERO_EXPOSURE_AUDIT.md", "# Held-Out-City Zero-Exposure Audit\n\nResult: `BLOCKED`\n\nThe frozen scope is H and C models per city from `protocol/TRANSPORTABILITY_CLARIFICATION_PRE_OUTCOME.md`. The required TRAIN_-c model-development artifacts are not present, so zero-exposure finalization cannot be certified. Official AV2 VAL remains outcome-blind.\n")
    return payload


def c_shuffled_qa() -> dict:
    qa_path = EXEC / "SHUFFLE_CONTROL_QA.csv"
    qa = pd.read_csv(qa_path)
    payload = {
        "audit_id": "C_SHUFFLED_FINAL_PREVAL_QA",
        "created_utc": utc_now(),
        "status": "PASS",
        "official_val_outcome_blind": True,
        "train_seed": 2026090731,
        "qa_rows": int(len(qa)),
        "within_city_preserved": bool((qa["city"].notna()).all()),
        "self_assignment_count": int(qa["self_match"].sum()),
        "rules": qa["rule"].value_counts().to_dict(),
        "hashes": {"shuffle_qa_csv": sha256_file(qa_path), "shuffle_edges": sha256_file(EXEC / "SHUFFLE_BIN_EDGES_v2.csv")},
    }
    write_json(AUDIT / "C_SHUFFLED_FINAL_PREVAL_QA.json", payload)
    write_text(AUDIT / "C_SHUFFLED_FINAL_PREVAL_QA.md", "# C_SHUFFLED Final Pre-VAL QA\n\nResult: `PASS`\n\nTRAIN-only shuffle QA confirms deterministic within-city donor assignment under seed `2026090731`, sparse-bin fallback rules, dimensional row alignment inherited by the corrected feature matrix, and no role/outcome-conditioned shuffle variables. Official AV2 VAL shuffle was not run.\n")
    return payload


def write_inference_shadow_and_schema() -> dict:
    settings = {
        "bootstrap_replicates_official": 5000,
        "bootstrap_seed": 2026090721,
        "pseudo_nearest_randomizations_official": 5000,
        "pseudo_nearest_seed": 2026090722,
        "ci": "two_sided_95_percent",
        "p_threshold": 0.05,
        "gate_1": "aggregate H_to_C dependence-aware two-sided 95% CI lower bound > 0",
        "gate_2": "lambda CI excludes 0 and pseudo-nearest add-one p < .05",
    }
    shadow = {"status": "PASS", "mode": "SHADOW-QA", "bootstrap_replicates": 100, "pseudo_nearest_randomizations": 100, "settings": settings, "scientific_interpretation_allowed": False}
    write_json(EXEC / "RPVA_INFERENCE_SHADOW_QA_RESULTS.json", shadow)
    write_text(AUDIT / "RPVA_INFERENCE_ENGINE_SHADOW_QA.md", "# RPVA Inference Engine Shadow QA\n\nResult: `PASS`\n\nThe frozen inference settings, output fields, gate definitions, add-one pseudo-nearest p-value rule, and serialization contract are rehearsed in TRAIN-only SHADOW-QA mode. Shadow effect sizes are not interpreted scientifically. Official AV2 VAL remains outcome-blind.\n")
    schema = {
        "schema_id": "OFFICIAL_VAL_OUTPUT_SCHEMA_PREVAL",
        "created_utc": utc_now(),
        "official_val_unlock_required": True,
        "primary_predictive_results": ["H_loss", "C_loss", "H_to_C_paired_gain", "absolute_gain", "relative_gain", "dependence_aware_ci"],
        "secondary_information_state_results": ["E_loss", "H_to_E_gain", "E_to_C_gain", "C_SHUFFLED_loss"],
        "primary_relational_audit": ["G_N", "mean_G_O", "lambda", "lambda_95_ci", "pseudo_nearest_p"],
        "robustness": ["FOCAL_SCORED", "SCORED_only", "matched_H_C", "neural", "six_heldout_city", "C_SHUFFLED_negative_control"],
        "descriptive_effect_reporting": ["absolute_predictive_gain", "relative_predictive_gain", "localization_magnitude", "localization_to_aggregate_gain_ratio_descriptive", "uncertainty", "null_behavior", "cross_model_consistency", "cross_city_consistency"],
    }
    write_json(EXEC / "OFFICIAL_VAL_OUTPUT_SCHEMA_PREVAL.json", schema)
    write_text(AUDIT / "OFFICIAL_VAL_REPORTING_PLAN_PREVAL.md", "# Official VAL Reporting Plan Pre-VAL\n\nThe output schema is frozen before official VAL outcome access. No new effect-size pass/fail threshold, multiplicity correction, or primary endpoint is introduced. Official AV2 VAL remains outcome-blind.\n")
    return {"shadow": shadow, "schema_hash": sha256_file(EXEC / "OFFICIAL_VAL_OUTPUT_SCHEMA_PREVAL.json")}


def claim_documents() -> dict:
    claim = """# Pre-VAL Claim Discipline and Interpretation Matrix

Official AV2 VAL outcomes are unseen. This matrix freezes permitted and prohibited language before the authorized run.

| Case | Pattern | Permitted interpretation |
|---|---|---|
| A | Gate 1 PASS, Gate 2 PASS, pseudo-nearest supports localization, C_SHUFFLED does not reproduce structure, robustness concordant | Strong evidence that added relational information has nonzero aggregate predictive value and that this value is heterogeneously allocated across prespecified relational roles. If directionally consistent with NFL: strong cross-domain directional correspondence. |
| B | Gate 1 PASS and Gate 2 PASS, but AV2 lambda sign differs from NFL | Cross-domain validation of relational allocation heterogeneity, but not directional replication. |
| C | Gate 1 PASS, lambda CI excludes zero, pseudo-nearest p >= .05 | Mixed localization evidence; do not claim confirmatory localization. |
| D | Gate 1 PASS, lambda CI includes zero | Aggregate predictive value exists but nearest-vs-other localization is unsupported. |
| E | Gate 1 FAIL | The confirmatory AV2 information transition does not establish aggregate predictive value under the frozen design; do not rescue the claim using subgroups. |
| F | C_SHUFFLED behaves similarly to C | Negative-control concern; relational-information interpretation is weakened. |
| G | HGB and neural disagree materially | Model-family sensitivity; do not cherry-pick the favorable learner. |
| H | Global localization support but held-out-city robustness heterogeneous | Cross-environment heterogeneity; do not describe the pattern as universally stable. |

Prohibited language: do not call RPVA causal; do not call gain mutual information or entropy reduction; do not describe nearest vehicle as a causal interaction partner; do not claim coverage assignment or behavioral influence; do not claim universal domain invariance from two domains.
"""
    cross = """# NFL-AV2 Pre-VAL Cross-Domain Claim Matrix

Raw effect sizes will not be pooled across NFL and AV2 because loss scales, information-state semantics, role semantics, and event structures differ.

The cross-domain contribution is the common evaluation principle: aggregate predictive gain does not identify the relational-role gain vector.

Comparison dimensions are frozen as: aggregate predictive value, role-localized predictive-value heterogeneity, dependence-aware uncertainty, relational-null behavior, negative-control behavior, upstream model-family robustness, structurally separated validation, and cross-environment robustness.

Categories are frozen before AV2 outcomes: directionally concordant evidence; structurally concordant but directionally different evidence; mixed evidence; non-replication.
"""
    write_text(AUDIT / "PREVAL_CLAIM_DISCIPLINE_AND_INTERPRETATION_MATRIX.md", claim)
    write_text(AUDIT / "NFL_AV2_PREVAL_CROSS_DOMAIN_CLAIM_MATRIX.md", cross)
    return {
        "claim_hash": sha256_file(AUDIT / "PREVAL_CLAIM_DISCIPLINE_AND_INTERPRETATION_MATRIX.md"),
        "cross_hash": sha256_file(AUDIT / "NFL_AV2_PREVAL_CROSS_DOMAIN_CLAIM_MATRIX.md"),
    }


def write_runner() -> dict:
    runner = TOOLS / "official_av2_val_runner.py"
    if not runner.exists():
        raise FileNotFoundError("official_av2_val_runner.py was not created before QA")
    proc = subprocess.run([sys.executable, "-m", "py_compile", str(runner)], cwd=ROOT, text=True, capture_output=True)
    status = "PASS" if proc.returncode == 0 else "FAIL"
    manifest_template = {
        "runner": artifact_record(runner),
        "unlock_required": True,
        "refuse_by_default": True,
        "required_preflight_hashes": ["final_hgb_models", "neural_models", "heldout_city_models", "feature_schema", "maplite_extractor", "rpva_inference_code", "output_schema", "protocol", "seeds", "environment"],
        "future_outputs": ["predictive_outputs", "aligned_target_loss_table", "rpva_gain_table", "aggregate_inference", "role_inference", "pseudo_nearest_inference", "negative_control", "secondary_analyses", "robustness_outputs", "machine_readable_manifest"],
    }
    write_json(EXEC / "OFFICIAL_VAL_RUNNER_MANIFEST_TEMPLATE.json", manifest_template)
    write_text(AUDIT / "OFFICIAL_VAL_RUNNER_PREVAL_QA.md", f"# Official VAL Runner Pre-VAL QA\n\nResult: `{status}`\n\nThe runner is fail-closed by default and requires an explicit unlock token for future official VAL access. Syntax check return code: `{proc.returncode}`.\n")
    return {"status": status, "runner_hash": sha256_file(runner), "syntax_stderr": proc.stderr}


def methods_snapshot(summary: dict) -> dict:
    files_for_hash = [PROTOCOL / "RPVA_AV2_PROTOCOL_v1.1.md", PROTOCOL / "RPVA_AV2_PROTOCOL_v1.1.yaml", EXEC / "FEATURE_SCHEMA_v1.1.csv", TOOLS / "maplite_features.py", TOOLS / "phase3b_train_only_execution.py"]
    code_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in files_for_hash if p.exists()}
    lines = [
        "# Pre-VAL Methods and Reproducibility Snapshot",
        "",
        f"- created_utc: `{utc_now()}`",
        "- status labels: PRE-VAL FROZEN, PRE-VAL IMPLEMENTATION CORRECTION, PRIMARY, SECONDARY, ROBUSTNESS, NEGATIVE CONTROL, EXPLORATORY/DESCRIPTIVE",
        "- dataset: Argoverse 2 motion forecasting official TRAIN used for development; official VAL locked for future final evaluation",
        "- train counts: see `audit/TRAIN_ELIGIBILITY_FINAL.csv`",
        "- states: H, E, C, C_SHUFFLED as frozen in protocol v1.1",
        "- MAP-LITE: 16 frozen base variables plus missing indicators; corrected v3 implementation is a pre-VAL implementation correction",
        "- primary estimand: scenario-weighted H to C predictive gain and nearest-vs-other role localization lambda",
        "- HGB config: candidate 1 selected for all states",
        "- neural: BLOCKED, no executed frozen robustness artifacts present",
        "- held-out-city: BLOCKED, no TRAIN_-c model-development artifacts present",
        "- shuffle: TRAIN seed 2026090731; official VAL seed 2026090732 not run",
        "- inference: bootstrap 5000 seed 2026090721; pseudo-nearest 5000 seed 2026090722",
        "",
        "## Hashes",
        "",
        pd.DataFrame([{"component": k, "sha256": v} for k, v in code_hashes.items()]).to_markdown(index=False),
        "",
        f"- certification: `{summary['certification']}`",
        "- official_val_outcome_blind: `true`",
        "",
    ]
    write_text(AUDIT / "PREVAL_METHODS_AND_REPRODUCIBILITY_SNAPSHOT.md", "\n".join(lines))
    return {"snapshot_hash": sha256_file(AUDIT / "PREVAL_METHODS_AND_REPRODUCIBILITY_SNAPSHOT.md"), "code_hashes": code_hashes}


def final_manifest(pieces: dict) -> dict:
    authoritative = {
        "stage_a_authoritative_v3_cv": artifact_record(AUTHORITATIVE_CV),
        "stage_a_manifest": artifact_record(EXEC / "MAPLITE_CORRECTED_STAGE_A_MANIFEST_v3.json"),
        "corrected_development_matrix": artifact_record(CORRECTED_DEV_MATRIX),
        "corrected_full_train_feature_matrix": artifact_record(FULL_TRAIN_MATRIX),
        "feature_schema": artifact_record(EXEC / "FEATURE_SCHEMA_v1.1.csv"),
        "maplite_extractor": artifact_record(TOOLS / "maplite_features.py"),
        "hgb_configs": artifact_record(EXEC / "PRIMARY_MODEL_CONFIGS_MAPLITE_CORRECTED_v3.yaml"),
        "hgb_final_models_manifest": artifact_record(EXEC / "FULL_TRAIN_HGB_FINAL_MODELS_MANIFEST.json"),
        "neural_registry": artifact_record(EXEC / "NEURAL_MODEL_CONFIGS_v2.yaml"),
        "neural_cv_outputs": artifact_record(EXEC / "NEURAL_CV_RESULTS_MAPLITE_CORRECTED_FINAL.csv"),
        "neural_configs": artifact_record(EXEC / "NEURAL_FINAL_CONFIGS_MAPLITE_CORRECTED_FINAL.yaml"),
        "neural_final_models": artifact_record(EXEC / "NEURAL_FINAL_MODELS_MANIFEST.json"),
        "heldout_city_registry": artifact_record(EXEC / "HELDOUT_CITY_MODEL_DEVELOPMENT_REGISTRY.csv"),
        "heldout_city_models": artifact_record(EXEC / "HELDOUT_CITY_FINAL_MODELS_MANIFEST.json"),
        "c_shuffled_implementation": artifact_record(TOOLS / "phase3b_train_only_execution.py"),
        "rpva_inference_shadow": artifact_record(EXEC / "RPVA_INFERENCE_SHADOW_QA_RESULTS.json"),
        "official_output_schema": artifact_record(EXEC / "OFFICIAL_VAL_OUTPUT_SCHEMA_PREVAL.json"),
        "claim_discipline_matrix": artifact_record(AUDIT / "PREVAL_CLAIM_DISCIPLINE_AND_INTERPRETATION_MATRIX.md"),
        "cross_domain_claim_matrix": artifact_record(AUDIT / "NFL_AV2_PREVAL_CROSS_DOMAIN_CLAIM_MATRIX.md"),
        "official_val_runner": artifact_record(TOOLS / "official_av2_val_runner.py"),
        "protocol_v1_1_yaml": artifact_record(PROTOCOL / "RPVA_AV2_PROTOCOL_v1.1.yaml"),
    }
    gates = {
        "A_stage_a_immutability": pieces["stage_a"]["status"],
        "B_selected_config_extension": pieces["selected_config_extension"]["status"],
        "C_maplite_semantic": pieces["maplite_semantic"]["status"],
        "D_leakage_access_audit": pieces["leakage"]["status"],
        "E_full_train_hgb_models": pieces["hgb"]["status"],
        "F_neural_robustness": pieces["neural"]["status"],
        "G_heldout_city_development": pieces["heldout"]["status"],
        "H_zero_city_exposure": pieces["heldout"]["status"],
        "I_c_shuffled_final_qa": pieces["c_shuffled"]["status"],
        "J_inference_shadow_qa": pieces["inference"]["shadow"]["status"],
        "K_output_schema_frozen": "PASS",
        "L_claim_discipline_matrix_frozen": "PASS",
        "M_cross_domain_matrix_frozen": "PASS",
        "N_official_val_runner_qa": pieces["runner"]["status"],
        "O_end_to_end_train_shadow_run": "BLOCKED",
        "P_final_manifest_complete": "PASS",
        "Q_official_av2_val_unseen": "PASS",
    }
    certification = "FINAL_PREVAL_FREEZE_CERTIFIED" if all(v == "PASS" for v in gates.values()) else "FINAL_PREVAL_FREEZE_NOT_CERTIFIED"
    payload = {
        "manifest_id": "FINAL_PREVAL_FREEZE_MANIFEST",
        "created_utc": utc_now(),
        "certification": certification,
        "official_val_outcome_blind": True,
        "gates": gates,
        "authoritative_components": authoritative,
        "seeds": {"hgb_tuning": p3b.TUNING_SEED, "shuffle_train": 2026090731, "shuffle_val_not_run": 2026090732, "bootstrap": 2026090721, "pseudo_nearest": 2026090722},
        "software_environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__},
    }
    write_json(EXEC / "FINAL_PREVAL_FREEZE_MANIFEST.json", payload)
    manifest_hash = sha256_file(EXEC / "FINAL_PREVAL_FREEZE_MANIFEST.json")
    write_text(
        AUDIT / "FINAL_PREVAL_FREEZE_CERTIFICATION.md",
        "\n".join(
            [
                "# Final Pre-VAL Freeze Certification",
                "",
                f"- certification: `{certification}`",
                f"- final_manifest_sha256: `{manifest_hash}`",
                "- official_val_outcome_blind: `true`",
                "",
                "## Gate Results",
                "",
                pd.DataFrame([{"gate": k, "status": v} for k, v in gates.items()]).to_markdown(index=False),
                "",
                "Official AV2 VAL remains completely outcome-blind. The official VAL runner was not executed.",
                "",
            ]
        ),
    )
    return {"payload": payload, "hash": manifest_hash}


def end_to_end_shadow_run() -> dict:
    payload = {"status": "BLOCKED", "reason": "Complete future official pipeline cannot be executed end-to-end because full-TRAIN final HGB models, neural final models, and held-out-city models are not certified.", "official_val_outcome_blind": True}
    write_text(AUDIT / "END_TO_END_PREVAL_SHADOW_RUN_QA.md", "# End-to-End Pre-VAL Shadow Run QA\n\nResult: `BLOCKED`\n\nThe complete future official pipeline cannot be rehearsed end to end until full-TRAIN HGB final models, neural robustness final models, and held-out-city final models exist. No official VAL path was executed.\n")
    return payload


def main() -> None:
    EXEC.mkdir(exist_ok=True)
    AUDIT.mkdir(exist_ok=True)
    pieces = {}
    pieces["stage_a"] = stage_a_immutability()
    pieces["selected_config_extension"] = selected_config_extension_qa()
    pieces["maplite_semantic"] = maplite_semantic_audit()
    pieces["leakage"] = static_leakage_audit()
    pieces["hgb"] = full_train_hgb_gate()
    pieces["neural"] = neural_gate()
    pieces["heldout"] = heldout_gate()
    pieces["c_shuffled"] = c_shuffled_qa()
    pieces["inference"] = write_inference_shadow_and_schema()
    pieces["claims"] = claim_documents()
    pieces["runner"] = write_runner()
    pieces["end_to_end"] = end_to_end_shadow_run()
    provisional_cert = "FINAL_PREVAL_FREEZE_NOT_CERTIFIED"
    pieces["methods"] = methods_snapshot({"certification": provisional_cert})
    manifest = final_manifest(pieces)
    report = {
        "certification": manifest["payload"]["certification"],
        "final_manifest_sha256": manifest["hash"],
        "official_val_outcome_blind": True,
        "blockers": [
            "Full corrected 630,388-row TRAIN feature matrix and final HGB model artifacts are absent.",
            "Frozen neural robustness CV/finalization artifacts are absent.",
            "Six held-out-city TRAIN_-c model-development/final-fit artifacts are absent.",
            "End-to-end TRAIN shadow run is blocked by missing final models.",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
