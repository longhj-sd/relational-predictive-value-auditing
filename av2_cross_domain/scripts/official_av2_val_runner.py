from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import phase3a_train_only_freeze as p3a
import phase3b_train_only_execution as p3b


ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "execution"
AUDIT = ROOT / "audit"
SHADOW = EXEC / "shadow_official_val_rehearsal_v4"
CHECKPOINTS = EXEC / "official_val_v4_checkpoints"
MODELS_DIR = EXEC / "full_train_hgb_models_v4"
REQUIRED_UNLOCK_TOKEN = "HUMAN_AUTHORIZED_OFFICIAL_AV2_VAL_RUN_AFTER_FINAL_PREVAL_FREEZE_REVIEW"
EXPECTED_ENV = {
    "python": "3.12.3",
    "numpy": "2.4.3",
    "pandas": "3.0.1",
    "pyarrow": "24.0.0",
    "sklearn": "1.8.0",
    "yaml": "6.0.3",
}
EXPECTED_ENV_HASH = "c27192539c77751658ee66759843e5bc3142063823ca13ef18e6e287a345150f"
EXPECTED_MATRIX_HASH = "b2f212698f0a1773597b8eff4341787e95bab2a20f12c71215d95f7b26c4a9c5"
EXPECTED_REGISTRY_HASH = "e32035d6d56f74163402020df748a10f470e28e87d5475826123b152f75b1e20"
OLD_STUB_HASH = "ad18bf08180b160295c145e09ff9466e8ea8ab089eb6851300a25fbdc3e83946"
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 2026090721
PSEUDO_NEAREST_REPS = 5000
PSEUDO_NEAREST_SEED = 2026090722
VAL_SHUFFLE_SEED = 2026090732
SHADOW_BOOTSTRAP_REPS = 25
SHADOW_PSEUDO_NEAREST_REPS = 25
RUN_ORDINAL = 1
STATE_ORDER = ["H", "E", "C", "C_SHUFFLED"]
TARGET_COLUMNS = ["dx_6s", "dy_6s"]
STAGES = [
    "00_PRE_READ_LOCK",
    "01_VAL_POPULATION",
    "02_VAL_FEATURES",
    "03_PREDICTIONS",
    "04_PRIMARY_AGGREGATE",
    "05_RPVA_SCENARIO",
    "06_BOOTSTRAP",
    "07_PSEUDONEAREST",
    "08_SECONDARY",
    "09_FINALIZE",
]
VAL_FORBIDDEN = (ROOT / "data" / "raw" / "av2_motion_forecasting" / "official_s3" / "val").resolve()
TRAIN_ROOT = (ROOT / "data" / "raw" / "av2_motion_forecasting" / "official_s3" / "train").resolve()
STATUS_PATH = EXEC / "OFFICIAL_VAL_V4_STATUS.json"
SENTINEL_PATH = AUDIT / "OFFICIAL_VAL_RUN_START_SENTINEL_v4_ORDINAL_1.json"
COMPLETION_CERT_JSON = AUDIT / "OFFICIAL_VAL_RUN_COMPLETION_CERTIFICATE_v4.json"


class ValAccessFirewall(RuntimeError):
    pass


class FrozenEstimatorProxy:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.model = payload["model"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def fit(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("model fitting is forbidden in official VAL execution")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def guard_not_val(path: Path, *, official_access_granted: bool = False) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(VAL_FORBIDDEN)
    except ValueError:
        return
    if not official_access_granted:
        raise ValAccessFirewall(f"official VAL path access blocked before run-start sentinel: {resolved}")


def sha256_file(path: Path, *, official_access_granted: bool = False) -> str:
    guard_not_val(path, official_access_granted=official_access_granted)
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


def observed_environment() -> dict[str, str]:
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "sklearn": sklearn.__version__,
        "yaml": yaml.__version__,
    }


def write_status(**updates: Any) -> None:
    previous: dict[str, Any] = {}
    if STATUS_PATH.exists():
        previous = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    payload = {
        "phase": "not_started",
        "process_state": "idle",
        "run_ordinal": None,
        "official_val_accessed": False,
        "official_val_result_observed": False,
        "eligible_scenarios": None,
        "eligible_targets": None,
        "current_stage": None,
        "completed_stage_count": 0,
        "total_stage_count": len(STAGES),
        "bootstrap_completed": 0,
        "bootstrap_total": BOOTSTRAP_REPS,
        "pseudo_nearest_completed": 0,
        "pseudo_nearest_total": PSEUDO_NEAREST_REPS,
        "elapsed": None,
        "failed_stage": None,
        "latest_checkpoint": None,
    }
    payload.update(previous)
    payload.update(updates)
    payload["updated_utc"] = utc_now()
    atomic_json(STATUS_PATH, payload)


def load_json(path: Path) -> dict[str, Any]:
    guard_not_val(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_model_manifest() -> dict[str, Any]:
    manifest = load_json(EXEC / "FULL_TRAIN_HGB_FINAL_MODELS_MANIFEST_v4.json")
    if manifest.get("input_matrix_sha256") != EXPECTED_MATRIX_HASH:
        raise SystemExit("Refusing official VAL access: final model manifest matrix hash mismatch.")
    if manifest.get("candidate_registry_hash") != EXPECTED_REGISTRY_HASH:
        raise SystemExit("Refusing official VAL access: final model manifest registry hash mismatch.")
    if manifest.get("scientific_environment_hash") != EXPECTED_ENV_HASH:
        raise SystemExit("Refusing official VAL access: final model manifest environment hash mismatch.")
    if int(manifest.get("completed_model_units", 0)) != 8:
        raise SystemExit("Refusing official VAL access: final model manifest does not contain all 8 model units.")
    for rec in manifest.get("models", []):
        model_path = MODELS_DIR / rec["model_filename"]
        meta_path = MODELS_DIR / rec["metadata_filename"]
        if sha256_file(model_path) != rec.get("model_sha256"):
            raise SystemExit(f"Refusing official VAL access: model hash mismatch for {model_path.name}.")
        if sha256_file(meta_path) != rec.get("metadata_sha256"):
            raise SystemExit(f"Refusing official VAL access: metadata hash mismatch for {meta_path.name}.")
    return manifest


def validate_lock_candidate() -> dict[str, Any]:
    lock = load_json(AUDIT / "PREVAL_LOCK_CANDIDATE_v4.json")
    if lock.get("technical_preval_ready") is not True:
        raise SystemExit("Refusing official VAL access: technical_preval_ready is not true.")
    if lock.get("human_authorized") is not False:
        raise SystemExit("Refusing official VAL access: lock candidate must not pre-authorize the run.")
    if lock.get("official_val_accessed") is not False:
        raise SystemExit("Refusing official VAL access: lock candidate indicates prior VAL access.")
    env = observed_environment()
    if env != EXPECTED_ENV:
        raise SystemExit(f"Refusing official VAL access: scientific environment mismatch: {env}")
    return {"lock": lock, "manifest": validate_model_manifest(), "environment": env}


def validate_successor_lock_if_present() -> dict[str, Any] | None:
    path = AUDIT / "PREVAL_FINAL_LOCK_v4_1.json"
    if not path.exists():
        return None
    lock = load_json(path)
    if lock.get("technical_preval_ready") is not True:
        raise SystemExit("Refusing official VAL access: successor technical_preval_ready is not true.")
    if lock.get("human_authorized") is not False:
        raise SystemExit("Refusing official VAL access: successor lock must not contain transferred human authorization.")
    if int(lock.get("authorized_official_val_runs", -1)) != 0:
        raise SystemExit("Refusing official VAL access: successor lock must reset authorized official runs to zero.")
    if lock.get("official_val_accessed") is not False or lock.get("official_val_result_observed") is not False:
        raise SystemExit("Refusing official VAL access: successor lock records official VAL access.")
    expected_runner = lock.get("hashes", {}).get("new_official_runner")
    current_runner = sha256_file(Path(__file__).resolve())
    if expected_runner and expected_runner != current_runner:
        raise SystemExit(f"Refusing official VAL access: successor lock runner hash mismatch: {current_runner}")
    return lock


def preflight(unlock_token: str | None, *, require_unlock: bool) -> dict[str, Any]:
    certified = validate_lock_candidate()
    successor_lock = validate_successor_lock_if_present()
    freeze = load_json(EXEC / "FINAL_PREVAL_FREEZE_MANIFEST.json")
    if require_unlock and unlock_token != REQUIRED_UNLOCK_TOKEN:
        raise SystemExit("Refusing official VAL access: explicit human unlock token is required.")
    successor_certified = successor_lock is not None and successor_lock.get("technical_preval_ready") is True
    if freeze.get("certification") != "FINAL_PREVAL_FREEZE_CERTIFIED" and not successor_certified:
        raise SystemExit(f"Refusing official VAL access: certification is {freeze.get('certification')!r}.")
    if freeze.get("official_val_outcome_blind") is not True:
        raise SystemExit("Refusing official VAL access: manifest does not assert outcome-blind status.")
    failed = {k: v for k, v in freeze.get("gates", {}).items() if v != "PASS"}
    if failed and not successor_certified:
        raise SystemExit(f"Refusing official VAL access: non-PASS gates remain: {failed}")
    missing = [name for name, rec in freeze.get("authoritative_components", {}).items() if not rec.get("exists") or not rec.get("sha256")]
    if missing and not successor_certified:
        raise SystemExit(f"Refusing official VAL access: missing authoritative components: {missing}")
    return {**certified, "freeze_manifest": freeze, "successor_lock": successor_lock}


def checkpoint(stage: str, payload: dict[str, Any], run_context: dict[str, str]) -> Path:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "created_utc": utc_now(),
        "run_ordinal": RUN_ORDINAL,
        **run_context,
        "stage_input_hashes": payload.pop("stage_input_hashes", {}),
        "payload_hash": sha256_json(payload),
        "payload": payload,
    }
    path = CHECKPOINTS / f"{stage}.json"
    atomic_json(path, record)
    write_status(
        phase="running",
        process_state="running",
        run_ordinal=RUN_ORDINAL,
        current_stage=stage,
        completed_stage_count=STAGES.index(stage) + 1,
        latest_checkpoint=str(path),
        official_val_accessed=STAGES.index(stage) >= 1,
    )
    return path


def create_run_start_sentinel(*, resume: bool) -> dict[str, str]:
    if COMPLETION_CERT_JSON.exists() or (AUDIT / "OFFICIAL_VAL_CONFIRMATORY_RESULT_v4.json").exists():
        raise SystemExit("Refusing official VAL access: prior official completion artifact exists.")
    if SENTINEL_PATH.exists():
        if not resume:
            raise SystemExit("Refusing fresh official launch: run-start sentinel exists; use --resume for same-run continuation only.")
        payload = load_json(SENTINEL_PATH)
        if payload.get("run_ordinal") != RUN_ORDINAL:
            raise SystemExit("Refusing official VAL access: run-start sentinel ordinal mismatch.")
        return {"sentinel_hash": sha256_file(SENTINEL_PATH), "sentinel_path": str(SENTINEL_PATH)}
    payload = {
        "artifact_id": "OFFICIAL_VAL_RUN_START_SENTINEL_v4",
        "created_utc": utc_now(),
        "run_ordinal": RUN_ORDINAL,
        "official_val_run_start": True,
        "human_authorized": True,
        "unlock_token_accepted": True,
        "pre_val_lock_candidate_sha256": sha256_file(AUDIT / "PREVAL_LOCK_CANDIDATE_v4.json"),
        "final_preval_lock_sha256": sha256_file(AUDIT / "PREVAL_FINAL_LOCK_v4.json") if (AUDIT / "PREVAL_FINAL_LOCK_v4.json").exists() else None,
        "final_models_manifest_sha256": sha256_file(EXEC / "FULL_TRAIN_HGB_FINAL_MODELS_MANIFEST_v4.json"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "environment_sha256": EXPECTED_ENV_HASH,
        "prohibitions": ["no hyperparameter tuning", "no model refitting", "no feature redesign", "no gate changes", "no repeated confirmatory reruns"],
    }
    atomic_json(SENTINEL_PATH, payload)
    return {"sentinel_hash": sha256_file(SENTINEL_PATH), "sentinel_path": str(SENTINEL_PATH)}


def validate_existing_checkpoints(run_context: dict[str, str]) -> None:
    for path in sorted(CHECKPOINTS.glob("*.json")):
        rec = load_json(path)
        if rec.get("run_ordinal") != RUN_ORDINAL:
            raise SystemExit(f"Refusing resume: checkpoint ordinal mismatch in {path.name}.")
        for key, expected in run_context.items():
            if rec.get(key) != expected:
                raise SystemExit(f"Refusing resume: checkpoint {path.name} {key} mismatch.")


def load_models(manifest: dict[str, Any]) -> dict[tuple[str, str], FrozenEstimatorProxy]:
    models: dict[tuple[str, str], FrozenEstimatorProxy] = {}
    for rec in manifest["models"]:
        with (MODELS_DIR / rec["model_filename"]).open("rb") as f:
            payload = pickle.load(f)
        models[(str(rec["state"]), str(rec["coordinate"]))] = FrozenEstimatorProxy(payload)
    expected = {(s, t) for s in STATE_ORDER for t in TARGET_COLUMNS}
    if set(models) != expected:
        raise RuntimeError("loaded model set is not the frozen 4x2 official set")
    return models


def scenario_parquets(root: Path, *, official_access_granted: bool) -> list[Path]:
    guard_not_val(root, official_access_granted=official_access_granted)
    return sorted(root.glob("*/scenario_*.parquet"))


def population_from_root(root: Path, *, official_access_granted: bool, limit: int | None = None) -> pd.DataFrame:
    files = scenario_parquets(root, official_access_granted=official_access_granted)
    if limit is not None:
        files = files[:limit]
    rows = []
    for path in files:
        guard_not_val(path, official_access_granted=official_access_granted)
        rows.append(p3a.inspect_scenario(path).__dict__)
    df = pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("no scenarios available for population construction")
    return df[df["eligible_primary_targets"] >= 2].copy()


def val_shuffle(pop: pd.DataFrame, futures: dict[str, np.ndarray], seed: int) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
    sample = pop[["scenario_id", "city", "av_speed_49", "av_yaw_rate_45_49"]].copy()
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
    exact: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    city_speed: dict[tuple[str, int], list[str]] = defaultdict(list)
    city_only: dict[str, list[str]] = defaultdict(list)
    for r in sample.itertuples(index=False):
        exact[(r.city, int(r.speed_bin), int(r.yaw_bin))].append(r.scenario_id)
        city_speed[(r.city, int(r.speed_bin))].append(r.scenario_id)
        city_only[r.city].append(r.scenario_id)
    for d in [exact, city_speed, city_only]:
        for key in d:
            d[key] = sorted(d[key])
    donors = {}
    qa_rows = []
    for row in sample.sort_values("scenario_id").itertuples(index=False):
        same = exact[(row.city, int(row.speed_bin), int(row.yaw_bin))]
        rule = "exact_city_speed_yaw"
        if len(same) < 2:
            same = city_speed[(row.city, int(row.speed_bin))]
            rule = "merge_yaw_rate_bin"
        if len(same) < 2:
            same = city_only[row.city]
            rule = "city_only"
        candidates = [sid for sid in same if sid != row.scenario_id]
        self_match = 0
        if candidates:
            donor = candidates[int(hashlib.sha256(f"{seed}:{row.scenario_id}".encode()).hexdigest(), 16) % len(candidates)]
        else:
            donor = row.scenario_id
            self_match = 1
        donors[row.scenario_id] = donor
        qa_rows.append({"scenario_id": row.scenario_id, "city": row.city, "speed_bin": int(row.speed_bin), "yaw_bin": int(row.yaw_bin), "donor_scenario_id": donor, "rule": rule, "self_match": self_match})
    if set(donors) - set(futures):
        raise RuntimeError("shuffle donor universe and future registry mismatch")
    return donors, edges, pd.DataFrame(qa_rows)


def build_features_from_population(root: Path, pop: pd.DataFrame, *, official_access_granted: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_root = p3b.TRAIN_ROOT
    p3b.TRAIN_ROOT = root
    try:
        rows: list[dict[str, Any]] = []
        futures: dict[str, np.ndarray] = {}
        for r in pop.sort_values("scenario_id").itertuples(index=False):
            guard_not_val(root / r.scenario_id / f"scenario_{r.scenario_id}.parquet", official_access_granted=official_access_granted)
            out_rows, av_future = p3b.scenario_features(str(r.scenario_id), str(r.city), 0)
            rows.extend(out_rows)
            futures[str(r.scenario_id)] = av_future
    finally:
        p3b.TRAIN_ROOT = old_root
    df = pd.DataFrame(rows).sort_values(["scenario_id", "track_id"]).reset_index(drop=True)
    donors, edges, qa = val_shuffle(pop, futures, VAL_SHUFFLE_SEED)
    shuffled_cols = []
    for t in p3a.FUTURE_STEPS:
        shuffled_cols.extend([f"shuffled_av_future_x_{t}_fixed", f"shuffled_av_future_y_{t}_fixed", f"shuffled_av_future_vx_{t}_fixed", f"shuffled_av_future_vy_{t}_fixed"])
    shuffled_matrix = np.vstack([futures[donors[sid]].astype(np.float32, copy=False) for sid in df["scenario_id"].astype(str)])
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(shuffled_matrix, columns=shuffled_cols)], axis=1), edges, qa


def predict_losses(features: pd.DataFrame, manifest: dict[str, Any], out_path: Path) -> pd.DataFrame:
    models = load_models(manifest)
    records = features[["scenario_id", "city", "track_id", "is_nearest", "dx_6s", "dy_6s"]].copy()
    y = features[TARGET_COLUMNS].to_numpy(np.float32)
    for state in STATE_ORDER:
        names = p3b.feature_names_for_state(state)
        x = features[names].astype(np.float32).to_numpy(np.float32)
        pred = np.column_stack([models[(state, "dx_6s")].predict(x), models[(state, "dy_6s")].predict(x)]).astype(np.float32)
        records[f"{state}_pred_dx_6s"] = pred[:, 0]
        records[f"{state}_pred_dy_6s"] = pred[:, 1]
        records[f"{state}_loss"] = np.linalg.norm(pred - y, axis=1)
        records[f"{state}_endpoint_mse"] = np.sum((pred - y) ** 2, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(records, preserve_index=False), out_path)
    return records


def scenario_results(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in pred.groupby("scenario_id", sort=True):
        gains = (g["H_loss"] - g["C_loss"]).to_numpy(float)
        n_mask = g["is_nearest"].astype(int).to_numpy() == 1
        o_mask = ~n_mask
        rows.append({
            "scenario_id": sid,
            "city": str(g["city"].iloc[0]),
            "eligible_targets": int(len(g)),
            "g_N": float(gains[n_mask][0]),
            "mean_g_O": float(gains[o_mask].mean()),
            "lambda": float(gains[n_mask][0] - gains[o_mask].mean()),
            "delta_H_to_C": float(gains.mean()),
            "delta_H_to_E": float((g["H_loss"] - g["E_loss"]).mean()),
            "delta_H_to_C_SHUFFLED": float((g["H_loss"] - g["C_SHUFFLED_loss"]).mean()),
        })
    return pd.DataFrame(rows)


def ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap(sr: pd.DataFrame, reps: int, seed: int, status_prefix: str) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    deltas = sr["delta_H_to_C"].to_numpy(float)
    lambdas = sr["lambda"].to_numpy(float)
    n = len(sr)
    b_delta = np.empty(reps, dtype=float)
    b_lambda = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        b_delta[i] = float(deltas[idx].mean())
        b_lambda[i] = float(lambdas[idx].mean())
        if (i + 1) % 100 == 0 or i + 1 == reps:
            write_status(**{f"{status_prefix}_completed": i + 1})
    return {"replicates": reps, "seed": seed, "delta_distribution": b_delta, "lambda_distribution": b_lambda, "delta_ci": ci(b_delta), "lambda_ci": ci(b_lambda)}


def pseudo_nearest(pred: pd.DataFrame, reps: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    observed = float(scenario_results(pred)["lambda"].mean())
    nulls = np.empty(reps, dtype=float)
    grouped = [(sid, g.reset_index(drop=True)) for sid, g in pred.groupby("scenario_id", sort=True)]
    for i in range(reps):
        vals = []
        for _sid, g in grouped:
            gains = (g["H_loss"] - g["C_loss"]).to_numpy(float)
            j = int(rng.integers(0, len(gains)))
            vals.append(float(gains[j] - np.delete(gains, j).mean()))
        nulls[i] = float(np.mean(vals))
        if (i + 1) % 100 == 0 or i + 1 == reps:
            write_status(pseudo_nearest_completed=i + 1)
    extreme = int((np.abs(nulls) >= abs(observed)).sum())
    return {"replicates": reps, "seed": seed, "lambda_observed": observed, "lambda_null": nulls, "extreme_count": extreme, "p_value": float((extreme + 1) / (reps + 1))}


def summarize(pred: pd.DataFrame, sr: pd.DataFrame, boot: dict[str, Any], pseudo: dict[str, Any]) -> dict[str, Any]:
    gate1 = bool(boot["delta_ci"][0] > 0.0)
    lambda_excludes_zero = bool(boot["lambda_ci"][0] > 0.0 or boot["lambda_ci"][1] < 0.0)
    pseudo_pass = bool(pseudo["p_value"] < 0.05)
    gate2 = bool(lambda_excludes_zero and pseudo_pass)
    disposition = "CONFIRMATORY" if gate1 and gate2 else "MIXED" if gate1 and (lambda_excludes_zero != pseudo_pass) else "AGGREGATE_ONLY" if gate1 else "GATE_1_FAIL"
    return {
        "delta_H_to_C": float(sr["delta_H_to_C"].mean()),
        "delta_H_to_C_ci95": list(boot["delta_ci"]),
        "lambda": float(sr["lambda"].mean()),
        "lambda_ci95": list(boot["lambda_ci"]),
        "pseudo_nearest_p": pseudo["p_value"],
        "gate_1": gate1,
        "gate_2": gate2,
        "lambda_ci_excludes_zero": lambda_excludes_zero,
        "gate_2_requires_lambda_positive": False,
        "pseudo_nearest_rule": "|lambda_null| >= |lambda_obs|, add-one two-sided",
        "gate_disagreement_disposition": "MIXED",
        "confirmatory_disposition": disposition,
        "eligible_scenarios": int(pred["scenario_id"].nunique()),
        "eligible_targets": int(len(pred)),
        "secondary": {
            "delta_H_to_E": float(sr["delta_H_to_E"].mean()),
            "delta_H_to_C_SHUFFLED": float(sr["delta_H_to_C_SHUFFLED"].mean()),
            "c_minus_c_shuffled_gain_difference": float((sr["delta_H_to_C"] - sr["delta_H_to_C_SHUFFLED"]).mean()),
        },
    }


def write_final_artifacts(summary: dict[str, Any], sr: pd.DataFrame, pred_path: Path, pseudo: dict[str, Any], boot: dict[str, Any], run_context: dict[str, str]) -> dict[str, str]:
    sr_path = EXEC / "OFFICIAL_VAL_RPVA_SCENARIO_RESULTS_v4.parquet"
    pq.write_table(pa.Table.from_pandas(sr, preserve_index=False), sr_path)
    json_hash = atomic_json(EXEC / "OFFICIAL_VAL_PRIMARY_RESULTS_v4.json", summary)
    pd.DataFrame([{**summary, "secondary": json.dumps(summary["secondary"], sort_keys=True)}]).to_csv(EXEC / "OFFICIAL_VAL_PRIMARY_RESULTS_v4.csv", index=False)
    np.savez_compressed(EXEC / "OFFICIAL_VAL_PSEUDONEAREST_v4.npz", lambda_null=pseudo["lambda_null"], lambda_observed=np.asarray([pseudo["lambda_observed"]]), p_value=np.asarray([pseudo["p_value"]]), bootstrap_delta=boot["delta_distribution"], bootstrap_lambda=boot["lambda_distribution"])
    audit_payload = {"artifact_id": "OFFICIAL_VAL_EXECUTION_AUDIT_v4", "created_utc": utc_now(), **run_context, "status": "PASS", "prediction_artifact": str(pred_path), "scenario_results_artifact": str(sr_path), "primary_results_sha256": json_hash, "official_val_accessed": True, "official_val_result_observed": True}
    audit_hash = atomic_json(AUDIT / "OFFICIAL_VAL_EXECUTION_AUDIT_v4.json", audit_payload)
    atomic_text(AUDIT / "OFFICIAL_VAL_EXECUTION_AUDIT_v4.md", "\n".join(["# Official VAL Execution Audit v4", "", "- result: `PASS`", f"- primary_results_sha256: `{json_hash}`", f"- audit_json_sha256: `{audit_hash}`", ""]))
    confirm_hash = atomic_json(AUDIT / "OFFICIAL_VAL_CONFIRMATORY_RESULT_v4.json", summary)
    atomic_text(AUDIT / "OFFICIAL_VAL_CONFIRMATORY_RESULT_v4.md", "\n".join(["# Official VAL Confirmatory Result v4", "", f"- gate_1: `{summary['gate_1']}`", f"- gate_2: `{summary['gate_2']}`", f"- disposition: `{summary['confirmatory_disposition']}`", ""]))
    cert = {"artifact_id": "OFFICIAL_VAL_RUN_COMPLETION_CERTIFICATE_v4", "created_utc": utc_now(), **run_context, "status": "COMPLETE", "official_val_accessed": True, "official_val_result_observed": True, "primary_results_sha256": json_hash, "confirmatory_result_sha256": confirm_hash}
    cert_hash = atomic_json(COMPLETION_CERT_JSON, cert)
    atomic_text(AUDIT / "OFFICIAL_VAL_RUN_COMPLETION_CERTIFICATE_v4.md", "\n".join(["# Official VAL Run Completion Certificate v4", "", "- result: `COMPLETE`", f"- certificate_json_sha256: `{cert_hash}`", ""]))
    return {"primary_results_hash": json_hash, "audit_hash": audit_hash, "confirmatory_hash": confirm_hash, "completion_certificate_hash": cert_hash}


def execute_official(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    pre = preflight(args.unlock_token, require_unlock=True)
    sent = create_run_start_sentinel(resume=args.resume)
    run_context = {
        "sentinel_hash": sent["sentinel_hash"],
        "authorization_hash": sha256_file(AUDIT / "PREVAL_HUMAN_AUTHORIZATION_v4.json") if (AUDIT / "PREVAL_HUMAN_AUTHORIZATION_v4.json").exists() else "",
        "final_lock_hash": sha256_file(AUDIT / "PREVAL_FINAL_LOCK_v4.json") if (AUDIT / "PREVAL_FINAL_LOCK_v4.json").exists() else "",
        "runner_hash": sha256_file(Path(__file__).resolve()),
        "model_manifest_hash": sha256_file(EXEC / "FULL_TRAIN_HGB_FINAL_MODELS_MANIFEST_v4.json"),
        "environment_hash": EXPECTED_ENV_HASH,
    }
    if args.resume:
        validate_existing_checkpoints(run_context)
    checkpoint("00_PRE_READ_LOCK", {"preflight": "PASS"}, run_context)
    pop = population_from_root(VAL_FORBIDDEN, official_access_granted=True)
    atomic_json(EXEC / "OFFICIAL_VAL_POPULATION_v4.json", {"scenario_count": int(len(pop)), "target_count": int(pop["eligible_primary_targets"].sum()), "scenarios_sha256": sha256_json(pop[["scenario_id", "city", "eligible_primary_targets"]].to_dict(orient="records"))})
    checkpoint("01_VAL_POPULATION", {"eligible_scenarios": int(len(pop)), "eligible_targets": int(pop["eligible_primary_targets"].sum())}, run_context)
    features, _edges, qa = build_features_from_population(VAL_FORBIDDEN, pop, official_access_granted=True)
    checkpoint("02_VAL_FEATURES", {"rows": int(len(features)), "c_shuffled_seed": VAL_SHUFFLE_SEED, "self_donor_count": int(qa["self_match"].sum())}, run_context)
    pred_path = EXEC / "OFFICIAL_VAL_PREDICTIONS_v4.parquet"
    pred = predict_losses(features, pre["manifest"], pred_path)
    checkpoint("03_PREDICTIONS", {"prediction_rows": int(len(pred)), "prediction_sha256": sha256_file(pred_path)}, run_context)
    sr = scenario_results(pred)
    checkpoint("04_PRIMARY_AGGREGATE", {"delta_H_to_C": float(sr["delta_H_to_C"].mean())}, run_context)
    checkpoint("05_RPVA_SCENARIO", {"lambda": float(sr["lambda"].mean())}, run_context)
    boot = bootstrap(sr, BOOTSTRAP_REPS, BOOTSTRAP_SEED, "bootstrap")
    checkpoint("06_BOOTSTRAP", {"bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": BOOTSTRAP_SEED}, run_context)
    pseudo = pseudo_nearest(pred, PSEUDO_NEAREST_REPS, PSEUDO_NEAREST_SEED)
    checkpoint("07_PSEUDONEAREST", {"pseudo_nearest_reps": PSEUDO_NEAREST_REPS, "pseudo_nearest_seed": PSEUDO_NEAREST_SEED, "p_value": pseudo["p_value"]}, run_context)
    summary = summarize(pred, sr, boot, pseudo)
    checkpoint("08_SECONDARY", {"secondary": summary["secondary"]}, run_context)
    hashes = write_final_artifacts(summary, sr, pred_path, pseudo, boot, run_context)
    checkpoint("09_FINALIZE", {"summary": summary, "artifact_hashes": hashes}, run_context)
    write_status(phase="complete", process_state="completed", elapsed=round(time.time() - started, 3), official_val_accessed=True, official_val_result_observed=True, eligible_scenarios=summary["eligible_scenarios"], eligible_targets=summary["eligible_targets"])
    return {"status": "COMPLETE", **hashes, "summary": summary}


def shadow_rehearsal() -> dict[str, Any]:
    if SENTINEL_PATH.exists():
        raise SystemExit("Shadow rehearsal refuses to run after an official sentinel exists.")
    started = time.time()
    SHADOW.mkdir(parents=True, exist_ok=True)
    pop = population_from_root(TRAIN_ROOT, official_access_granted=False, limit=8)
    features, _edges, qa = build_features_from_population(TRAIN_ROOT, pop, official_access_granted=False)
    pred_path = SHADOW / "SHADOW_OFFICIAL_VAL_PREDICTIONS_v4.parquet"
    pred = predict_losses(features, validate_model_manifest(), pred_path)
    sr = scenario_results(pred)
    boot = bootstrap(sr, SHADOW_BOOTSTRAP_REPS, BOOTSTRAP_SEED, "bootstrap")
    pseudo = pseudo_nearest(pred, SHADOW_PSEUDO_NEAREST_REPS, PSEUDO_NEAREST_SEED)
    summary = summarize(pred, sr, boot, pseudo)
    pq.write_table(pa.Table.from_pandas(sr, preserve_index=False), SHADOW / "SHADOW_OFFICIAL_VAL_RPVA_SCENARIO_RESULTS_v4.parquet")
    np.savez_compressed(SHADOW / "SHADOW_OFFICIAL_VAL_PSEUDONEAREST_v4.npz", lambda_null=pseudo["lambda_null"], bootstrap_delta=boot["delta_distribution"], bootstrap_lambda=boot["lambda_distribution"])
    result = {
        "artifact_id": "OFFICIAL_VAL_SHADOW_REHEARSAL_v4",
        "status": "PASS",
        "created_utc": utc_now(),
        "elapsed_seconds": round(time.time() - started, 3),
        "official_val_accessed": False,
        "official_val_result_observed": False,
        "run_start_sentinel_created": False,
        "train_shadow_scenarios": int(pop["scenario_id"].nunique()),
        "train_shadow_targets": int(len(features)),
        "c_shuffled_seed_verified": VAL_SHUFFLE_SEED,
        "c_shuffled_self_donors": int(qa["self_match"].sum()),
        "bootstrap_reps_shadow": SHADOW_BOOTSTRAP_REPS,
        "pseudo_nearest_reps_shadow": SHADOW_PSEUDO_NEAREST_REPS,
        "production_bootstrap_reps": BOOTSTRAP_REPS,
        "production_pseudo_nearest_reps": PSEUDO_NEAREST_REPS,
        "summary_not_scientifically_interpretable": summary,
    }
    json_hash = atomic_json(SHADOW / "SHADOW_REHEARSAL_RESULT_v4.json", result)
    atomic_text(SHADOW / "SHADOW_REHEARSAL_RESULT_v4.md", "\n".join(["# Official VAL Runner Shadow Rehearsal v4", "", "- result: `PASS`", "- official_val_accessed: `false`", f"- result_json_sha256: `{json_hash}`", ""]))
    print(json.dumps({"shadow_rehearsal": "PASS", "official_val_accessed": False, "result_json_sha256": json_hash}, indent=2, sort_keys=True))
    return result


def static_fit_guard_status() -> dict[str, Any]:
    text = Path(__file__).read_text(encoding="utf-8")
    count = text.count("." + "fit" + "(")
    return {"status": "PASS" if count == 0 else "FAIL", "fit_call_occurrences_in_runner": count, "official_mode_uses_frozen_estimator_proxy": True}


def firewall_test() -> dict[str, Any]:
    try:
        guard_not_val(VAL_FORBIDDEN / "sentinel.parquet")
    except ValAccessFirewall as exc:
        return {"status": "PASS", "result": "BLOCKED_BEFORE_SENTINEL", "message": str(exc), "official_val_accessed": False}
    return {"status": "FAIL", "official_val_accessed": True}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed official AV2 VAL v4 runner.")
    parser.add_argument("--unlock-token", default=None)
    parser.add_argument("--dry-run-preflight", action="store_true")
    parser.add_argument("--preval-lock-check", action="store_true")
    parser.add_argument("--shadow-rehearsal", action="store_true")
    parser.add_argument("--execute-official", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--firewall-test", action="store_true")
    parser.add_argument("--fit-guard-check", action="store_true")
    args = parser.parse_args()
    if args.firewall_test:
        print(json.dumps(firewall_test(), indent=2, sort_keys=True))
        return 0
    if args.fit_guard_check:
        status = static_fit_guard_status()
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0 if status["status"] == "PASS" else 2
    if args.preval_lock_check:
        certified = validate_lock_candidate()
        write_status(phase="preval_lock_check", process_state="complete", official_val_accessed=False)
        print(json.dumps({"preval_lock_check": "PASS", "official_val_accessed": False, "environment": certified["environment"]}, indent=2, sort_keys=True))
        return 0
    if args.dry_run_preflight:
        certified = preflight(args.unlock_token, require_unlock=False)
        write_status(phase="dry_run_preflight", process_state="complete", official_val_accessed=False)
        print(json.dumps({"preflight": "PASS", "manifest_id": certified["freeze_manifest"].get("manifest_id"), "official_val_accessed": False}, indent=2, sort_keys=True))
        return 0
    if args.shadow_rehearsal:
        shadow_rehearsal()
        return 0
    if args.execute_official:
        result = execute_official(args)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
