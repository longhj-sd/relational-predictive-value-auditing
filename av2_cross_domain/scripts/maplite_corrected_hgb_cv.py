from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

import phase3b_train_only_execution as p3b


ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "execution"
AUDIT = ROOT / "audit"
MATRIX = ROOT / "data" / "processed" / "phase3b" / "TRAIN_50K_FEATURES_MAPLITE_CORRECTED_v2.parquet"
CV_PATH = EXEC / "PRIMARY_CV_RESULTS_MAPLITE_CORRECTED_v3.csv"
STATE_SUMMARY = EXEC / "PRIMARY_CV_STATE_SUMMARY_MAPLITE_CORRECTED_v3.csv"
CONFIG_PATH = EXEC / "PRIMARY_MODEL_CONFIGS_MAPLITE_CORRECTED_v3.yaml"
SELECTION_MD = EXEC / "PRIMARY_MODEL_SELECTION_MAPLITE_CORRECTED_v3.md"
MATCHED_CSV = EXEC / "MATCHED_HC_CANDIDATE_RESULTS_MAPLITE_CORRECTED_v3.csv"
MATCHED_YAML = EXEC / "MATCHED_HC_CONFIG_MAPLITE_CORRECTED_v3.yaml"
MATCHED_MD = EXEC / "MATCHED_HC_SELECTION_MAPLITE_CORRECTED_v3.md"
ADEQUACY_CSV = EXEC / "TRAIN_MODEL_ADEQUACY_MAPLITE_CORRECTED_v3.csv"
ADEQUACY_MD = EXEC / "TRAIN_MODEL_ADEQUACY_MAPLITE_CORRECTED_v3.md"
COMPLETION_JSON = AUDIT / "PRIMARY_CV_480_COMPLETION_AUDIT_MAPLITE_CORRECTED_v3.json"
COMPLETION_MD = AUDIT / "PRIMARY_CV_480_COMPLETION_AUDIT_MAPLITE_CORRECTED_v3.md"
EQUIV_CSV = AUDIT / "HGB_EXECUTION_EQUIVALENCE_QA_MAPLITE_CORRECTED_v3.csv"
EQUIV_MD = AUDIT / "HGB_EXECUTION_EQUIVALENCE_QA_MAPLITE_CORRECTED_v3.md"
LOG_PATH = ROOT / "logs" / "maplite_corrected_hgb_subprocess_controller.log"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_csv_safely(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def append_row(path: Path, row: dict) -> str:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=p3b.PRIMARY_CV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row[k] for k in p3b.PRIMARY_CV_COLUMNS})
        f.flush()
        os.fsync(f.fileno())
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_cv() -> pd.DataFrame:
    if not CV_PATH.exists():
        return pd.DataFrame(columns=p3b.PRIMARY_CV_COLUMNS)
    return pd.read_csv(CV_PATH)


def completed_keys(candidate_limit: int = 24) -> set[tuple[str, int, int]]:
    df = read_cv()
    if df.empty:
        return set()
    duplicates = df.duplicated(p3b.HGB_UNIT_COLUMNS, keep=False)
    if bool(duplicates.any()):
        raise RuntimeError("Corrected CV contains duplicate unit keys")
    expected = p3b.hgb_expected_units(candidate_limit)
    keys = {(r.state, int(r.candidate_index), int(r.fold)) for r in df.itertuples(index=False)}
    invalid = keys - expected
    if invalid:
        raise RuntimeError(f"Corrected CV contains invalid keys: {sorted(invalid)[:3]}")
    return keys


def run_single(args: argparse.Namespace) -> None:
    state = args.state
    candidate_index = int(args.candidate_index)
    fold = int(args.fold)
    key = (state, candidate_index, fold)
    if key in completed_keys(args.hgb_candidates):
        print(f"SINGLE_UNIT SKIP state={state} candidate_index={candidate_index} fold={fold} reason=already_complete", flush=True)
        return
    df = pd.read_parquet(MATRIX)
    row = p3b.fit_evaluate_hgb_unit(df, state, candidate_index, fold, args.hgb_candidates)
    row_hash = append_row(CV_PATH, row)
    print(
        f"SINGLE_UNIT success state={state} candidate_index={candidate_index} fold={fold} "
        f"mean_endpoint_mse={row['mean_endpoint_mse']:.10f} mean_endpoint_fde={row['mean_fde']:.10f} "
        f"row_hash={row_hash} exit_success=true",
        flush=True,
    )
    del df
    gc.collect()


def controller(args: argparse.Namespace) -> pd.DataFrame:
    expected = p3b.hgb_expected_units(args.hgb_candidates)
    if args.force and CV_PATH.exists():
        CV_PATH.unlink()
    new_units = 0
    for state, candidate_index, fold in p3b.hgb_units_in_order(args.hgb_candidates):
        key = (state, candidate_index, fold)
        before = completed_keys(args.hgb_candidates)
        if key in before:
            continue
        if args.max_new_units is not None and new_units >= args.max_new_units:
            break
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
        started = time.time()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate()
        after = completed_keys(args.hgb_candidates)
        status = "success" if proc.returncode == 0 and key in after and len(after) == len(before) + 1 else "failed"
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": utc_timestamp(),
                "state": state,
                "candidate_index": candidate_index,
                "fold": fold,
                "child_exit_code": proc.returncode,
                "elapsed_seconds": round(time.time() - started, 3),
                "status": status,
                "stdout_tail": stdout.strip().splitlines()[-3:],
                "stderr_tail": stderr.strip().splitlines()[-3:],
            }, sort_keys=True) + "\n")
        print(stdout, end="", flush=True)
        if stderr:
            print(stderr, file=sys.stderr, end="", flush=True)
        if status != "success":
            raise RuntimeError(f"Corrected HGB subprocess controller stopped on {key}")
        new_units += 1
    res = read_cv().sort_values(p3b.HGB_UNIT_COLUMNS).reset_index(drop=True)
    write_csv_safely(res, CV_PATH)
    if completed_keys(args.hgb_candidates) == expected:
        write_selection_artifacts(res, args.hgb_candidates)
        write_completion_audit(res, args.hgb_candidates)
        write_execution_equivalence(res, args.hgb_candidates)
        write_adequacy(pd.read_parquet(MATRIX), yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["selected_configs"]["H"])
    return res


def write_selection_artifacts(res: pd.DataFrame, candidate_limit: int) -> None:
    configs = p3b.select_configs(res)
    matched = corrected_matched_config(res, candidate_limit)
    payload = {
        "selection_rule": "empirical TRAIN-only grouped-CV minimum mean endpoint MSE",
        "randomized_search_seed": p3b.TUNING_SEED,
        "candidate_count_per_state": int(res["candidate_index"].nunique()),
        "corrected_maplite_matrix": str(MATRIX),
        "corrected_maplite_matrix_sha256": sha256_file(MATRIX),
        "selected_configs": configs,
        "MATCHED_HC_CONFIG_FINAL": {
            **p3b.config_candidates(candidate_limit)[int(matched.iloc[0].candidate_index)],
            "candidate_index": int(matched.iloc[0].candidate_index),
            "mean_standardized_HC_loss": float(matched.iloc[0].mean_standardized_HC_loss),
        },
    }
    CONFIG_PATH.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    summary = res.groupby(["state", "candidate_index"], as_index=False).agg(mean_endpoint_mse=("mean_endpoint_mse", "mean"), mean_fde=("mean_fde", "mean"))
    summary["rank"] = summary.groupby("state")["mean_endpoint_mse"].rank(method="first")
    summary["selected"] = summary["rank"] == 1
    write_csv_safely(summary.sort_values(["state", "rank"]), STATE_SUMMARY)
    best_table = summary[summary["selected"]].sort_values("state")
    SELECTION_MD.write_text(
        "\n".join([
            "# Primary Model Selection MAP-LITE Corrected v3",
            "",
            "Empirical winners selected using corrected TRAIN-only grouped city-stratified folds and predictive endpoint loss only.",
            "",
            "## Selected State-Specific Configs",
            "",
            best_table.to_markdown(index=False),
            "",
            "## Matched H/C Config",
            "",
            pd.DataFrame([payload["MATCHED_HC_CONFIG_FINAL"]]).to_markdown(index=False),
            "",
            f"- CV results hash: `{sha256_file(CV_PATH)}`",
            "",
        ]),
        encoding="utf-8",
    )
    MATCHED_YAML.write_text(yaml.safe_dump(payload["MATCHED_HC_CONFIG_FINAL"], sort_keys=False), encoding="utf-8")
    MATCHED_MD.write_text(
        "# Matched H/C Selection MAP-LITE Corrected v3\n\n"
        + matched.head(10).to_markdown(index=False)
        + f"\n\n- matched_config_sha256: `{sha256_file(MATCHED_YAML)}`\n",
        encoding="utf-8",
    )


def corrected_matched_config(res: pd.DataFrame, candidate_limit: int) -> pd.DataFrame:
    hc = res[res.state.isin(["H", "C"])].copy()
    state_means = hc.groupby("state")["mean_endpoint_mse"].agg(["mean", "std"]).to_dict("index")
    rows = []
    for ci, g in hc.groupby("candidate_index"):
        vals = []
        for state, sg in g.groupby("state"):
            vals.append((sg["mean_endpoint_mse"].mean() - state_means[state]["mean"]) / state_means[state]["std"])
        rows.append({"candidate_index": int(ci), "mean_standardized_HC_loss": float(np.mean(vals))})
    out = pd.DataFrame(rows).sort_values(["mean_standardized_HC_loss", "candidate_index"])
    write_csv_safely(out, MATCHED_CSV)
    return out


def write_completion_audit(res: pd.DataFrame, candidate_limit: int) -> None:
    expected = p3b.hgb_expected_units(candidate_limit)
    keys = [(r.state, int(r.candidate_index), int(r.fold)) for r in res.itertuples(index=False)]
    counts = res.groupby("state").size().to_dict()
    payload = {
        "audit_id": "PRIMARY_CV_480_COMPLETION_AUDIT_MAPLITE_CORRECTED_v3",
        "input_path": str(CV_PATH),
        "input_sha256": sha256_file(CV_PATH),
        "expected_units": len(expected),
        "observed_rows": int(len(res)),
        "unique_keys": int(len(set(keys))),
        "duplicates": int(len(keys) - len(set(keys))),
        "missing_expected_units": int(len(expected - set(keys))),
        "state_counts": {state: int(counts.get(state, 0)) for state in p3b.STATE_ORDER},
        "status": "PASS" if len(res) == len(expected) and len(set(keys)) == len(expected) else "FAIL",
        "official_val_outcome_blind": True,
    }
    COMPLETION_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    COMPLETION_MD.write_text(
        "# Primary CV 480 Completion Audit MAP-LITE Corrected v3\n\n"
        + f"- status: `{payload['status']}`\n"
        + f"- input_sha256: `{payload['input_sha256']}`\n"
        + f"- observed_rows: `{payload['observed_rows']}`\n"
        + f"- unique_keys: `{payload['unique_keys']}`\n"
        + f"- missing_expected_units: `{payload['missing_expected_units']}`\n"
        + f"- state_counts: `{payload['state_counts']}`\n"
        + "- official_val_outcome_blind: `true`\n",
        encoding="utf-8",
    )


def write_execution_equivalence(res: pd.DataFrame, candidate_limit: int, tolerance: float = 1e-9) -> None:
    df = pd.read_parquet(MATRIX)
    rows = []
    for unit in res.head(3)[p3b.HGB_UNIT_COLUMNS].to_dict(orient="records"):
        state = str(unit["state"])
        candidate_index = int(unit["candidate_index"])
        fold = int(unit["fold"])
        observed = res[(res.state == state) & (res.candidate_index == candidate_index) & (res.fold == fold)].iloc[0]
        rerun = p3b.fit_evaluate_hgb_unit(df, state, candidate_index, fold, candidate_limit)
        row = {"state": state, "candidate_index": candidate_index, "fold": fold, "tolerance_abs": tolerance}
        max_abs_diff = 0.0
        for metric in ["dx_6s_mse", "dy_6s_mse", "mean_endpoint_mse", "mean_fde"]:
            diff = abs(float(rerun[metric]) - float(observed[metric]))
            row[f"authoritative_{metric}"] = float(observed[metric])
            row[f"rerun_{metric}"] = float(rerun[metric])
            row[f"absdiff_{metric}"] = diff
            max_abs_diff = max(max_abs_diff, diff)
        row["max_abs_diff"] = max_abs_diff
        row["pass"] = max_abs_diff <= tolerance
        rows.append(row)
    out = pd.DataFrame(rows)
    write_csv_safely(out, EQUIV_CSV)
    EQUIV_MD.write_text(
        "# HGB Execution-Equivalence QA MAP-LITE Corrected v3\n\n"
        + f"- result: `{'PASS' if bool(out['pass'].all()) else 'FAIL'}`\n"
        + f"- max_abs_diff: `{float(out['max_abs_diff'].max())}`\n"
        + f"- csv_sha256: `{sha256_file(EQUIV_CSV)}`\n\n"
        + out.to_markdown(index=False)
        + "\n\nOfficial VAL remains outcome-blind.\n",
        encoding="utf-8",
    )


def write_adequacy(df: pd.DataFrame, h_config: dict) -> None:
    X = df[p3b.feature_names_for_state("H")].astype(np.float32).to_numpy()
    y = df[p3b.TARGET_COLUMNS].to_numpy(np.float32)
    cv_pred = df[["cv_dx_6s", "cv_dy_6s"]].to_numpy(np.float32)
    folds = df.fold.to_numpy()
    cfg = {k: h_config[k] for k in p3b.p3a.HGB_SPACE}
    rows = []
    for fold in range(5):
        tr, te = folds != fold, folds == fold
        pred = np.zeros((te.sum(), 2), dtype=np.float32)
        for j in range(2):
            model = HistGradientBoostingRegressor(random_state=p3b.TUNING_SEED + 900 + fold + j, **cfg)
            model.fit(X[tr], y[tr, j])
            pred[:, j] = model.predict(X[te])
        fde_h = np.linalg.norm(pred - y[te], axis=1)
        fde_cv = np.linalg.norm(cv_pred[te] - y[te], axis=1)
        rows.append({"fold": fold, "hgb_h_mean_fde": float(fde_h.mean()), "constant_velocity_mean_fde": float(fde_cv.mean()), "paired_mean_fde_reduction": float((fde_cv - fde_h).mean()), "n_rows": int(te.sum())})
    out = pd.DataFrame(rows)
    write_csv_safely(out, ADEQUACY_CSV)
    ADEQUACY_MD.write_text(
        "# TRAIN Model Adequacy MAP-LITE Corrected v3\n\n"
        + out.to_markdown(index=False)
        + f"\n\n- mean paired FDE reduction: `{out['paired_mean_fde_reduction'].mean():.6f}`\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-unit", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--state", choices=p3b.STATE_ORDER)
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--hgb-candidates", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-new-units", type=int)
    args = parser.parse_args()
    if args.single_unit:
        run_single(args)
    elif args.run_all:
        controller(args)
    else:
        print(json.dumps({"completed_units": len(completed_keys(args.hgb_candidates)), "cv_path": str(CV_PATH)}, indent=2))


if __name__ == "__main__":
    main()
