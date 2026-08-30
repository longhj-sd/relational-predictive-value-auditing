# RPVA 2018 External-Season Replication Package

This package freezes the 2018 external-season replication protocol used for the Knowledge-Based Systems submission of Relational Predictive-Value Auditing (RPVA). It provides executable code, protocol files, aggregate expected outputs, quality-control reports, environment notes, and manifests. It does not redistribute raw NFL Big Data Bowl files or row-level restricted derived data.

## Data Sources

Required local files:

- NFL Big Data Bowl 2021 competition data: `games.csv`, `plays.csv`, `players.csv`, and `week1.csv` through `week17.csv`.
- Target-receiver labels: `targetedReceiver.csv` from `tombliss/nfl-big-data-bowl-2021-bonus`.

The target labels link by `gameId` and `playId`; the primary target identifier is `targetNflId`.

## Raw-Data Boundary

Raw NFL tracking files, competition metadata files, competition archives, row-level tracking extracts, and row-level held-out predictions are not redistributed. Reviewers should obtain the competition files and target-label file independently under the applicable access terms.

## Frozen Protocol

- Temporal split: weeks 1-11 train, weeks 12-13 validation, weeks 14-17 held out.
- Event chain: `ball_snap` -> `pass_forward` -> `pass_arrived`.
- Audit states: A, H, D.
- Roles: R is the targeted receiver; N is the nearest coverage-eligible defender to R at `pass_forward`; O is the within-play mean over remaining coverage-eligible defenders.
- Primary defender set: CB, DB, FS, S, SS.
- Loss: Euclidean endpoint error in yards.
- Inference: game-cluster bootstrap with 2,000 replicates.
- Relational-null diagnostic: pseudo-nearest randomization with 5,000 permutations.

## Environment

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
pip install -r nfl_external_validation/2018/environment/requirements-2018.txt
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -r nfl_external_validation\2018\environment\requirements-2018.txt
```

## Run

```bash
python nfl_external_validation/2018/scripts/run_2018_external_replication.py \
  --config nfl_external_validation/2018/protocol/protocol_config.yaml \
  --raw-data-root "/path/to/authorized/nfl_bdb2021_data" \
  --target-label-file "/path/to/targetedReceiver.csv" \
  --output-dir outputs_2018
```

PowerShell example:

```powershell
python nfl_external_validation\2018\scripts\run_2018_external_replication.py `
  --config nfl_external_validation\2018\protocol\protocol_config.yaml `
  --raw-data-root "D:\path\to\authorized\nfl_bdb2021_data" `
  --target-label-file "D:\path\to\targetedReceiver.csv" `
  --output-dir outputs_2018
```

## Expected Primary Results

Small numerical differences may occur across supported package versions. Verification uses documented tolerances rather than exact binary equality.

- Held-out plays: 3,475.
- `G_R = 0.918`.
- `G_N = 0.781`.
- `G_O = 0.335`.
- `lambda_D = 0.446`.
- `lambda_D` 95% CI: [0.384, 0.510].
- Pseudo-nearest null 95% interval: [-0.057, 0.059].
- Pseudo-nearest P = 0.00020.

The expanded-defender sensitivity estimates `G_R = 0.917`, `G_N = 0.917`, `G_O = 0.381`, and `lambda_D = 0.535`; this supports positive nearest-other localization but does not support a strict receiver-nearest ordering in that sensitivity.

## Verify

```bash
python nfl_external_validation/2018/scripts/verify_2018_release.py
```

Optional full rerun verification:

```bash
python nfl_external_validation/2018/scripts/verify_2018_release.py \
  --run-pipeline \
  --raw-data-root "/path/to/authorized/nfl_bdb2021_data" \
  --target-label-file "/path/to/targetedReceiver.csv" \
  --output-dir outputs_2018_verify
```
