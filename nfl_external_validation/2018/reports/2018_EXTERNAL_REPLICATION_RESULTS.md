# 2018 External Replication Results

## Objective

Execute a held-out external-season RPVA replication on official 2018 NFL Big Data Bowl 2021 data using the locked A/H/D, R/N/O, bootstrap, and pseudo-nearest definitions.

## Data Provenance

Official BDB2021 competition files were obtained locally under the applicable access terms. Raw files are not redistributed in this repository.

## 2018 Data Structure

The primary modeling table contains 86,816 player-play rows. Held-out weeks 14-17 contain 21,098 player rows and 3,475 three-role eligible plays.

## Target Receiver Source

Primary labels use `targetedReceiver.csv` from the accessible bonus dataset. Reconstructed targets were computed only for QC and ambiguity sensitivity.

## 2023-to-2018 Harmonization

Feature construction uses the prespecified left-to-right coordinate normalization and harmonized 2018 feature definitions because the submitted repository package did not contain a verified full raw-data transformation pipeline for the 2023 NFL application.

## Frozen Protocol

The protocol was written and hashed before fitting final models. See `nfl_external_validation/2018/protocol/protocol_sha256.txt`.

## Model Setup

HGB model: sklearn `HistGradientBoostingRegressor` defaults with deterministic preprocessing and random_state 20260730. Separate x/y models were fit for A, H, and D on weeks 1-11. Weeks 14-17 were held out.

## Primary RPVA Estimates

| estimand   |   estimate |    ci_low |   ci_high |
|:-----------|-----------:|----------:|----------:|
| G_R        |   0.917786 | 0.866265  |  0.971487 |
| G_N        |   0.78121  | 0.723957  |  0.838939 |
| G_O        |   0.335089 | 0.305382  |  0.362108 |
| lambda_D   |   0.446121 | 0.384398  |  0.509817 |
| delta_RN   |   0.136576 | 0.0815001 |  0.190805 |
| delta_RO   |   0.582696 | 0.526785  |  0.646572 |

## Game-Cluster Bootstrap

The 95% percentile game-cluster bootstrap used 2000 replicates without refitting models.

## Pseudo-Nearest Falsification

Observed lambda_D = 0.446121; null mean = 0.000159; 95% null interval = [-0.057225, 0.058860]; add-one corrected P = 0.000200.

## Weekly Stability

|      G_R |      G_N |      G_O |   lambda_D |   delta_RN |   delta_RO |   week |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|-------:|
| 0.791594 | 0.724712 | 0.321973 |   0.402739 |  0.0668821 |   0.469621 |     14 |
| 0.961081 | 0.785293 | 0.398942 |   0.386351 |  0.175788  |   0.562138 |     15 |
| 0.959698 | 0.778513 | 0.303752 |   0.474761 |  0.181185  |   0.655946 |     16 |
| 0.963264 | 0.840597 | 0.317604 |   0.522993 |  0.122667  |   0.645661 |     17 |

## Sensitivity Analyses

| analysis                                   |      G_R |      G_N |      G_O |   lambda_D |     delta_RN |   delta_RO |   heldout_plays |
|:-------------------------------------------|---------:|---------:|---------:|-----------:|-------------:|-----------:|----------------:|
| S1_primary_defender_set                    | 0.917786 | 0.78121  | 0.335089 |   0.446121 |  0.136576    |   0.582696 |            3475 |
| S2_expanded_defender_set                   | 0.916614 | 0.91668  | 0.381393 |   0.535287 | -6.62317e-05 |   0.535221 |            3480 |
| S4_endpoint_plus_minus_1_frame             | 0.917786 | 0.78121  | 0.335089 |   0.446121 |  0.136576    |   0.582696 |            3476 |
| S5_exclude_ambiguous_reconstructed_targets | 0.997092 | 0.830981 | 0.350942 |   0.480039 |  0.166111    |   0.64615  |            3389 |

## 2023-vs-2018 Comparison

|   season |      G_R |     G_N |      G_O |   lambda_D |   delta_RN |   delta_RO |   lambda_difference_vs_2023 |
|---------:|---------:|--------:|---------:|-----------:|-----------:|-----------:|----------------------------:|
|     2018 | 0.917786 | 0.78121 | 0.335089 |   0.446121 |   0.136576 |   0.582696 |                   -0.303879 |
|     2023 | 1.735    | 1.359   | 0.609    |   0.75     |   0.376    |   1.126    |                    0        |

## Deviations

No post-freeze deviations were recorded. ExtraTrees robustness was skipped because the exact submitted final ExtraTrees configuration was not recoverable.

## Limitations

The 2018 workflow is a harmonized external replication, not an exact raw-pipeline rerun of the submitted 2023 implementation. N is proximity-defined and should not be interpreted as assigned coverage.

## Replication Summary

The primary 2018 analysis reproduced the principal relational ordering, with positive nearest-other localization and pseudo-nearest falsification support.

## Scientific Interpretation

The 2018 evidence should be interpreted as a held-out audit of relational allocation under harmonized data definitions. It does not establish causality, tactical assignment, or deployment-time utility.
