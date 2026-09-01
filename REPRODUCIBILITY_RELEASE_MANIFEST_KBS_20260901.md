# KBS Submission Reproducibility Release Manifest, 2026-09-01

Raw NFL competition files and row-level derived tracking/prediction files are not redistributed.

| Artifact | Public release path | Status | Reason |
| --- | --- | --- | --- |
| Raw NFL Big Data Bowl 2026 source files | Not included | SOURCE DATA — USER MUST OBTAIN FROM KAGGLE | Source NFL competition data are governed by Kaggle/NFL access terms and are not redistributed. |
| Raw NFL Big Data Bowl 2021 source files | Not included | SOURCE DATA — USER MUST OBTAIN FROM KAGGLE | Source NFL competition data are governed by Kaggle/NFL access terms and are not redistributed. |
| `2023_DEEP_RECOVERY_FINAL_REPORT.md` | `derived_outputs/kbs_20260901/2023_reconstruction_reports/2023_DEEP_RECOVERY_FINAL_REPORT.md` | PUBLIC | Methodological reconstruction report with aggregate counts, estimates, differences, seeds, and provenance grade; no row-level NFL records are included. |
| `2023_CONFIG_PROVENANCE_AUDIT.md` | `derived_outputs/kbs_20260901/2023_reconstruction_reports/2023_CONFIG_PROVENANCE_AUDIT.md` | PUBLIC | Configuration provenance report with machine-specific local paths removed; no row-level NFL records are included. |
| `2023_PIPELINE_LINEAGE_AUDIT.md` | `derived_outputs/kbs_20260901/2023_reconstruction_reports/2023_PIPELINE_LINEAGE_AUDIT.md` | PUBLIC | Pipeline lineage report with machine-specific local paths generalized; no row-level NFL records are included. |
| `2023_complete_estimands_with_CI.csv` | `derived_outputs/kbs_20260901/2023_aggregate_outputs/2023_complete_estimands_with_CI.csv` | PUBLIC AGGREGATE | Aggregate estimands, bootstrap CIs, replicate count, seed, and provenance grade only. |
| `2023_reconstructed_heldout_predictions.parquet` | Not included | GENERATED LOCALLY — NOT REDISTRIBUTED | Row-level held-out prediction derivative generated locally from authorized competition data; not redistributed. |
| `2023_reconstructed_RNO_play_level.parquet` | Not included | GENERATED LOCALLY — NOT REDISTRIBUTED | Row-level/play-level derivative generated locally from authorized competition data; not redistributed. |
| `B2_absolute_vs_incremental_summary_CORRECTED.csv` | `derived_outputs/kbs_20260901/controlled_diagnostics/B2_absolute_vs_incremental_summary_CORRECTED.csv` | PUBLIC AGGREGATE | Controlled synthetic diagnostic summary; no NFL row-level records are included. |
| `B3_dependence_inference_summary.csv` | `derived_outputs/kbs_20260901/controlled_diagnostics/B3_dependence_inference_summary.csv` | PUBLIC AGGREGATE | Controlled simulation inference summary; no NFL row-level records are included. |
| `2018_ExtraTrees_estimands.csv` | `derived_outputs/kbs_20260901/2018_extratrees/2018_ExtraTrees_estimands.csv` | PUBLIC AGGREGATE | Aggregate second-estimator sensitivity estimands and CIs only. |
| `2018_ExtraTrees_bootstrap_replicates.parquet` | Not included | GENERATED LOCALLY — NOT REDISTRIBUTED | Generated locally as bootstrap replicate distribution; original parquet is not redistributed under the conservative public release rule. |
| `2018_ExtraTrees_bootstrap_summary.csv` | `derived_outputs/kbs_20260901/2018_extratrees/2018_ExtraTrees_bootstrap_summary.csv` | PUBLIC AGGREGATE | Public-safe summary of bootstrap distribution: point estimates, means, SEs, CIs, replicate count, and seed. |
| `2018_ExtraTrees_pseudo_nearest_null.parquet` | Not included | GENERATED LOCALLY — NOT REDISTRIBUTED | Generated locally as pseudo-nearest null replicate distribution; original parquet is not redistributed under the conservative public release rule. |
| `2018_ExtraTrees_pseudo_nearest_summary.csv` | `derived_outputs/kbs_20260901/2018_extratrees/2018_ExtraTrees_pseudo_nearest_summary.csv` | PUBLIC AGGREGATE | Public-safe summary of pseudo-nearest null distribution: observed statistic, null quantiles, P value, replicate count, and seed. |
