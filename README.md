# Relational Predictive-Value Auditing (RPVA)

## 1. What RPVA evaluates
RPVA evaluates how paired held-out incremental predictive value from a nested information transition is allocated across prespecified relational roles in a multi-agent predictive system.

## 2. Main estimands
`G_R`, `G_N`, and `G_O` are role-specific gains. `lambda_D = G_N - G_O`, `delta_RN = G_R - G_N`, and `delta_RO = G_R - G_O` summarize relational contrasts.

## 3. Controlled benchmark
Aggregate-matched controlled profiles test whether RPVA recovers homogeneous and localized relational gain structures and separates subgroup absolute performance from incremental value allocation.

## 4. 2023 primary audit
The 2023 forward-time audit reports R > N > O and positive nearest-other localization in held-out passing-play records.

## 5. 2023 deterministic reconstruction
A post hoc audit deterministically reconstructed the final 2023 HGB analysis from proven archived pipeline lineage. Source and held-out sample counts matched exactly, and all six headline estimates reproduced within 0.00045 yd.

## 6. 2018 external-season replication
A harmonized 2018 public data release independently reproduces the principal R > N > O ordering and positive nearest-other localization.

## 7. 2018 second-estimator sensitivity
A post hoc ExtraTrees sensitivity using an ARCHIVED_2023_PROJECT configuration preserves the principal relational gain structure; absolute magnitudes remain estimator-dependent.

## 8. How to reproduce
Install the package, place authorized Kaggle/NFL data under the documented data directories, and run the validation scripts. Public-safe aggregate outputs are included where redistribution is allowed.

## 9. Raw-data acquisition
Raw NFL tracking data are not redistributed. Obtain the 2023 and 2018 source data from the original Kaggle/NFL Big Data Bowl sources under their access terms.

## 10. Expected headline values
2023: G_R 1.735, G_N 1.359, G_O 0.609, lambda_D 0.750. 2018 HGB: G_R 0.918, G_N 0.781, G_O 0.335, lambda_D 0.446. 2018 ExtraTrees: G_R 1.293, G_N 0.824, G_O 0.351, lambda_D 0.473.

## 11. Repository/frozen release information
Submission-specific reproducibility state: tag `kbs-submission-20260831-r2`, commit `78e8c3a2581b8bd44adaa5903a18e4a5b3286d7f`.
