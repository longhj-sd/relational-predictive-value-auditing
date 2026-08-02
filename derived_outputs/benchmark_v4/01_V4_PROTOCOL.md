{
  "aggregate_matched_localized": {
    "aggregate": 0.8,
    "delta": 0.3999999999999999,
    "g_N": 0.8,
    "g_O": 0.40000000000000024,
    "g_R": 1.2,
    "lambda": 0.3999999999999998,
    "truth_aggregate_absolute_difference_vs_null": 0.0
  },
  "baseline_inputs": {
    "main_manuscript": "SANITIZED_LOCAL_SOURCE_PATH_NOT_PUBLIC",
    "supplement": "SANITIZED_LOCAL_SOURCE_PATH_NOT_PUBLIC",
    "use_as_frozen_baselines_only": true,
    "v31_folder": "SANITIZED_LOCAL_SOURCE_PATH_NOT_PUBLIC"
  },
  "calibration": {
    "replications": 500,
    "resampling_replicates_per_method": 499,
    "root_seed": 2026080401,
    "seed_family": "V4-CAL-20260804; not used in V1/V2/V2.1/V3/V3.1"
  },
  "candidate_methods": {
    "M1": "current game-cluster percentile bootstrap",
    "M2": "studentized game-cluster bootstrap-t",
    "M3": "game-level wild cluster bootstrap-t using Webb six-point weights",
    "M4": "not used; CR2 was not added because the RPVA estimand is a paired cluster-mean estimand and a correct CR2 implementation would require an additional regression layer not present in V3.1"
  },
  "confirmatory": {
    "freeze_rule": "No method, DGP, seed, gate, or interval modification after confirmation starts.",
    "replications": 2000,
    "resampling_replicates": 999,
    "root_seed": 2026081401,
    "seed_family": "V4-CONF-20260814; independent from calibration and prior benchmarks"
  },
  "frozen_at": "2026-08-01T23:57:10",
  "held_out_design": {
    "O_is_within_event_mean": true,
    "cluster_size_distribution": "same cluster-level distributional surrogate as V3.1",
    "clusters": 64,
    "estimands_preserved": [
      "g_R",
      "g_N",
      "g_O",
      "aggregate",
      "delta",
      "lambda"
    ],
    "paired_gain_construction": "previous endpoint error minus current endpoint error; cluster means aggregate event-level paired gains",
    "role_weights": {
      "N": 0.3333333333333333,
      "O": 0.3333333333333333,
      "R": 0.3333333333333333
    }
  },
  "protocol": "Benchmark V4 independent inference calibration for RPVA",
  "strict_relational_null": {
    "aggregate": 0.8,
    "delta": 0.0,
    "g_N": 0.8,
    "g_O": 0.8,
    "g_R": 0.8,
    "lambda": 0.0,
    "positive_aggregate_gain_retained": true
  }
}