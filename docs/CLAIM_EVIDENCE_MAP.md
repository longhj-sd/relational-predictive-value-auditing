# Claim Evidence Map

| Claim | Primary manuscript evidence | Repository evidence | Scope |
|---|---|---|---|
| aggregate non-identifiability | Proposition 1 | framework docs and examples | formal |
| controlled recovery | controlled benchmark | preserved controlled outputs | simulation |
| NFL 2023 | Domain I main audit | `nfl_example/`, `derived_outputs/` | NFL 2023 |
| NFL 2018 replication | temporal replication | `nfl_external_validation/` | NFL 2018 |
| NFL pseudo-nearest | relational-null diagnostic | preserved pseudo-nearest summaries | Domain I falsification |
| AV2 alignment specificity | C_SHUFFLED control | `av2_cross_domain/aggregate_results/` | Domain II control |
| held-out-city robustness | six held-out city analyses | `av2_cross_domain/aggregate_results/heldout_city_summary.csv` | Domain II development |
| official AV2 Gate 1 | lower bound of the two-sided 95% CI for Delta H to C is > 0 | `official_gate_summary.json` | confirmatory VAL |
| official AV2 Gate 2 | lambda two-sided 95% CI excludes 0 and two-sided add-one pseudo-nearest P < 0.05 | `official_gate_summary.json` | confirmatory VAL |
| one-shot governance | pre-VAL lock and final seal | `av2_cross_domain/public_provenance/` | execution governance |
