# Relational Predictive-Value Auditing

## Overview

RPVA is a model-agnostic evaluation framework for auditing how the held-out predictive value of an added information state is allocated across prespecified relational positions in multi-agent predictive systems.

## Why RPVA

Aggregate utility asks whether adding information improves predictive performance. Relational predictive-value allocation asks where that improvement is allocated across prespecified roles. RPVA separates those questions so that a useful information state is not mistaken for evidence about a particular relational position without an explicit allocation audit.

## Framework

The framework uses nested information states, paired predictive gains, role-indexed values, prespecified contrasts, dependence-aware inference, and a relational null. The estimand is defined before outcome inspection, and confirmatory gates are evaluated under the frozen protocol.

## Evidence Architecture

| Layer | Evidence role |
|---|---|
| Formal non-identifiability | Aggregate predictive gain alone does not identify relational allocation. |
| Controlled recovery | RPVA recovers a known localized relational signal under controlled conditions. |
| NFL 2023 | Domain I large-scale football audit. |
| NFL 2018 | Independent temporal replication in Domain I. |
| NFL pseudo-nearest | Relational-null falsification for arbitrary one-versus-rest isolation. |
| AV2 C_SHUFFLED | Alignment-disruption control for cross-agent future information. |
| Six held-out cities | Cross-geography robustness within Domain II development. |
| Official AV2 VAL | One-shot frozen confirmatory execution in Domain II. |

## Official AV2 Confirmation

The authors' official confirmatory execution was run once under the frozen protocol.

| Quantity | Estimate | 95% CI |
|---|---:|---:|
| Delta H to C | 0.441 | 0.420-0.463 |
| lambda | 0.126 | 0.086-0.167 |

Pseudo-nearest P = 0.00020. Gate 1 PASS. Gate 2 PASS.

## Installation

```bash
pip install -e .
```

## Minimal Example

```bash
python examples/minimal_rpva_example.py
```

## Tests

```bash
pytest
```

## Repository Structure

- `rpva/`: generic RPVA code.
- `tests/`: generic package tests.
- `examples/`: runnable examples.
- `nfl_example/`: Domain I NFL 2023 public reproducibility materials.
- `nfl_external_validation/`: Domain I independent 2018 validation package.
- `derived_outputs/`: preserved public aggregate outputs.
- `av2_cross_domain/`: Domain II AV2 public-safe protocol, aggregate results, provenance, and source-code review.
- `figures/`: public-safe final submission figure assets.
- `docs/`: reproducibility, governance, data-access, and claim-evidence documentation.

## Domain I - NFL

Domain I establishes temporal reproducibility. The repository preserves the original NFL 2023 package and the independent 2018 temporal replication under harmonized frozen logic, including aggregate outputs and pseudo-nearest summaries.

## Domain II - Argoverse 2

Domain II provides independent cross-domain confirmation in Argoverse 2. The public-safe package includes protocol documentation, aggregate frozen results, governance provenance, environment notes, and a source-code snapshot where public-safety review allows redistribution.

## Reproducibility Scope

Executable where complete; auditable/provenance-reproducible where source data or restricted artifacts cannot be redistributed.

- `rpva/`: executable generic framework and tests.
- `nfl_external_validation/`: NFL 2018 external-validation implementation, executable subject to access to the required authorized/public NFL source data and documented dependencies.
- `nfl_example/`: NFL 2023 implementation scaffolding plus aggregate/provenance outputs; this public package should not be described as a verified full end-to-end reproduction where guarded scripts state that verification is outside the public scaffolding package.
- `av2_cross_domain/`: AV2 public-safe protocol, aggregate frozen outputs, provenance, and redistributable frozen source snapshots. Full end-to-end AV2 rerun requires separately obtained official AV2 source data and restricted intermediate artifacts that are not redistributed here.

## Data Access

NFL source data must be obtained from the official Kaggle NFL Big Data Bowl sources. AV2 source data must be obtained from the official Argoverse 2 Motion Forecasting source. Unofficial mirrors are not used.

## Citation

See `CITATION.cff`.

## Limitations

The evidence supports the prespecified relational predictive-value claims in the studied domains and frozen protocols. It does not establish universal validity across all multi-agent systems, all sports, all autonomy datasets, or all possible predictive endpoints.

## License / Rights

No software license is currently assigned. All rights are reserved by the authors unless permission is granted separately.
