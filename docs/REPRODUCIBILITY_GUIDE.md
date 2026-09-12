# Reproducibility Guide

## 1. Generic installation

```bash
pip install -e .
```

## 2. Generic RPVA tests

```bash
python -m pytest
```

## 3. Minimal example

```bash
python examples/minimal_rpva_example.py
```

## 4. NFL 2018 reproduction entry point

Use the preserved `nfl_external_validation/` package. It is designed for authorized local copies of the official NFL Big Data Bowl source files.

## 5. NFL source-data boundary

Raw NFL tracking files are not redistributed. Obtain them from the official Kaggle NFL Big Data Bowl sources.

## 6. AV2 data-access boundary

Raw AV2 scenarios and row-level restricted derivatives are not redistributed. Obtain source data from the official Argoverse 2 Motion Forecasting source under the applicable terms.

## 7. AV2 public-safe protocol and result verification

The AV2 public package provides protocol documents, aggregate frozen result transcriptions, governance provenance, and public-safe code where redistribution passed review. Verify aggregate transcriptions against `av2_cross_domain/aggregate_results/` and checksums in `av2_cross_domain/public_provenance/CHECKSUMS_SHA256.txt`.

## 8. Reconstruction scope

Full executable reconstruction requires authorized source data and local execution resources. Where restricted source data are absent, this repository supports audit/provenance reproduction through protocols, frozen aggregate outputs, hashes, and public-safe code snapshots.
