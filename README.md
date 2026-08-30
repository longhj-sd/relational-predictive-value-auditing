# Relational Predictive-Value Auditing

Relational predictive-value auditing (RPVA) is a held-out evaluation procedure for decomposing the incremental predictive value of nested information states across predefined relational positions in multi-agent predictions.

## What RPVA answers

RPVA asks where the held-out value of an added information state is expressed across predefined roles.

## What RPVA does not claim

RPVA is not a universal model evaluation, causal information allocation method, state-of-the-art predictor, production-ready system, live decision-support tool, or evidence of validation across sports.

## Workflow

Held-out predictions -> paired losses -> predictive gains -> role aggregation -> contrasts and uncertainty.

## Installation

```bash
pip install -e .
```

## Minimal example

```bash
python examples/minimal_rpva_example.py
```

Software usage example; not an additional validation study.

## Input schema

Required: `event_id`, `agent_id`, `role`, `information_state`, `y_true`, `y_pred`. Optional: `context`, `cluster_id`.

## Outputs

Agent-level loss, paired gain, role-specific gains, weighted aggregate gain, prespecified contrasts, context modulation, confidence intervals, and optional null diagnostics.

## NFL implementation and validation materials

NFL-specific interface templates are separated under `nfl_example/` and require local official data paths. They document the expected command-line entry points for preparing data, fitting models, running RPVA, and reproducing tables and figures from locally obtained official competition files. For the 2023 analysis, these remain implementation scaffolding rather than a verified full archived raw-data reconstruction pipeline.

The independent 2018 external-season replication package is under `nfl_external_validation/2018/`. It includes the frozen protocol, executable reconstruction script, aggregate expected outputs, release verifier, environment notes, reports, and reviewer data-access instructions. It reconstructs the 2018 aggregate validation outputs from locally obtained authorized NFL Big Data Bowl 2021 files and the documented `targetedReceiver.csv` source.

## Data access boundary

Raw NFL tracking files are not redistributed. Users must obtain them from the official competition source and follow access terms.

## Tests

```bash
pytest
```

## Repository structure

Generic code is under `rpva/`; NFL-specific implementation scaffolding is under `nfl_example/`; permitted derived summaries are under `derived_outputs/`.

## Reproducibility

Frozen audit configuration summaries and permitted derived summaries are included. The 2018 external-season package has an automated verifier and documented command-line interface for reconstruction from authorized local competition files. The 2023 benchmark evidence is not upgraded by the 2018 package and should still be read as released implementation/audit scaffolding unless the full archived 2023 raw-data pipeline is independently recovered and verified.

## Authorship and maintenance

This repository was created and is maintained by Haojie Long, Longji Li, and Lifeng Zhang, the authors of the associated manuscript. The authors are responsible for the scientific design, analytical protocol, interpretation, released code, documentation, and versioned outputs.

## Citation

See `CITATION.cff`.

## License

No software license has currently been assigned. All rights are reserved by the authors unless permission is granted separately.

## Limitations

Role definitions must be justified before analysis. Nested information states do not automatically have causal interpretation. Leakage control is the application user's responsibility. The NFL example does not prove cross-domain validity.
