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

## NFL implementation scaffolding

NFL-specific interface templates are separated under `nfl_example/` and require local official data paths. They document the expected command-line entry points for preparing data, fitting models, running RPVA, and reproducing tables and figures from locally obtained official competition files. They are implementation scaffolding, not a verified full NFL analysis pipeline.

## Data access boundary

Raw NFL tracking files are not redistributed. Users must obtain them from the official competition source and follow access terms.

## Tests

```bash
pytest
```

## Repository structure

Generic code is under `rpva/`; NFL-specific implementation scaffolding is under `nfl_example/`; permitted derived summaries are under `derived_outputs/`.

## Reproducibility

Frozen audit configuration summaries and permitted derived summaries are included. Benchmark evidence is not rerun for release packaging. End-to-end reconstruction from raw official competition files has not been verified in this repository package.

## Authorship and maintenance

This repository was created and is maintained by Haojie Long, Longji Li, and Lifeng Zhang, the authors of the associated manuscript. The authors are responsible for the scientific design, analytical protocol, interpretation, released code, documentation, and versioned outputs.

## Citation

See `CITATION.cff`.

## License

License selection requires author confirmation. MIT is prepared as a candidate in `LICENSE_SELECTION_REQUIRED.md`.

## Limitations

Role definitions must be justified before analysis. Nested information states do not automatically have causal interpretation. Leakage control is the application user's responsibility. The NFL example does not prove cross-domain validity.
