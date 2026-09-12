# Excluded AV2 Source Code

Public safety takes priority over byte-for-byte redistribution. For RC3, the previously missing frozen modules `phase3a_train_only_freeze.py` and `phase3b_train_only_execution.py` were located, reviewed for credentials and machine-specific absolute paths, and included as public-safe frozen source snapshots.

No AV2 source file is excluded in RC3 solely because of the two known missing-module issues. Raw AV2 scenarios, row-level prediction dumps, restricted derivatives, and model binaries remain outside the public package.

| filename | scientific role | original_sha256 | exclusion_reason |
|---|---|---|---|
