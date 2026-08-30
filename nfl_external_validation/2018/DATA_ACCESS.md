# Data Access

This repository does not redistribute NFL Big Data Bowl raw data or row-level derived tracking files.

Reviewers need two independently obtained sources:

1. NFL Big Data Bowl 2021 competition files containing `games.csv`, `plays.csv`, `players.csv`, and weekly tracking files `week1.csv` through `week17.csv`.
2. The target-receiver label file `targetedReceiver.csv` from `tombliss/nfl-big-data-bowl-2021-bonus`.

The replication script links target labels to plays using `gameId` and `playId`, with `targetNflId` identifying the targeted receiver. The file is passed by local path through `--target-label-file`.

Only aggregate summaries, protocol files, reports, and executable code are included in this release.
