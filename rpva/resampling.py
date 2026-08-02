from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Mapping


def cluster_bootstrap_mean(rows: Iterable[Mapping[str, object]], value_key: str = "gain", cluster_key: str = "cluster_id", n_resamples: int = 200, seed: int = 1) -> list[float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[cluster_key]].append(float(row[value_key]))
    clusters = list(grouped)
    rng = random.Random(seed)
    values = []
    for _ in range(n_resamples):
        sample = []
        for _ in clusters:
            sample.extend(grouped[rng.choice(clusters)])
        values.append(sum(sample) / len(sample))
    return values
