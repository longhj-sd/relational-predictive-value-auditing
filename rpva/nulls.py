from __future__ import annotations

import random
from typing import Iterable, Mapping


def shuffle_roles_within_event(rows: Iterable[Mapping[str, object]], seed: int = 1) -> list[dict]:
    rng = random.Random(seed)
    by_event = {}
    for row in rows:
        by_event.setdefault(row["event_id"], []).append(dict(row))
    out = []
    for event_rows in by_event.values():
        roles = [r["role"] for r in event_rows]
        rng.shuffle(roles)
        for row, role in zip(event_rows, roles):
            row["role"] = role
            out.append(row)
    return out
