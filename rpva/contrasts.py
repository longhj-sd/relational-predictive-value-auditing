from __future__ import annotations

from typing import Mapping


def compute_contrasts(role_gains: Mapping[str, float], contrasts: Mapping[str, tuple[str, str]]) -> dict[str, float]:
    out = {}
    for name, (left, right) in contrasts.items():
        out[name] = float(role_gains[left]) - float(role_gains[right])
    return out
