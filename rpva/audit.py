from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence

Record = Mapping[str, object]


def squared_error(y_true: float, y_pred: float) -> float:
    return (float(y_true) - float(y_pred)) ** 2


def compute_losses(records: Iterable[Record], loss_fn: Callable[[float, float], float] = squared_error) -> list[dict]:
    out = []
    for row in records:
        value = dict(row)
        value["loss"] = loss_fn(float(row["y_true"]), float(row["y_pred"]))
        out.append(value)
    return out


def compute_paired_gains(loss_rows: Iterable[Record], state_order: Sequence[str]) -> list[dict]:
    by_key: dict[tuple[object, object], dict[str, Record]] = defaultdict(dict)
    for row in loss_rows:
        by_key[(row["event_id"], row["agent_id"])][str(row["information_state"])] = row
    gains = []
    for (event_id, agent_id), states in by_key.items():
        for prev_state, next_state in zip(state_order, state_order[1:]):
            if prev_state not in states or next_state not in states:
                continue
            prev = states[prev_state]
            nxt = states[next_state]
            gains.append(
                {
                    "event_id": event_id,
                    "agent_id": agent_id,
                    "role": nxt["role"],
                    "from_state": prev_state,
                    "to_state": next_state,
                    "gain": float(prev["loss"]) - float(nxt["loss"]),
                    "context": nxt.get("context"),
                    "cluster_id": nxt.get("cluster_id", event_id),
                }
            )
    return gains


def aggregate_role_gains(gains: Iterable[Record], within_event: str = "mean") -> dict[str, float]:
    if within_event != "mean":
        raise ValueError("Only mean within-event aggregation is implemented in the reference example.")
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in gains:
        buckets[str(row["role"])].append(float(row["gain"]))
    return {role: sum(values) / len(values) for role, values in buckets.items() if values}


def run_rpva(records: Iterable[Record], state_order: Sequence[str], loss_fn: Callable[[float, float], float] = squared_error) -> dict:
    losses = compute_losses(records, loss_fn)
    gains = compute_paired_gains(losses, state_order)
    return {"losses": losses, "gains": gains, "role_gains": aggregate_role_gains(gains)}
