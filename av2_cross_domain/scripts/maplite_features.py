from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any


MAP_LITE_FEATURES = [
    "map_nearest_lane_heading",
    "map_target_to_lane_heading",
    "map_lateral_lane_coord",
    "map_longitudinal_lane_coord",
    "map_is_intersection",
    "map_turn_direction_left",
    "map_turn_direction_right",
    "map_turn_direction_straight",
    "map_left_neighbor_present",
    "map_right_neighbor_present",
    "map_predecessor_count",
    "map_successor_count",
    "map_target_av_same_lane",
    "map_target_av_lane_heading_diff",
    "map_target_in_drivable_area",
    "map_distance_to_drivable_boundary",
]


def empty_maplite_values() -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for name in MAP_LITE_FEATURES:
        out[name] = 0.0
        out[f"{name}_missing"] = 1
    return out


def wrap_angle(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


def _xy(point: dict[str, Any]) -> tuple[float, float]:
    return float(point["x"]), float(point["y"])


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _segment_projection(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> tuple[float, tuple[float, float], float, float, float]:
    vx = b[0] - a[0]
    vy = b[1] - a[1]
    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 0.0:
        return _dist2(p, a), a, 0.0, 0.0, 0.0
    t = ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / seg_len2
    t_clamped = min(1.0, max(0.0, t))
    proj = (a[0] + t_clamped * vx, a[1] + t_clamped * vy)
    cross = vx * (p[1] - proj[1]) - vy * (p[0] - proj[0])
    seg_len = math.sqrt(seg_len2)
    signed_lateral = cross / seg_len
    return _dist2(p, proj), proj, t_clamped, signed_lateral, seg_len


def _point_to_polyline(
    p: tuple[float, float], points: list[tuple[float, float]]
) -> dict[str, float | int] | None:
    if len(points) < 2:
        return None
    cumulative = 0.0
    best: dict[str, float | int] | None = None
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        d2, _proj, t, signed_lateral, seg_len = _segment_projection(p, a, b)
        s_coord = cumulative + t * seg_len
        heading = math.atan2(b[1] - a[1], b[0] - a[0])
        candidate = {
            "distance": math.sqrt(d2),
            "distance2": d2,
            "segment_index": i,
            "heading": heading,
            "lateral": signed_lateral,
            "longitudinal": s_coord,
        }
        if best is None or (d2, i) < (float(best["distance2"]), int(best["segment_index"])):
            best = candidate
        cumulative += seg_len
    return best


def _point_in_polygon(p: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    if len(poly) < 3:
        return False
    x, y = p
    inside = False
    prev = poly[-1]
    for curr in poly:
        x1, y1 = prev
        x2, y2 = curr
        if min(y1, y2) <= y <= max(y1, y2):
            d2, _proj, _t, _lat, _seg_len = _segment_projection(p, prev, curr)
            if d2 <= 1e-12:
                return True
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x <= x_cross:
                inside = not inside
        prev = curr
    return inside


def _point_to_polygon_boundary_distance(p: tuple[float, float], poly: list[tuple[float, float]]) -> float | None:
    if len(poly) < 2:
        return None
    best = math.inf
    closed = poly + [poly[0]]
    for a, b in zip(closed[:-1], closed[1:]):
        d2, _proj, _t, _lat, _seg_len = _segment_projection(p, a, b)
        best = min(best, math.sqrt(d2))
    return best if math.isfinite(best) else None


def _turn_direction(centerline: list[tuple[float, float]]) -> str:
    if len(centerline) < 3:
        return "straight"
    first = math.atan2(centerline[1][1] - centerline[0][1], centerline[1][0] - centerline[0][0])
    last = math.atan2(centerline[-1][1] - centerline[-2][1], centerline[-1][0] - centerline[-2][0])
    delta = wrap_angle(last - first)
    if delta > math.radians(20.0):
        return "left"
    if delta < -math.radians(20.0):
        return "right"
    return "straight"


@lru_cache(maxsize=4096)
def load_map_index(map_path: str) -> dict[str, Any]:
    path = Path(map_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    lanes = []
    for lane_id, lane in raw.get("lane_segments", {}).items():
        centerline = [_xy(p) for p in lane.get("centerline", [])]
        if lane.get("lane_type") != "VEHICLE" or len(centerline) < 2:
            continue
        lanes.append(
            {
                "id": str(lane.get("id", lane_id)),
                "centerline": centerline,
                "is_intersection": bool(lane.get("is_intersection", False)),
                "turn_direction": _turn_direction(centerline),
                "left_neighbor_id": lane.get("left_neighbor_id"),
                "right_neighbor_id": lane.get("right_neighbor_id"),
                "predecessors": lane.get("predecessors") or [],
                "successors": lane.get("successors") or [],
            }
        )
    lanes.sort(key=lambda x: x["id"])
    drivable = []
    for area_id, area in raw.get("drivable_areas", {}).items():
        boundary = [_xy(p) for p in area.get("area_boundary", [])]
        if len(boundary) >= 3:
            drivable.append({"id": str(area.get("id", area_id)), "boundary": boundary})
    drivable.sort(key=lambda x: x["id"])
    return {
        "schema_keys": sorted(raw.keys()),
        "lane_count_raw": len(raw.get("lane_segments", {})),
        "vehicle_lane_count": len(lanes),
        "drivable_area_count": len(drivable),
        "lanes": lanes,
        "drivable_areas": drivable,
    }


def nearest_lane(index: dict[str, Any], xy: tuple[float, float]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for lane in index["lanes"]:
        proj = _point_to_polyline(xy, lane["centerline"])
        if proj is None:
            continue
        key = (float(proj["distance2"]), lane["id"], int(proj["segment_index"]))
        if best is None or key < best["tie_key"]:
            best = {"lane": lane, "projection": proj, "tie_key": key}
    return best


def drivable_features(index: dict[str, Any], xy: tuple[float, float]) -> tuple[int | None, float | None]:
    if not index["drivable_areas"]:
        return None, None
    inside_any = False
    best_distance = math.inf
    for area in index["drivable_areas"]:
        boundary = area["boundary"]
        inside_any = inside_any or _point_in_polygon(xy, boundary)
        dist = _point_to_polygon_boundary_distance(xy, boundary)
        if dist is not None:
            best_distance = min(best_distance, dist)
    if not math.isfinite(best_distance):
        return None, None
    return int(inside_any), best_distance


def compute_maplite_features(
    map_path: str | Path,
    target_xy_49: tuple[float, float],
    target_heading_49: float,
    av_xy_49: tuple[float, float],
    av_heading_49: float,
) -> dict[str, float | int]:
    try:
        index = load_map_index(str(map_path))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return empty_maplite_values()

    target_lane = nearest_lane(index, target_xy_49)
    av_lane = nearest_lane(index, av_xy_49)
    in_area, boundary_distance = drivable_features(index, target_xy_49)
    if target_lane is None:
        out = empty_maplite_values()
        if in_area is not None:
            out["map_target_in_drivable_area"] = in_area
            out["map_target_in_drivable_area_missing"] = 0
        if boundary_distance is not None:
            out["map_distance_to_drivable_boundary"] = float(boundary_distance)
            out["map_distance_to_drivable_boundary_missing"] = 0
        return out

    lane = target_lane["lane"]
    proj = target_lane["projection"]
    lane_heading_fixed = wrap_angle(float(proj["heading"]) - av_heading_49)
    target_heading_fixed = wrap_angle(target_heading_49 - av_heading_49)
    target_to_lane = wrap_angle(target_heading_fixed - lane_heading_fixed)
    turn = lane["turn_direction"]

    out: dict[str, float | int] = {
        "map_nearest_lane_heading": lane_heading_fixed,
        "map_target_to_lane_heading": target_to_lane,
        "map_lateral_lane_coord": float(proj["lateral"]),
        "map_longitudinal_lane_coord": float(proj["longitudinal"]),
        "map_is_intersection": int(lane["is_intersection"]),
        "map_turn_direction_left": int(turn == "left"),
        "map_turn_direction_right": int(turn == "right"),
        "map_turn_direction_straight": int(turn == "straight"),
        "map_left_neighbor_present": int(lane["left_neighbor_id"] is not None),
        "map_right_neighbor_present": int(lane["right_neighbor_id"] is not None),
        "map_predecessor_count": int(len(lane["predecessors"])),
        "map_successor_count": int(len(lane["successors"])),
        "map_target_av_same_lane": int(av_lane is not None and av_lane["lane"]["id"] == lane["id"]),
        "map_target_av_lane_heading_diff": 0.0,
        "map_target_in_drivable_area": int(in_area) if in_area is not None else 0.0,
        "map_distance_to_drivable_boundary": float(boundary_distance) if boundary_distance is not None else 0.0,
    }
    if av_lane is not None:
        av_heading_fixed_lane = wrap_angle(float(av_lane["projection"]["heading"]) - av_heading_49)
        out["map_target_av_lane_heading_diff"] = wrap_angle(lane_heading_fixed - av_heading_fixed_lane)
    for name in MAP_LITE_FEATURES:
        out[f"{name}_missing"] = 0
    if av_lane is None:
        out["map_target_av_same_lane_missing"] = 1
        out["map_target_av_lane_heading_diff_missing"] = 1
    if in_area is None:
        out["map_target_in_drivable_area_missing"] = 1
    if boundary_distance is None:
        out["map_distance_to_drivable_boundary_missing"] = 1
    return out
