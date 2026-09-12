# AV2 Information States

Observed history: t0-49.

Future: t50-109.

Coordinate system: AV-centric at t49.

H: target/relational observed history + MAP-LITE.

E: H + AV endpoint at t109.

C: H + full aligned AV future t50-109.

C_SHUFFLED: same feature form as C with frozen stage-local shuffled AV future.

Primary transition: H→C.

Frozen 16 MAP-LITE base features: map_nearest_lane_heading; map_target_to_lane_heading; map_lateral_lane_coord; map_longitudinal_lane_coord; map_is_intersection; map_turn_direction_left; map_turn_direction_right; map_turn_direction_straight; map_left_neighbor_present; map_right_neighbor_present; map_predecessor_count; map_successor_count; map_target_av_same_lane; map_target_av_lane_heading_diff; map_target_in_drivable_area; map_distance_to_drivable_boundary.
