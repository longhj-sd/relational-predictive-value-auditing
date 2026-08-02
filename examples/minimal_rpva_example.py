from rpva.audit import run_rpva
from rpva.contrasts import compute_contrasts

records = [
    {"event_id": 1, "agent_id": "a", "role": "role_1", "information_state": "S0", "y_true": 10, "y_pred": 8, "cluster_id": 1},
    {"event_id": 1, "agent_id": "a", "role": "role_1", "information_state": "S1", "y_true": 10, "y_pred": 9, "cluster_id": 1},
    {"event_id": 1, "agent_id": "b", "role": "role_2", "information_state": "S0", "y_true": 5, "y_pred": 3, "cluster_id": 1},
    {"event_id": 1, "agent_id": "b", "role": "role_2", "information_state": "S1", "y_true": 5, "y_pred": 4, "cluster_id": 1},
    {"event_id": 2, "agent_id": "a", "role": "role_1", "information_state": "S0", "y_true": 2, "y_pred": 0, "cluster_id": 2},
    {"event_id": 2, "agent_id": "a", "role": "role_1", "information_state": "S1", "y_true": 2, "y_pred": 1.5, "cluster_id": 2},
    {"event_id": 2, "agent_id": "b", "role": "role_2", "information_state": "S0", "y_true": 8, "y_pred": 7, "cluster_id": 2},
    {"event_id": 2, "agent_id": "b", "role": "role_2", "information_state": "S1", "y_true": 8, "y_pred": 7.5, "cluster_id": 2},
]

result = run_rpva(records, ["S0", "S1"])
contrast = compute_contrasts(result["role_gains"], {"role_1_minus_role_2": ("role_1", "role_2")})
print({"role_gains": result["role_gains"], "contrasts": contrast})
