from rpva.audit import aggregate_role_gains


def test_role_aggregation_mean():
    assert aggregate_role_gains([{'role':'a','gain':1},{'role':'a','gain':3}]) == {'a': 2.0}
