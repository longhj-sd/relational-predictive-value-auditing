from rpva.nulls import shuffle_roles_within_event


def test_shuffle_preserves_count():
    rows=[{'event_id':1,'role':'a'},{'event_id':1,'role':'b'}]
    assert len(shuffle_roles_within_event(rows)) == 2
