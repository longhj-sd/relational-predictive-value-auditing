from rpva.resampling import cluster_bootstrap_mean


def test_resampling_count():
    out=cluster_bootstrap_mean([{'cluster_id':1,'gain':1},{'cluster_id':2,'gain':3}], n_resamples=5, seed=2)
    assert len(out) == 5
