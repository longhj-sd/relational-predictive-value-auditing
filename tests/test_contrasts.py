from rpva.contrasts import compute_contrasts


def test_contrast():
    assert compute_contrasts({'a':3,'b':1}, {'a_b':('a','b')})['a_b'] == 2.0
