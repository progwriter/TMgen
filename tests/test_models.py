# coding: utf-8

from tmgen.models import modulated_gravity_tm


def test_modulated_gravity():
    tm = modulated_gravity_tm(num_nodes=2, num_tms=10, mean_traffic=1.0)

    # test number of dimensions
    assert tm.matrix.ndim == 3
    assert tm.matrix.shape == (2, 2, 10)

    # test non-negativity
    assert tm.matrix.all() > 0
