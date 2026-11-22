from tmgen.models import lognormal_tm, modulated_gravity_tm


def test_modulated_gravity():
    tm = modulated_gravity_tm(num_nodes=2, num_tms=10, mean_traffic=1.0)

    assert tm.matrix.ndim == 3
    assert tm.matrix.shape == (2, 2, 10)
    assert (tm.matrix >= 0).all()


def test_modulated_gravity_many_epochs():
    # regression test for #9: float epoch count broke numpy.linspace
    tm = modulated_gravity_tm(
        num_nodes=14, num_tms=24, mean_traffic=3750, pm_ratio=1.5, t_ratio=0.75
    )
    assert tm.matrix.shape == (14, 14, 24)


def test_lognormal_exported():
    assert lognormal_tm(3, 1.0, num_epochs=2).matrix.shape == (3, 3, 2)
