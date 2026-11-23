from collections.abc import Callable

import numpy
import pytest

from tmgen.models import (
    _peak_mean_cycle,
    exp_tm,
    gravity_tm,
    lognormal_tm,
    modulated_gravity_tm,
    random_gravity_tm,
    spike_tm,
    uniform_tm,
)
from tmgen.tm import TrafficMatrix

_RANDOM_MODELS: dict[str, Callable[..., TrafficMatrix]] = {
    "modulated_gravity": lambda **kw: modulated_gravity_tm(4, 6, 10.0, **kw),
    "random_gravity": lambda **kw: random_gravity_tm(4, 10.0, **kw),
    "uniform": lambda **kw: uniform_tm(4, 1.0, 5.0, num_epochs=3, **kw),
    "exp": lambda **kw: exp_tm(4, 2.0, num_epochs=3, **kw),
    "spike": lambda **kw: spike_tm(4, 3, 2.0, num_epochs=3, **kw),
    "lognormal": lambda **kw: lognormal_tm(4, 1.0, num_epochs=3, **kw),
}


def test_modulated_gravity() -> None:
    tm = modulated_gravity_tm(num_nodes=2, num_tms=10, mean_traffic=1.0)

    assert tm.matrix.ndim == 3
    assert tm.matrix.shape == (2, 2, 10)
    assert (tm.matrix >= 0).all()


def test_modulated_gravity_many_epochs() -> None:
    # regression test for #9: float epoch count broke numpy.linspace
    tm = modulated_gravity_tm(
        num_nodes=14, num_tms=24, mean_traffic=3750, pm_ratio=1.5, t_ratio=0.75
    )
    assert tm.matrix.shape == (14, 14, 24)


def test_lognormal_exported() -> None:
    assert lognormal_tm(3, 1.0, num_epochs=2).matrix.shape == (3, 3, 2)


@pytest.mark.parametrize("name", _RANDOM_MODELS)
def test_int_seed_is_reproducible(name: str) -> None:
    model = _RANDOM_MODELS[name]
    a, b = model(rng=7), model(rng=7)
    assert isinstance(a, TrafficMatrix)
    numpy.testing.assert_array_equal(a.matrix, b.matrix)
    assert not numpy.array_equal(a.matrix, model(rng=8).matrix)


@pytest.mark.parametrize("name", _RANDOM_MODELS)
def test_generator_matches_int_seed(name: str) -> None:
    model = _RANDOM_MODELS[name]
    numpy.testing.assert_array_equal(
        model(rng=numpy.random.default_rng(7)).matrix, model(rng=7).matrix
    )


@pytest.mark.parametrize("name", _RANDOM_MODELS)
def test_rng_does_not_touch_global_state(name: str) -> None:
    numpy.random.seed(1)
    expected = numpy.random.random(3)
    numpy.random.seed(1)
    _RANDOM_MODELS[name](rng=7)
    numpy.testing.assert_array_equal(numpy.random.random(3), expected)


@pytest.mark.parametrize("name", _RANDOM_MODELS)
def test_default_uses_global_seed(name: str) -> None:
    model = _RANDOM_MODELS[name]
    numpy.random.seed(3)
    a = model()
    numpy.random.seed(3)
    b = model()
    numpy.testing.assert_array_equal(a.matrix, b.matrix)


@pytest.mark.parametrize("name", _RANDOM_MODELS)
def test_rng_is_keyword_only(name: str) -> None:
    with pytest.raises(TypeError):
        random_gravity_tm(3, 1.0, 7)  # type: ignore[call-arg]


def test_spike_has_exactly_num_spikes_per_epoch() -> None:
    tm = spike_tm(5, 4, 2.0, num_epochs=6, rng=0)
    assert ((tm.matrix > 0).sum(axis=(0, 1)) == 4).all()


def test_gravity_matches_definition() -> None:
    pop = numpy.array([1.0, 2.0, 3.0])
    tm = gravity_tm(pop, 12.0)
    assert tm.matrix.shape == (3, 3, 1)
    numpy.testing.assert_allclose(tm.matrix[:, :, 0], numpy.outer(pop, pop) / 36 * 12)
    assert tm.matrix.sum() == pytest.approx(12.0)


def test_modulated_gravity_scales_base_by_sinusoid() -> None:
    tm = modulated_gravity_tm(3, 5, 10.0, rng=1)
    ratio = tm.matrix / tm.matrix[:, :, :1]
    numpy.testing.assert_allclose(ratio, numpy.broadcast_to(ratio[0, 0], ratio.shape))


@pytest.mark.parametrize("num_nodes", [2, 10])
def test_modulated_gravity_epoch_totals_follow_sinusoid(num_nodes: int) -> None:
    tm = modulated_gravity_tm(num_nodes, 48, 100.0, pm_ratio=2, t_ratio=1, rng=0)
    totals = tm.matrix.sum(axis=(0, 1))
    expected = _peak_mean_cycle(1 / 24, 48, 100.0, 2, 1)
    numpy.testing.assert_allclose(totals, expected, atol=1e-9)
    assert totals.mean() == pytest.approx(100.0, rel=0.05)
