import math
from collections.abc import Callable
from typing import Any

import numpy
import pytest

from tmgen.exceptions import TMgenException
from tmgen.models import (
    _peak_mean_cycle,
    _truncated_normal,
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
    tm = modulated_gravity_tm(
        num_nodes,
        48,
        100.0,
        pm_ratio=2,
        t_ratio=1,
        rng=0,
    )
    totals = tm.matrix.sum(axis=(0, 1))
    expected = _peak_mean_cycle(1 / 24, 48, 100.0, 2, 1)
    numpy.testing.assert_allclose(totals, expected, atol=1e-9)
    assert totals.mean() == pytest.approx(100.0, rel=0.05)


def _truncated_mean_std(mu: float, sigma: float) -> tuple[float, float]:
    alpha = -mu / sigma
    pdf = math.exp(-(alpha**2) / 2) / math.sqrt(2 * math.pi)
    tail = 0.5 * math.erfc(alpha / math.sqrt(2))
    lam = pdf / tail
    mean = mu + sigma * lam
    var = sigma**2 * (1 + alpha * lam - lam**2)
    return mean, math.sqrt(var)


@pytest.mark.parametrize("mu", [-3.0, -0.5, 0.0, 0.5, 2.0])
def test_truncated_normal_moments(mu: float) -> None:
    gen = numpy.random.default_rng(0)
    x = _truncated_normal(gen, numpy.full(200_000, mu), 1.0)
    mean, std = _truncated_mean_std(mu, 1.0)
    assert (x >= 0).all()
    assert x.mean() == pytest.approx(mean, rel=0.02)
    assert x.std() == pytest.approx(std, rel=0.03)


def test_truncated_normal_zero_std_returns_clipped_mean() -> None:
    gen = numpy.random.default_rng(0)
    out = _truncated_normal(gen, numpy.array([-1.0, 0.0, 2.5]), 0.0)
    numpy.testing.assert_array_equal(out, [0.0, 0.0, 2.5])


def test_modulated_gravity_temporal_variance_adds_noise() -> None:
    kwargs: dict[str, Any] = dict(
        num_nodes=4, num_tms=2000, mean_traffic=1000.0, pm_ratio=1.5, t_ratio=1.0
    )
    quiet = modulated_gravity_tm(
        spatial_variance=0, temporal_variance=0, rng=0, **kwargs
    )
    noisy = modulated_gravity_tm(
        spatial_variance=0, temporal_variance=100, rng=0, **kwargs
    )
    deviation = noisy.matrix.sum(axis=(0, 1)) - quiet.matrix.sum(axis=(0, 1))
    # the standard deviation of the total is sqrt(temporal_variance)
    assert deviation.std() == pytest.approx(10.0, rel=0.1)
    assert deviation.mean() == pytest.approx(0.0, abs=1.0)


def test_modulated_gravity_spatial_variance_changes_shares() -> None:
    kwargs: dict[str, Any] = dict(
        num_nodes=5, num_tms=4, mean_traffic=1000.0, temporal_variance=0
    )
    exact = modulated_gravity_tm(spatial_variance=0, rng=3, **kwargs)
    noisy = modulated_gravity_tm(spatial_variance=2500, rng=3, **kwargs)
    assert not numpy.allclose(exact.matrix, noisy.matrix)
    assert (noisy.matrix >= 0).all()


def test_modulated_gravity_rejects_negative_variance() -> None:
    with pytest.raises(TMgenException):
        modulated_gravity_tm(3, 4, 10.0, spatial_variance=-1)
