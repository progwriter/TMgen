import math
from collections.abc import Callable
from typing import Any

import numpy
import pytest

from tmgen.exceptions import TMgenException
from tmgen.models import (
    _peak_mean_cycle,
    _truncated_normal,
    exact_tm,
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
    tm = modulated_gravity_tm(num_nodes=2, num_epochs=10, total_traffic=1.0)

    assert tm.matrix.ndim == 3
    assert tm.matrix.shape == (2, 2, 10)
    assert (tm.matrix >= 0).all()


def test_modulated_gravity_many_epochs() -> None:
    # regression test for #9: float epoch count broke numpy.linspace
    tm = modulated_gravity_tm(
        num_nodes=14, num_epochs=24, total_traffic=3750, pm_ratio=1.5, t_ratio=0.75
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
        random_gravity_tm(3, 1.0, 7)  # type: ignore[call-arg,arg-type]


def test_spike_has_exactly_num_spikes_in_total() -> None:
    tm = spike_tm(5, 4, 2.0, num_epochs=6, rng=0)
    assert (tm.matrix > 0).sum() == 4


def test_spike_support_is_uniform_over_all_entries() -> None:
    counts = numpy.zeros((3, 3, 4))
    for seed in range(2000):
        counts += spike_tm(3, 5, 1.0, num_epochs=4, rng=seed).matrix > 0
    # each of the 36 entries is a spike with probability 5/36
    assert counts.mean() / 2000 == pytest.approx(5 / 36)
    assert counts.std() / 2000 < 0.02


def test_spike_values_are_exponential() -> None:
    tm = spike_tm(20, 300, 7.0, num_epochs=3, rng=1)
    values = tm.matrix[tm.matrix > 0]
    assert values.size == 300
    assert values.mean() == pytest.approx(7.0, rel=0.15)


@pytest.mark.parametrize("spikes", [-1, 3 * 3 * 2 + 1])
def test_spike_count_out_of_range(spikes: int) -> None:
    with pytest.raises(TMgenException):
        spike_tm(3, spikes, 1.0, num_epochs=2)


def test_spike_can_fill_every_entry() -> None:
    assert (spike_tm(2, 8, 1.0, num_epochs=2, rng=0).matrix > 0).all()


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
        num_nodes=4, num_epochs=2000, total_traffic=1000.0, pm_ratio=1.5, t_ratio=1.0
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
        num_nodes=5, num_epochs=4, total_traffic=1000.0, temporal_variance=0
    )
    exact = modulated_gravity_tm(spatial_variance=0, rng=3, **kwargs)
    noisy = modulated_gravity_tm(spatial_variance=2500, rng=3, **kwargs)
    assert not numpy.allclose(exact.matrix, noisy.matrix)
    assert (noisy.matrix >= 0).all()


def test_modulated_gravity_rejects_negative_variance() -> None:
    with pytest.raises(TMgenException):
        modulated_gravity_tm(3, 4, 10.0, spatial_variance=-1)


def test_random_gravity_normalize_gives_exact_total() -> None:
    tm = random_gravity_tm(7, 123.0, normalize=True, rng=4)
    assert tm.matrix.shape == (7, 7, 1)
    assert tm.matrix.sum() == pytest.approx(123.0)


@pytest.mark.parametrize(("symmetric", "factor"), [(False, 1.0), (True, 1.2)])
def test_random_gravity_total_is_random_around_expectation(
    symmetric: bool, factor: float
) -> None:
    num_nodes, total = 5, 100.0
    sums = numpy.array(
        [
            random_gravity_tm(num_nodes, total, symmetric=symmetric, rng=s).matrix.sum()
            for s in range(4000)
        ]
    )
    assert sums.std() > 1.0
    # 1 + 1 / num_nodes when rows and columns share their weights
    assert sums.mean() == pytest.approx(total * factor, rel=0.03)


def test_random_gravity_is_asymmetric_by_default() -> None:
    m = random_gravity_tm(6, 1.0, rng=0).matrix[:, :, 0]
    assert not numpy.allclose(m, m.T)
    assert numpy.linalg.matrix_rank(m) == 1


def test_random_gravity_symmetric_option() -> None:
    m = random_gravity_tm(6, 1.0, symmetric=True, rng=0).matrix[:, :, 0]
    numpy.testing.assert_allclose(m, m.T)


def test_random_gravity_ingress_and_egress_are_independent() -> None:
    m = random_gravity_tm(5000, 1.0, rng=0).matrix[:, :, 0]
    assert numpy.corrcoef(m.sum(axis=1), m.sum(axis=0))[0, 1] == pytest.approx(
        0, abs=0.05
    )


def test_random_gravity_weights_are_exponential() -> None:
    n = 2000
    tm = random_gravity_tm(n, 1.0, rng=0).matrix[:, :, 0]
    # the diagonal is T/N^2 * u * v with u, v ~ Exp(1) independent:
    # E[uv] = 1 and E[(uv)^2] = 4
    uv = numpy.diag(tm) * n**2
    assert uv.mean() == pytest.approx(1.0, rel=0.1)
    assert (uv**2).mean() == pytest.approx(4.0, rel=0.35)
    # a row sum is proportional to u, whose spread is that of Exp(1)
    rows = tm.sum(axis=1)
    assert rows.std() / rows.mean() == pytest.approx(1.0, rel=0.1)


_INVALID_CALLS = {
    "nodes_zero": lambda: exp_tm(0, 1.0),
    "nodes_negative": lambda: uniform_tm(-1, 0.0, 1.0),
    "nodes_fraction": lambda: exact_tm(2.5, 1.0),  # type: ignore[arg-type]
    "nodes_text": lambda: random_gravity_tm("3", 1.0),  # type: ignore[arg-type]
    "epochs_zero": lambda: exp_tm(3, 1.0, num_epochs=0),
    "epochs_negative": lambda: spike_tm(3, 1, 1.0, num_epochs=-1),
    "modulated_epochs_zero": lambda: modulated_gravity_tm(3, 0, 1.0),
    "exp_negative_mean": lambda: exp_tm(3, -1.0),
    "exp_nan_mean": lambda: exp_tm(3, float("nan")),
    "exp_infinite_mean": lambda: exp_tm(3, float("inf")),
    "lognormal_negative_sigma": lambda: lognormal_tm(3, 0.0, sigma=-1.0),
    "lognormal_nan_mu": lambda: lognormal_tm(3, float("nan")),
    "exact_negative": lambda: exact_tm(3, -1.0),
    "uniform_low_above_high": lambda: uniform_tm(3, 5.0, 1.0),
    "uniform_negative_low": lambda: uniform_tm(3, -1.0, 1.0),
    "random_gravity_negative_total": lambda: random_gravity_tm(3, -1.0),
    "gravity_negative_population": lambda: gravity_tm([1.0, -1.0], 1.0),
    "gravity_zero_populations": lambda: gravity_tm([0.0, 0.0], 1.0),
    "gravity_empty": lambda: gravity_tm([], 1.0),
    "gravity_2d": lambda: gravity_tm([[1.0, 2.0]], 1.0),
    "gravity_text": lambda: gravity_tm(["a", "b"], 1.0),
    "gravity_nan": lambda: gravity_tm([1.0, float("nan")], 1.0),
    "gravity_negative_total": lambda: gravity_tm([1.0, 2.0], -1.0),
    "spike_too_many": lambda: spike_tm(2, 5, 1.0),
    "spike_negative": lambda: spike_tm(2, -1, 1.0),
    "spike_fraction": lambda: spike_tm(2, 1.5, 1.0),  # type: ignore[arg-type]
    "spike_negative_mean": lambda: spike_tm(2, 1, -1.0),
    "modulated_zero_mean": lambda: modulated_gravity_tm(3, 4, 0.0),
    "modulated_nan_freq": lambda: modulated_gravity_tm(
        3, 4, 1.0, diurnal_freq=float("nan")
    ),
}


@pytest.mark.parametrize("name", _INVALID_CALLS)
def test_invalid_input_raises_tmgen_exception(name: str) -> None:
    with pytest.raises(TMgenException):
        _INVALID_CALLS[name]()


def test_invalid_input_can_be_caught_as_value_error() -> None:
    with pytest.raises(ValueError, match="num_nodes"):
        exp_tm(0, 1.0)


def test_equal_bounds_and_integral_floats_are_accepted() -> None:
    assert (uniform_tm(2, 3.0, 3.0).matrix == 3.0).all()
    assert exp_tm(3.0, 1.0, num_epochs=2.0).matrix.shape == (3, 3, 2)  # type: ignore[arg-type]
    assert exact_tm(2, 0.0).matrix.sum() == 0
    assert spike_tm(2, 4, 1.0).matrix.shape == (2, 2, 1)


_RENAMES: list[tuple[Callable[..., TrafficMatrix], dict[str, Any], dict[str, Any]]] = [
    (
        lambda **kw: modulated_gravity_tm(3, rng=0, **kw),
        {"num_tms": 4, "mean_traffic": 5.0},
        {"num_epochs": 4, "total_traffic": 5.0},
    ),
    (
        lambda **kw: random_gravity_tm(3, rng=0, **kw),
        {"mean_traffic": 5.0},
        {"total_traffic": 5.0},
    ),
    (
        lambda **kw: exp_tm(3, rng=0, **kw),
        {"mean_traffic": 5.0},
        {"mean_entry": 5.0},
    ),
    (
        lambda **kw: lognormal_tm(3, rng=0, **kw),
        {"mean_traffic": 1.5},
        {"mu": 1.5},
    ),
    (lambda **kw: exact_tm(3, **kw), {"val": 2.0}, {"value": 2.0}),
]


@pytest.mark.parametrize(("call", "old", "new"), _RENAMES)
def test_old_parameter_names_still_work_but_warn(
    call: Callable[..., TrafficMatrix], old: dict[str, Any], new: dict[str, Any]
) -> None:
    with pytest.warns(DeprecationWarning, match="is deprecated, use"):
        from_old = call(**old)
    numpy.testing.assert_array_equal(from_old.matrix, call(**new).matrix)
