import inspect
import math
from collections.abc import Callable
from typing import Any

import numpy
import pytest

from tmgen import models
from tmgen.exceptions import TMgenException
from tmgen.hmc import hmc_sample
from tmgen.lowrank import LowRankModel
from tmgen.models import (
    _truncated_normal,
    datacenter_tm,
    diurnal_profile,
    elephants_and_mice_tm,
    exact_tm,
    exp_tm,
    fluctuating_tm,
    fourier_profile,
    gravity_tm,
    lognormal_tm,
    modulated_gravity_tm,
    ncim_tm,
    random_gravity_tm,
    shot_noise_tm,
    spike_tm,
    time_zone_profiles,
    traffic_class_tm,
    uniform_tm,
)
from tmgen.tm import TrafficMatrix

_LOWRANK = LowRankModel.fit(modulated_gravity_tm(4, 48, 10.0, rng=0))
_HMC = (numpy.eye(3), numpy.ones(3), numpy.eye(3), numpy.zeros(3), numpy.ones(3))

_RANDOM_MODELS: dict[str, Callable[..., Any]] = {
    "modulated_gravity": lambda **kw: modulated_gravity_tm(4, 6, 10.0, **kw),
    "random_gravity": lambda **kw: random_gravity_tm(4, 10.0, **kw),
    "uniform": lambda **kw: uniform_tm(4, 1.0, 5.0, num_epochs=3, **kw),
    "exp": lambda **kw: exp_tm(4, 2.0, num_epochs=3, **kw),
    "spike": lambda **kw: spike_tm(4, 3, 2.0, num_epochs=3, **kw),
    "lognormal": lambda **kw: lognormal_tm(4, 1.0, num_epochs=3, **kw),
    "ncim": lambda **kw: ncim_tm(numpy.ones(5), 3, **kw),
    "ncim_node_traffic": lambda **kw: ncim_tm(
        numpy.ones(5),
        ingress_traffic=[1.0, 2.0, 3.0],
        egress_traffic=[3.0, 2.0, 1.0],
        **kw,
    ),
    "traffic_class": lambda **kw: traffic_class_tm(
        [1.0, 2.0], [0, 1, 1], [[0.5, 0.25], [0.125, 0.125]], **kw
    ),
    "datacenter": lambda **kw: datacenter_tm(10, 1.0, **kw),
    "elephants_and_mice": lambda **kw: elephants_and_mice_tm(5, **kw),
    "fluctuating": lambda **kw: fluctuating_tm(exact_tm(3, 5e6), 4, **kw),
    "shot_noise": lambda **kw: shot_noise_tm(
        exact_tm(2, 2.0), 20, epoch_length=5.0, **_SHOT, **kw
    ),
    "lowrank": lambda **kw: _LOWRANK.sample(10, **kw),
    "hmc": lambda **kw: hmc_sample(*_HMC, 20, **kw),
}


@pytest.mark.parametrize("name", _RANDOM_MODELS)
def test_rng_contract(name: str) -> None:
    model = _RANDOM_MODELS[name]
    a = numpy.asarray(model(rng=7))
    numpy.testing.assert_array_equal(a, numpy.asarray(model(rng=7)))
    assert not numpy.array_equal(a, numpy.asarray(model(rng=8)))
    numpy.testing.assert_array_equal(
        a, numpy.asarray(model(rng=numpy.random.default_rng(7)))
    )

    numpy.random.seed(1)
    expected = numpy.random.random(3)
    numpy.random.seed(1)
    model(rng=7)
    numpy.testing.assert_array_equal(numpy.random.random(3), expected)

    numpy.random.seed(3)
    first = numpy.asarray(model())
    numpy.random.seed(3)
    numpy.testing.assert_array_equal(first, numpy.asarray(model()))


def test_rng_is_keyword_only() -> None:
    functions: list[Callable[..., Any]] = [
        f
        for name, f in vars(models).items()
        if name in models.__all__
        and inspect.isfunction(f)
        and "rng" in inspect.signature(f).parameters
    ]
    assert len(functions) == 12
    for f in [*functions, LowRankModel.sample]:
        assert (
            inspect.signature(f).parameters["rng"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )


@pytest.mark.parametrize("model", ["random_gravity", "modulated_gravity"])
def test_gravity_is_rank_one_and_asymmetric_unless_asked(model: str) -> None:
    def make(**kw: Any) -> numpy.ndarray[Any, Any]:
        tm = (
            random_gravity_tm(6, 1.0, rng=0, **kw)
            if model == "random_gravity"
            else modulated_gravity_tm(6, 4, 100.0, rng=0, **kw)
        )
        return numpy.asarray(tm)

    m = make()
    assert not numpy.allclose(m[:, :, 0], m[:, :, 0].T)
    assert numpy.linalg.matrix_rank(m[:, :, 0]) == 1
    sym = make(symmetric=True)
    numpy.testing.assert_allclose(sym, sym.transpose(1, 0, 2))


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("use_profile", [False, True])
def test_modulated_gravity_totals_follow_the_cycle(
    symmetric: bool, use_profile: bool
) -> None:
    if use_profile:
        expected = fourier_profile(96, 100.0, [(1 / 96, 30.0, 0.0)])
        tm = modulated_gravity_tm(
            7, 96, 1.0, profile=expected, symmetric=symmetric, rng=1
        )
    else:
        expected = diurnal_profile(96, 100.0, 2, diurnal_freq=1 / 96)
        tm = modulated_gravity_tm(
            7, 96, 100.0, 2, diurnal_freq=1 / 96, symmetric=symmetric, rng=1
        )
    numpy.testing.assert_allclose(tm.matrix.sum(axis=(0, 1)), expected)
    ratio = tm.matrix / tm.matrix[:, :, :1]
    numpy.testing.assert_allclose(ratio, numpy.broadcast_to(ratio[0, 0], ratio.shape))


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
        num_nodes=4,
        num_epochs=2000,
        total_traffic=1000.0,
        peak_to_mean=1.5,
        trough_to_mean=1.0,
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
    ingress, egress = noisy.at_time(0).sum(axis=1), noisy.at_time(0).sum(axis=0)
    assert not numpy.allclose(ingress, egress)


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
    assert modulated_gravity_tm(3, 24.0, 1.0).matrix.shape == (3, 3, 24)  # type: ignore[arg-type]


_RENAMES: list[tuple[Callable[..., TrafficMatrix], dict[str, Any], dict[str, Any]]] = [
    (
        lambda **kw: modulated_gravity_tm(3, rng=0, **kw),
        {"num_tms": 4, "mean_traffic": 5.0, "pm_ratio": 1.2, "t_ratio": 0.9},
        {
            "num_epochs": 4,
            "total_traffic": 5.0,
            "peak_to_mean": 1.2,
            "trough_to_mean": 0.9,
        },
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


def test_default_cycle_is_symmetric_like_the_paper() -> None:
    # Fig. 2: peak-to-mean 1.5 swings between 50 and 150 around a mean of 100
    cycle = diurnal_profile(96, 100.0, peak_to_mean=1.5, diurnal_freq=1 / 96)
    assert cycle[0] == pytest.approx(100.0)
    assert cycle.max() == pytest.approx(150.0, rel=1e-3)
    assert cycle.min() == pytest.approx(50.0, rel=1e-3)
    assert cycle.mean() == pytest.approx(100.0)


def test_peak_to_mean_two_touches_zero() -> None:
    cycle = diurnal_profile(96, 100.0, peak_to_mean=2, diurnal_freq=1 / 96)
    assert cycle.max() == pytest.approx(200.0, rel=1e-3)
    assert cycle.min() == pytest.approx(0.0, abs=0.1)


def test_trough_to_mean_is_the_lowest_value_over_the_mean() -> None:
    cycle = diurnal_profile(
        96, 10.0, peak_to_mean=1.5, trough_to_mean=0.75, diurnal_freq=1 / 96
    )
    assert cycle.max() == pytest.approx(15.0, rel=1e-3)
    assert cycle.min() == pytest.approx(7.5, rel=1e-3)
    assert (cycle[cycle < 10.0] < 10.0).all()
    # an asymmetric cycle does not average to the mean
    assert cycle.mean() != pytest.approx(10.0, rel=1e-3)


def test_peak_to_mean_one_is_flat() -> None:
    assert (diurnal_profile(10, 5.0, peak_to_mean=1) == 5.0).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"peak_to_mean": 0.5},
        {"peak_to_mean": 2.5},
        {"peak_to_mean": 1.5, "trough_to_mean": 1.5},
        {"peak_to_mean": 1.5, "trough_to_mean": -0.1},
        {"num_epochs": 0},
        {"mean": -1.0},
        {"diurnal_freq": float("nan")},
    ],
)
def test_diurnal_profile_rejects_invalid_arguments(kwargs: dict[str, Any]) -> None:
    args: dict[str, Any] = {"num_epochs": 24, "mean": 1.0}
    args.update(kwargs)
    with pytest.raises(TMgenException):
        diurnal_profile(**args)


def test_asymmetric_peak_above_two_is_allowed_with_a_trough() -> None:
    cycle = diurnal_profile(
        96, 1.0, peak_to_mean=3, trough_to_mean=0, diurnal_freq=1 / 96
    )
    assert cycle.min() >= 0
    assert cycle.max() == pytest.approx(3.0, rel=1e-3)


def test_default_rng_is_numpys_global_legacy_stream() -> None:
    # documented behavior: numpy.random.seed keeps working, with the same
    # stream as calling the legacy numpy.random functions directly
    numpy.random.seed(5)
    model = exp_tm(3, 7.0, num_epochs=2)
    numpy.random.seed(5)
    direct = numpy.random.exponential(7.0, size=(3, 3, 2))
    numpy.testing.assert_array_equal(model.matrix, direct)


def test_fourier_profile_adds_the_harmonics_to_the_mean() -> None:
    profile = fourier_profile(48, 10.0, [(1 / 24, 4.0, 0.0), (1 / 12, 1.0, numpy.pi)])
    epochs = numpy.arange(48)
    expected = (
        10.0
        + 4 * numpy.cos(2 * numpy.pi * epochs / 24)
        - numpy.cos(2 * numpy.pi * epochs / 12)
    )
    numpy.testing.assert_allclose(profile, expected)
    assert profile.mean() == pytest.approx(10.0)


def test_fourier_profile_without_harmonics_is_constant() -> None:
    numpy.testing.assert_array_equal(fourier_profile(5, 3.0, []), numpy.full(5, 3.0))


def test_fourier_profile_rejects_a_negative_profile() -> None:
    with pytest.raises(TMgenException, match="negative"):
        fourier_profile(24, 1.0, [(1 / 24, 2.0, 0.0)])


@pytest.mark.parametrize(
    "profile",
    [numpy.ones(5), numpy.full(6, -1.0), numpy.zeros(6), numpy.full(6, numpy.nan)],
)
def test_modulated_gravity_rejects_a_bad_profile(profile: Any) -> None:
    with pytest.raises(TMgenException):
        modulated_gravity_tm(3, 6, 1.0, profile=profile)


def test_time_zone_profiles_delay_each_node() -> None:
    profile = numpy.arange(6.0)
    shifted = time_zone_profiles(profile, [0, 2, -1])
    assert shifted.shape == (3, 6)
    numpy.testing.assert_array_equal(shifted[0], profile)
    numpy.testing.assert_array_equal(shifted[1], [4, 5, 0, 1, 2, 3])
    numpy.testing.assert_array_equal(shifted[2], [1, 2, 3, 4, 5, 0])


@pytest.mark.parametrize(
    ("profile", "shifts"),
    [([], [0]), ([[1.0]], [0]), ([1.0, 2.0], []), ([1.0, 2.0], [0.5])],
)
def test_time_zone_profiles_validate(profile: Any, shifts: Any) -> None:
    with pytest.raises(TMgenException):
        time_zone_profiles(profile, shifts)


def test_ncim_entries_have_mean_total_over_n_squared() -> None:
    total = numpy.full(4000, 40.0)
    tm = ncim_tm(total, num_nodes=2, rng=0)
    assert tm.shape == (2, 2, 4000)
    numpy.testing.assert_allclose(tm.matrix.mean(axis=2), 10.0, rtol=0.1)
    # exponential: the standard deviation equals the mean
    numpy.testing.assert_allclose(tm.matrix.std(axis=2), 10.0, rtol=0.1)


def test_ncim_epoch_means_follow_the_totals() -> None:
    total = numpy.repeat([10.0, 1000.0], 2000)
    tm = ncim_tm(total, num_nodes=3, rng=0)
    means = tm.matrix.sum(axis=(0, 1))
    assert means[:2000].mean() == pytest.approx(10.0, rel=0.05)
    assert means[2000:].mean() == pytest.approx(1000.0, rel=0.05)


def test_ncim_with_node_traffic_has_the_gravity_mean() -> None:
    ingress = [6.0, 3.0, 1.0]
    egress = [2.0, 2.0, 6.0]
    tm = ncim_tm(
        numpy.full(20000, 10.0), ingress_traffic=ingress, egress_traffic=egress, rng=0
    )
    expected = numpy.outer(ingress, egress) / 10.0
    numpy.testing.assert_allclose(tm.matrix.mean(axis=2), expected, rtol=0.1)


def test_ncim_time_zones_shift_the_busy_hour_of_each_node() -> None:
    cycle = diurnal_profile(24, 10.0, peak_to_mean=1.9)
    ingress = time_zone_profiles(cycle, [0, 12])
    egress = numpy.full((2, 24), 5.0)
    total = numpy.full(24, 20.0)
    draws = [
        ncim_tm(total, ingress_traffic=ingress, egress_traffic=egress, rng=i).matrix
        for i in range(2000)
    ]
    by_ingress = numpy.mean(draws, axis=0).sum(axis=1)
    expected = ingress * egress.sum(axis=0) / total
    numpy.testing.assert_allclose(by_ingress, expected, rtol=0.2)
    assert by_ingress[0].argmax() != by_ingress[1].argmax()


def test_ncim_node_traffic_may_be_constant_in_time() -> None:
    tm = ncim_tm(
        [5.0, 5.0, 5.0], ingress_traffic=[1.0, 2.0], egress_traffic=[3.0, 4.0], rng=0
    )
    assert tm.shape == (2, 2, 3)


def test_ncim_spatial_variance_gives_truncated_normal_weights() -> None:
    kwargs: dict[str, Any] = {
        "ingress_traffic": [6.0, 4.0],
        "egress_traffic": [5.0, 5.0],
    }
    tm = ncim_tm(numpy.full(3000, 10.0), spatial_variance=1.0, rng=0, **kwargs)
    assert (tm.matrix >= 0).all()
    numpy.testing.assert_allclose(
        tm.matrix.mean(axis=2), numpy.outer([6, 4], [5, 5]) / 10, rtol=0.1
    )
    plain = ncim_tm(numpy.full(3000, 10.0), rng=0, **kwargs)
    assert tm.matrix.std() < plain.matrix.std()


def test_ncim_zero_total_gives_an_empty_epoch() -> None:
    tm = ncim_tm(
        [0.0, 5.0], ingress_traffic=[1.0, 1.0], egress_traffic=[1.0, 1.0], rng=0
    )
    assert (tm.at_time(0) == 0).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"num_nodes": 2, "spatial_variance": 1.0},
        {"ingress_traffic": [1.0, 1.0]},
        {"ingress_traffic": [1.0, 1.0], "egress_traffic": [1.0, 1.0, 1.0]},
        {"ingress_traffic": [1.0, 1.0], "egress_traffic": [1.0, 1.0], "num_nodes": 3},
        {"ingress_traffic": numpy.ones((2, 4)), "egress_traffic": [1.0, 1.0]},
        {"ingress_traffic": [1.0, -1.0], "egress_traffic": [1.0, 1.0]},
    ],
)
def test_ncim_validates(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TMgenException):
        ncim_tm(numpy.ones(3), **kwargs)


@pytest.mark.parametrize("total", [[], [[1.0]], [-1.0], [numpy.nan]])
def test_ncim_validates_the_totals(total: Any) -> None:
    with pytest.raises(TMgenException):
        ncim_tm(total, num_nodes=2)


def test_traffic_class_totals_between_classes_are_exact() -> None:
    classes = [0, 0, 1, 1, 1]
    q = numpy.array([[0.5, 0.2], [0.1, 0.2]])
    total = numpy.array([100.0, 40.0, 0.0])
    tm = traffic_class_tm(total, classes, q, rng=0).matrix
    for m in range(2):
        for n in range(2):
            block = tm[numpy.ix_(numpy.array(classes) == m, numpy.array(classes) == n)]
            numpy.testing.assert_allclose(block.sum(axis=(0, 1)), total * q[m, n])
    numpy.testing.assert_allclose(tm.sum(axis=(0, 1)), total)


def test_traffic_class_fractions_may_vary_in_time() -> None:
    q = numpy.zeros((2, 2, 2))
    q[0, 1, 0] = 1.0
    q[1, 0, 1] = 1.0
    tm = traffic_class_tm([10.0, 10.0], [0, 1], q, rng=0).matrix
    assert tm[0, 1, 0] == pytest.approx(10.0)
    assert tm[1, 0, 1] == pytest.approx(10.0)
    assert tm[0, 1, 1] == 0
    assert tm[1, 0, 0] == 0


def test_traffic_class_weights_are_independent_inside_a_class() -> None:
    tm = traffic_class_tm(numpy.full(200, 10.0), [0, 0, 0], [[1.0]], rng=0).matrix
    assert numpy.linalg.matrix_rank(tm[:, :, 0]) == 1
    assert not numpy.allclose(tm, tm.transpose(1, 0, 2))


def test_traffic_class_with_one_class_is_random_gravity_with_exact_total() -> None:
    tm = traffic_class_tm([7.0], [0, 0, 0, 0], [[1.0]], rng=1)
    assert tm.shape == (4, 4, 1)
    assert tm.matrix.sum() == pytest.approx(7.0)


@pytest.mark.parametrize(
    "node_classes",
    [[], [[0, 1]], [0.5, 1.0], [-1, 0], [0, 2], ["a", "b"]],
)
def test_traffic_class_validates_the_classes(node_classes: Any) -> None:
    with pytest.raises(TMgenException):
        traffic_class_tm([1.0], node_classes, numpy.full((2, 2), 0.25))


@pytest.mark.parametrize(
    "fractions",
    [
        numpy.full((2, 2), 0.3),
        numpy.full((3, 3), 1 / 9),
        numpy.full((2, 2, 2), 0.25),
        [[1.5, -0.5], [0.0, 0.0]],
        [[numpy.nan, 0.0], [0.0, 1.0]],
    ],
)
def test_traffic_class_validates_the_fractions(fractions: Any) -> None:
    with pytest.raises(TMgenException):
        traffic_class_tm([1.0], [0, 1], fractions)


def test_traffic_class_validates_the_totals() -> None:
    with pytest.raises(TMgenException):
        traffic_class_tm([-1.0], [0, 1], numpy.full((2, 2), 0.25))


def test_datacenter_hot_nodes_carry_the_hot_load() -> None:
    tm = datacenter_tm(20, 100.0, hot_nodes=[0, 1, 2, 3], hot_load_fraction=0.55)
    m = tm.matrix[:, :, 0]
    assert m.sum() == pytest.approx(100.0)
    assert (numpy.diag(m) == 0).all()
    numpy.testing.assert_allclose(m.sum(axis=1)[:4], 55.0 / 4)
    numpy.testing.assert_allclose(m.sum(axis=0)[:4], 55.0 / 4)
    numpy.testing.assert_allclose(m.sum(axis=1)[4:], 45.0 / 16)


def test_datacenter_picks_the_default_fraction_of_hot_nodes() -> None:
    m = datacenter_tm(20, 1.0, rng=3).matrix[:, :, 0]
    assert (m.sum(axis=1) > 0.1).sum() == 4


def test_datacenter_racks_set_the_inter_rack_share() -> None:
    tm = datacenter_tm(20, 100.0, num_racks=4, inter_rack_fraction=0.7, rng=1)
    m = tm.matrix[:, :, 0]
    racks = numpy.arange(20) * 4 // 20
    assert m[racks[:, None] != racks[None, :]].sum() == pytest.approx(70.0)
    assert m.sum() == pytest.approx(100.0)


def test_datacenter_without_skew_is_uniform_off_the_diagonal() -> None:
    m = datacenter_tm(5, 20.0, hot_nodes=[0, 1, 2, 3, 4], hot_load_fraction=1).matrix
    off = ~numpy.eye(5, dtype=bool)
    numpy.testing.assert_allclose(m[:, :, 0][off], 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_nodes": 1},
        {"hot_load_fraction": 1.5},
        {"hot_load_fraction": 0.9, "hot_node_fraction": 0.05},
        {"hot_node_fraction": 2},
        {"hot_nodes": []},
        {"hot_nodes": [10]},
        {"hot_nodes": list(range(10)), "hot_load_fraction": 0.5},
        {"num_racks": 2},
        {"inter_rack_fraction": 0.5},
        {"num_racks": 11, "inter_rack_fraction": 0.5},
        {"num_racks": 2, "inter_rack_fraction": 1.5},
        {"num_racks": 10, "inter_rack_fraction": 0.5},
        {
            "num_racks": 2,
            "inter_rack_fraction": 0.99,
            "hot_nodes": [0],
            "hot_load_fraction": 0.45,
        },
    ],
)
def test_datacenter_validates(kwargs: dict[str, Any]) -> None:
    args: dict[str, Any] = {"num_nodes": 10, "total_traffic": 1.0, **kwargs}
    with pytest.raises(TMgenException):
        datacenter_tm(**args)


def test_elephants_and_mice_defaults_span_orders_of_magnitude() -> None:
    tm = elephants_and_mice_tm(60, rng=0)
    assert tm.shape == (60, 60, 1)
    rates = tm.matrix[:, :, 0]
    assert (rates > 0).all()
    assert numpy.log10(rates.max() / rates.min()) > 5
    assert (rates < 1e6).mean() == pytest.approx(0.52, abs=0.05)


def test_elephants_and_mice_all_large_are_lognormal() -> None:
    rates = elephants_and_mice_tm(
        80, small_fraction=0, large_mu=3.0, large_sigma=0.5, rng=0
    )
    logs = numpy.log(rates.matrix)
    assert logs.mean() == pytest.approx(3.0, abs=0.05)
    assert logs.std() == pytest.approx(0.5, abs=0.05)


def test_elephants_and_mice_all_small_are_gamma() -> None:
    rates = elephants_and_mice_tm(
        80, small_fraction=1, small_shape=2.0, small_scale=3.0, rng=0
    )
    assert rates.matrix.mean() == pytest.approx(6.0, rel=0.05)
    assert rates.matrix.var() == pytest.approx(18.0, rel=0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"small_fraction": 1.5},
        {"small_fraction": -0.1},
        {"large_sigma": -1.0},
        {"small_shape": 0.0},
        {"small_scale": 0.0},
        {"large_mu": numpy.nan},
    ],
)
def test_elephants_and_mice_validates(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TMgenException):
        elephants_and_mice_tm(3, **kwargs)


def test_fluctuations_follow_the_power_law() -> None:
    means = TrafficMatrix(numpy.array([[2e7, 1e7], [5e6, 3e6]])[:, :, None])
    tm = fluctuating_tm(means, 20000, rng=0)
    for index in numpy.ndindex(2, 2):
        mean = means.matrix[index][0]
        expected = 1e6 * 10**-0.33 * (mean / 1e6) ** 0.8
        series = tm.matrix[index]
        assert series.mean() == pytest.approx(mean, rel=0.02)
        assert series.std() == pytest.approx(expected, rel=0.03)


def test_fluctuations_are_in_proportion_to_the_unit() -> None:
    base = TrafficMatrix(numpy.array([[4.0, 8.0], [2.0, 6.0]])[:, :, None])
    in_bytes = fluctuating_tm(base * 1e6, 50, unit=1e6, rng=3)
    in_megabytes = fluctuating_tm(base, 50, unit=1, rng=3)
    numpy.testing.assert_allclose(in_bytes.matrix, in_megabytes.matrix * 1e6)


def test_fluctuations_never_go_negative_and_keep_zero_flows_at_zero() -> None:
    means = TrafficMatrix(numpy.array([[0.0, 1e3], [1e5, 0.0]])[:, :, None])
    tm = fluctuating_tm(means, 200, rng=0)
    assert (tm.matrix >= 0).all()
    assert (tm.matrix[0, 0] == 0).all()


@pytest.mark.parametrize(
    "kwargs",
    [{"psi": -1.0}, {"gamma": numpy.nan}, {"unit": 0.0}, {"num_epochs": 0}],
)
def test_fluctuations_validate(kwargs: dict[str, Any]) -> None:
    args: dict[str, Any] = {"num_epochs": 3, **kwargs}
    with pytest.raises(TMgenException):
        fluctuating_tm(exact_tm(2, 1.0), **args)


def test_fluctuations_need_one_epoch_of_means() -> None:
    with pytest.raises(TMgenException, match="one epoch"):
        fluctuating_tm(exact_tm(2, 1.0, num_epochs=2), 3)


_SHOT: dict[str, Any] = {"mean_flow_size": 20.0, "flow_rate": 2.0}


def test_shot_noise_matches_the_mean_rates() -> None:
    means = TrafficMatrix(numpy.array([[3.0, 6.0], [1.5, 4.0]])[:, :, None])
    tm = shot_noise_tm(means, 4000, epoch_length=10.0, size_shape=4.0, rng=0, **_SHOT)
    assert tm.shape == (2, 2, 4000)
    numpy.testing.assert_allclose(
        tm.matrix.mean(axis=2), means.matrix[:, :, 0], rtol=0.08
    )


def test_shot_noise_longer_epochs_average_shorter_ones() -> None:
    means = TrafficMatrix(numpy.array([[3.0, 6.0], [1.5, 4.0]])[:, :, None])
    # flows last about 10 s on average, so many span several 2 s epochs
    short = shot_noise_tm(means, 200, epoch_length=2.0, rng=7, **_SHOT)
    long = shot_noise_tm(means, 100, epoch_length=4.0, rng=7, **_SHOT)
    averaged = short.matrix.reshape(2, 2, 100, 2).mean(axis=3)
    numpy.testing.assert_allclose(long.matrix, averaged, atol=1e-9)


def test_shot_noise_bigger_flows_are_burstier() -> None:
    means = TrafficMatrix(numpy.array([[20.0, 20.0], [20.0, 20.0]])[:, :, None])
    small = shot_noise_tm(
        means, 500, epoch_length=5.0, mean_flow_size=5.0, flow_rate=5.0, rng=0
    )
    large = shot_noise_tm(
        means, 500, epoch_length=5.0, mean_flow_size=200.0, flow_rate=5.0, rng=0
    )
    assert large.matrix.std() > 1.3 * small.matrix.std()


def test_shot_noise_busier_pairs_vary_less_relative_to_their_mean() -> None:
    means = TrafficMatrix(numpy.array([[1.0, 40.0], [1.0, 40.0]])[:, :, None])
    tm = shot_noise_tm(means, 1000, epoch_length=5.0, size_shape=4.0, rng=1, **_SHOT)
    cv = tm.matrix.std(axis=2) / tm.matrix.mean(axis=2)
    assert (cv[:, 1] < cv[:, 0]).all()


def test_shot_noise_without_traffic_stays_empty() -> None:
    means = TrafficMatrix(numpy.array([[0.0, 2.0], [2.0, 0.0]])[:, :, None])
    tm = shot_noise_tm(means, 50, epoch_length=5.0, rng=0, **_SHOT)
    assert (tm.matrix[0, 0] == 0).all()
    assert (tm.matrix >= 0).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epoch_length": 0.0},
        {"mean_flow_size": 0.0},
        {"flow_rate": 0.0},
        {"size_shape": 1.0},
        {"warmup": -1.0},
        {"num_epochs": 0},
        {"mean_flow_size": 1e-9},
    ],
)
def test_shot_noise_validates(kwargs: dict[str, Any]) -> None:
    args: dict[str, Any] = {"num_epochs": 5, "epoch_length": 5.0, **_SHOT, **kwargs}
    with pytest.raises(TMgenException):
        shot_noise_tm(exact_tm(2, 2.0), **args)


def test_shot_noise_needs_one_epoch_of_means() -> None:
    with pytest.raises(TMgenException, match="one epoch"):
        shot_noise_tm(exact_tm(2, 1.0, num_epochs=2), 3, epoch_length=1.0, **_SHOT)


def test_all_lists_every_public_function() -> None:
    public = {
        name
        for name, obj in vars(models).items()
        if inspect.isfunction(obj)
        and obj.__module__ == models.__name__
        and not name.startswith("_")
    }
    assert public <= set(models.__all__)
