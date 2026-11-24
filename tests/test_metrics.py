from typing import Any

import numpy
import pytest

from tmgen import TMgenException, TrafficMatrix
from tmgen.metrics import (
    autocorrelation,
    gravity_error,
    heavy_hitter_share,
    ks_distance,
    singular_spectrum,
    spatial_comparison,
    temporal_comparison,
    wasserstein_distance,
)
from tmgen.models import (
    exp_tm,
    gravity_tm,
    lognormal_tm,
    modulated_gravity_tm,
    random_gravity_tm,
)


def _tm(values: Any) -> TrafficMatrix:
    array = numpy.asarray(values, dtype=float)
    return TrafficMatrix(array.reshape(1, 1, -1))


def test_ks_is_zero_for_the_same_values_and_one_for_disjoint_ones() -> None:
    a = _tm([1, 2, 3, 4])
    assert ks_distance(a, a) == 0
    assert ks_distance(a, _tm([10, 11])) == 1


def test_ks_known_value() -> None:
    # F_a jumps at 1 and 3, F_b at 2 and 4: they differ by 1/2 between 1-2 and 3-4
    assert ks_distance(_tm([1, 3]), _tm([2, 4])) == pytest.approx(0.5)


def test_ks_is_symmetric_and_accepts_different_sizes() -> None:
    a = exp_tm(4, 1.0, num_epochs=20, rng=0)
    b = exp_tm(6, 1.0, num_epochs=7, rng=1)
    assert ks_distance(a, b) == pytest.approx(ks_distance(b, a))
    assert 0 < ks_distance(a, b) < 0.3


def test_wasserstein_of_a_shift_is_the_shift() -> None:
    a = _tm([1, 2, 3, 4])
    assert wasserstein_distance(a, _tm([4, 5, 6, 7])) == pytest.approx(3.0)
    assert wasserstein_distance(a, a) == 0


def test_wasserstein_with_different_sizes() -> None:
    # {0, 1} against {0, 0, 1, 1}: same distribution
    assert wasserstein_distance(_tm([0, 1]), _tm([0, 0, 1, 1])) == pytest.approx(0)
    assert wasserstein_distance(_tm([0, 0]), _tm([2, 2, 2])) == pytest.approx(2.0)


def test_wasserstein_sees_a_heavier_tail_that_ks_barely_does() -> None:
    base = numpy.ones(1000)
    heavy = base.copy()
    heavy[:5] = 1000.0
    assert ks_distance(_tm(base), _tm(heavy)) <= 0.0051
    assert wasserstein_distance(_tm(base), _tm(heavy)) > 4


def test_spectrum_of_a_rank_one_series_has_one_value() -> None:
    tm = modulated_gravity_tm(5, 48, 100.0, rng=0)
    spectrum = singular_spectrum(tm)
    assert spectrum[0] == pytest.approx(1.0)
    assert spectrum.sum() == pytest.approx(1.0)
    assert (numpy.diff(spectrum) <= 1e-12).all()


def test_spectrum_of_noise_is_spread_out() -> None:
    spectrum = singular_spectrum(exp_tm(4, 1.0, num_epochs=200, rng=0))
    assert spectrum[0] < 0.2


def test_spectrum_without_centering_includes_the_mean_pattern() -> None:
    tm = TrafficMatrix(numpy.full((3, 3, 10), 5.0))
    assert singular_spectrum(tm, center=False)[0] == pytest.approx(1.0)
    assert (singular_spectrum(tm) == 0).all()


def test_spectrum_needs_two_epochs() -> None:
    with pytest.raises(TMgenException):
        singular_spectrum(random_gravity_tm(3, 1.0))


def test_gravity_matrix_has_no_gravity_error() -> None:
    tm = gravity_tm([1, 2, 3, 4], 10.0)
    assert gravity_error(tm) == pytest.approx(0, abs=1e-12)
    assert gravity_error(TrafficMatrix(numpy.zeros((2, 2, 1)))) == 0


def test_independent_noise_deviates_from_gravity() -> None:
    assert gravity_error(lognormal_tm(8, 0.0, 1.5, num_epochs=1, rng=0)) > 0.3


def test_heavy_hitter_share_of_even_traffic_is_the_fraction() -> None:
    tm = TrafficMatrix(numpy.ones((10, 10, 3)))
    assert heavy_hitter_share(tm, 0.1) == pytest.approx(0.1)
    assert heavy_hitter_share(tm, 1.0) == pytest.approx(1.0)


def test_heavy_hitter_share_of_skewed_traffic_is_high() -> None:
    matrix = numpy.ones((10, 10, 1))
    matrix[0, 0] = 1000.0
    assert heavy_hitter_share(TrafficMatrix(matrix), 0.01) > 0.9


@pytest.mark.parametrize("fraction", [0, -0.1, 1.5])
def test_heavy_hitter_share_validates(fraction: float) -> None:
    with pytest.raises(TMgenException):
        heavy_hitter_share(exp_tm(3, 1.0), fraction)


def test_autocorrelation_of_a_cycle_and_of_noise() -> None:
    cycle = modulated_gravity_tm(3, 240, 10.0, diurnal_freq=1 / 24, rng=0)
    noise = exp_tm(3, 1.0, num_epochs=2000, rng=0)
    cyc = autocorrelation(cycle, 24)
    assert cyc[0] == 1
    assert cyc[1] > 0.9
    assert cyc[12] < -0.5  # half a cycle apart
    assert cyc[24] > 0.5
    assert numpy.abs(autocorrelation(noise, 10)[1:]).max() < 0.1


def test_autocorrelation_of_a_constant_series_is_zero_after_lag_zero() -> None:
    result = autocorrelation(TrafficMatrix(numpy.ones((2, 2, 10))), 3)
    numpy.testing.assert_array_equal(result, [1, 0, 0, 0])


@pytest.mark.parametrize("max_lag", [0, -1, 10, 11])
def test_autocorrelation_validates(max_lag: int) -> None:
    with pytest.raises(TMgenException):
        autocorrelation(exp_tm(2, 1.0, num_epochs=10), max_lag)


def test_a_series_is_close_to_another_draw_of_the_same_model() -> None:
    a = modulated_gravity_tm(6, 288, 100.0, diurnal_freq=1 / 144, rng=1)
    same = modulated_gravity_tm(6, 288, 100.0, diurnal_freq=1 / 144, rng=2)
    other = exp_tm(6, 100.0 / 36, num_epochs=288, rng=3)
    spatial_same = spatial_comparison(a, same)
    spatial_other = spatial_comparison(a, other)
    assert set(spatial_same) == {
        "ks",
        "wasserstein",
        "spectrum",
        "gravity",
        "heavy_hitters",
    }
    assert spatial_same["spectrum"] < 0.05 < spatial_other["spectrum"]
    assert spatial_same["gravity"] < spatial_other["gravity"]
    temporal_same = temporal_comparison(a, same)
    temporal_other = temporal_comparison(a, other)
    assert temporal_same["autocorrelation"] < 0.05 < temporal_other["autocorrelation"]


def test_comparing_a_series_with_itself_gives_zero() -> None:
    a = modulated_gravity_tm(4, 60, 10.0, rng=0)
    assert all(
        v == pytest.approx(0, abs=1e-12) for v in spatial_comparison(a, a).values()
    )
    assert all(
        v == pytest.approx(0, abs=1e-12) for v in temporal_comparison(a, a).values()
    )


def test_comparison_validates() -> None:
    a = exp_tm(3, 1.0, num_epochs=40, rng=0)
    with pytest.raises(TMgenException, match="same number of nodes"):
        spatial_comparison(a, exp_tm(4, 1.0, num_epochs=40))
    with pytest.raises(TMgenException, match="no traffic"):
        spatial_comparison(TrafficMatrix(numpy.zeros((3, 3, 40))), a)
    with pytest.raises(TMgenException, match="too short"):
        temporal_comparison(exp_tm(3, 1.0, num_epochs=3), a)
