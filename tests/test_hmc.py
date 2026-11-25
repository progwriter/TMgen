from typing import Any

import numpy
import pytest
from numpy.typing import NDArray

from tmgen import hmc
from tmgen.exceptions import TMgenException
from tmgen.hmc import hmc_sample


def _constrained_problem() -> tuple[NDArray[numpy.float64], ...]:
    gen = numpy.random.default_rng(0)
    d, m = 6, 14
    extra = gen.normal(size=(m - d, d))
    f = numpy.vstack([numpy.eye(d), extra])
    # keep the all-ones starting point feasible
    g = numpy.concatenate(
        [numpy.zeros(d), numpy.abs(extra.sum(axis=1)) + gen.uniform(0.5, 2.0, m - d)]
    )
    return f, g, numpy.eye(d), numpy.zeros(d), numpy.ones(d)


def test_samples_satisfy_constraints() -> None:
    f, g, cov, mu, x0 = _constrained_problem()
    xs = hmc_sample(f, g, cov, mu, x0, 500, rng=1)
    assert ((f @ xs + g[:, None]) > 0).all()


def test_matvec_fallback_matches_gram(monkeypatch: pytest.MonkeyPatch) -> None:
    f, g, cov, mu, x0 = _constrained_problem()
    with_gram = hmc_sample(f, g, cov, mu, x0, 200, rng=2)
    monkeypatch.setattr(hmc, "MAX_GRAM_BYTES", 0)
    without = hmc_sample(f, g, cov, mu, x0, 200, rng=2)
    numpy.testing.assert_allclose(with_gram, without, atol=1e-6)


@pytest.mark.parametrize("covariance", [True, False])
def test_orthant_moments(covariance: bool) -> None:
    # half-normal in every coordinate: mean sqrt(2/pi), variance 1 - 2/pi
    xs = hmc_sample(
        numpy.eye(3),
        numpy.zeros(3),
        numpy.eye(3),
        numpy.zeros(3),
        numpy.ones(3),
        20000,
        rng=11,
        covariance=covariance,
    )
    assert xs.shape == (3, 20000)
    assert (xs > 0).all()
    assert xs.mean(axis=1) == pytest.approx(numpy.sqrt(2 / numpy.pi), abs=0.03)
    assert xs.var(axis=1) == pytest.approx(1 - 2 / numpy.pi, abs=0.03)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_samples": 0},
        {"g": numpy.zeros(3)},
        {"g": numpy.zeros((2, 1))},
        {"m": numpy.eye(3)},
        {"mu_r": numpy.zeros(3)},
        {"initial_x": numpy.ones(3)},
        {"initial_x": -numpy.ones(2)},
    ],
)
def test_invalid_arguments_raise_tmgen_exception(kwargs: dict[str, Any]) -> None:
    args: dict[str, Any] = {
        "f": numpy.eye(2),
        "g": numpy.zeros(2),
        "m": numpy.eye(2),
        "mu_r": numpy.zeros(2),
        "initial_x": numpy.ones(2),
        "num_samples": 5,
    }
    args.update(kwargs)
    with pytest.raises(TMgenException):
        hmc_sample(**args)


def test_tmgen_exception_is_a_value_error() -> None:
    assert issubclass(TMgenException, ValueError)


def test_single_sample_returns_the_initial_point() -> None:
    xs = hmc_sample(
        numpy.eye(2), numpy.zeros(2), numpy.eye(2), numpy.zeros(2), numpy.ones(2), 1
    )
    numpy.testing.assert_array_equal(xs, [[1.0], [1.0]])


def test_degenerate_constraints_raise_instead_of_looping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a zero row of f with g = 0 can never satisfy the strict inequality
    monkeypatch.setattr(hmc, "MAX_CONSECUTIVE_REJECTIONS", 20)
    f = numpy.array([[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(TMgenException, match="consecutive rejections"):
        hmc_sample(f, numpy.zeros(2), numpy.eye(2), numpy.zeros(2), numpy.ones(2), 5)


def test_covariance_flag_is_keyword_only() -> None:
    args = (
        numpy.eye(2),
        numpy.zeros(2),
        numpy.eye(2),
        numpy.zeros(2),
        numpy.ones(2),
        3,
    )
    with pytest.raises(TypeError):
        hmc_sample(*args, False)  # type: ignore[call-arg]
    assert hmc_sample(*args, covariance=False).shape == (2, 3)
