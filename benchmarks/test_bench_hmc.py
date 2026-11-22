"""Benchmarks for the truncated-Gaussian HMC sampler (the compute-heavy part)."""

import numpy
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tmgen.hmc import hmc_sample

SAMPLES = 200


def _problem(d: int, extra_constraints: int) -> tuple[numpy.ndarray, ...]:
    """Standard Gaussian restricted to x >= 0, plus random loose half-spaces."""
    rng = numpy.random.default_rng(0)
    f = numpy.vstack([numpy.eye(d), rng.normal(size=(extra_constraints, d))])
    g = numpy.concatenate([numpy.zeros(d), numpy.full(extra_constraints, 10.0 * d)])
    return f, g, numpy.eye(d), numpy.zeros(d), numpy.ones(d)


@pytest.mark.parametrize(
    ("d", "extra"), [(10, 0), (10, 20), (50, 0), (50, 100), (200, 0), (200, 400)]
)
def test_hmc(benchmark: BenchmarkFixture, d: int, extra: int) -> None:
    f, g, m, mu, x0 = _problem(d, extra)
    numpy.random.seed(0)
    benchmark.pedantic(  # type: ignore[no-untyped-call]
        hmc_sample, args=(f, g, m, mu, x0, SAMPLES), rounds=3, iterations=1
    )
