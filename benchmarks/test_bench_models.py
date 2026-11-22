"""Benchmarks for the traffic matrix models.

Run with `pytest benchmarks --benchmark-only` (not collected by plain `pytest`).
"""

import numpy
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tmgen import TrafficMatrix, models

SIZES = [10, 100, 1000]
EPOCHS = 100


@pytest.mark.parametrize("n", SIZES)
def test_random_gravity(benchmark: BenchmarkFixture, n: int) -> None:
    benchmark(models.random_gravity_tm, n, 1.0)


@pytest.mark.parametrize("n", SIZES)
def test_gravity(benchmark: BenchmarkFixture, n: int) -> None:
    populations = numpy.random.exponential(size=n)
    benchmark(models.gravity_tm, populations, 1000.0)


@pytest.mark.parametrize("n", [10, 100, 300])
def test_modulated_gravity(benchmark: BenchmarkFixture, n: int) -> None:
    benchmark(models.modulated_gravity_tm, n, 1000, 1000.0)


@pytest.mark.parametrize("n", [10, 100, 300])
def test_uniform(benchmark: BenchmarkFixture, n: int) -> None:
    benchmark(models.uniform_tm, n, 0.0, 1.0, EPOCHS)


@pytest.mark.parametrize("n", [10, 100, 300])
def test_exp(benchmark: BenchmarkFixture, n: int) -> None:
    benchmark(models.exp_tm, n, 1.0, EPOCHS)


@pytest.mark.parametrize("n", [10, 100, 300])
def test_lognormal(benchmark: BenchmarkFixture, n: int) -> None:
    benchmark(models.lognormal_tm, n, 1.0, 1.0, EPOCHS)


@pytest.mark.parametrize("n", [10, 100, 300])
def test_spike(benchmark: BenchmarkFixture, n: int) -> None:
    benchmark(models.spike_tm, n, n, 10.0, EPOCHS)


@pytest.mark.parametrize("n", [10, 100, 300])
def test_mean_over_epochs(benchmark: BenchmarkFixture, n: int) -> None:
    tm = TrafficMatrix(numpy.random.rand(n, n, EPOCHS))
    benchmark(tm.mean)
