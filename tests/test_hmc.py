# coding: utf-8
import numpy
import pytest

from tmgen.hmc import hmc_sample


@pytest.mark.parametrize('covariance', [True, False])
def test_hmc_truncated_gaussian(covariance):
    # standard 2d gaussian restricted to x >= 0 (f = I, g = 0)
    numpy.random.seed(0)
    xs = hmc_sample(numpy.eye(2), numpy.zeros(2), numpy.eye(2),
                    numpy.zeros(2), numpy.array([1.0, 1.0]), 3000,
                    covariance=covariance)
    assert xs.shape == (2, 3000)
    assert (xs > 0).all()
    # mean of a half-normal is sqrt(2/pi)
    assert xs.mean(axis=1) == pytest.approx(numpy.sqrt(2 / numpy.pi),
                                            abs=0.1)
