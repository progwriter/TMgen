# coding: utf-8
import numpy
import pytest
from tmgen import TrafficMatrix


def test_empty_constructor():
    with pytest.raises(TypeError):
        TrafficMatrix()


def test_mean_and_worst_case():
    m = numpy.stack([numpy.ones((2, 2)), 3 * numpy.ones((2, 2))], axis=2)
    tm = TrafficMatrix(m)
    assert (tm.mean().matrix == 2).all()
    assert (tm.worst_case().matrix == 3).all()
