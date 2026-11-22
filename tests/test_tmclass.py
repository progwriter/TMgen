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


@pytest.mark.parametrize('epochs', [1, 3])
def test_csv_shape(tmp_path, epochs):
    tm = TrafficMatrix(numpy.random.rand(2, 2, epochs))
    f = tmp_path / 'tm.csv'
    tm.to_csv(str(f))
    loaded = numpy.loadtxt(str(f), delimiter=',', ndmin=2)
    assert loaded.shape == ((2, 2) if epochs == 1 else (epochs, 4))
