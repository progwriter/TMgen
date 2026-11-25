from typing import Any

import numpy
import pytest

from tmgen import TMgenException, TrafficMatrix
from tmgen.lowrank import LowRankModel
from tmgen.models import modulated_gravity_tm


@pytest.fixture
def series() -> TrafficMatrix:
    return modulated_gravity_tm(5, 96, 100.0, diurnal_freq=1 / 24, rng=0)


def test_a_modulated_gravity_series_is_rank_one(series: TrafficMatrix) -> None:
    model = LowRankModel.fit(series)
    assert model.rank == 1
    assert model.explained_variance == pytest.approx(1.0)
    assert model.num_nodes == 5


def test_samples_follow_the_fitted_series(series: TrafficMatrix) -> None:
    model = LowRankModel.fit(series)
    sample = model.sample(4000, rng=0)
    assert sample.shape == (5, 5, 4000)
    assert (sample.matrix >= 0).all()
    numpy.testing.assert_allclose(
        sample.matrix.mean(axis=2), series.matrix.mean(axis=2), rtol=0.2, atol=0.5
    )
    assert sample.matrix.std() == pytest.approx(series.matrix.std(), rel=0.25)


def test_samples_keep_the_slow_changes(series: TrafficMatrix) -> None:
    model = LowRankModel.fit(series)
    assert model.autocorrelation[0] > 0.9
    totals = model.sample(2000, rng=0).matrix.sum(axis=(0, 1))
    lag_one = numpy.corrcoef(totals[:-1], totals[1:])[0, 1]
    assert lag_one > 0.8


def test_rank_can_be_chosen() -> None:
    noise = numpy.random.default_rng(0).exponential(1.0, size=(4, 4, 30))
    many = TrafficMatrix(noise)
    assert LowRankModel.fit(many, rank=3).rank == 3
    assert LowRankModel.fit(many, explained_variance=1.0).rank == 16
    assert LowRankModel.fit(many, 3).explained_variance < 1.0


def test_a_single_epoch_can_be_sampled(series: TrafficMatrix) -> None:
    assert LowRankModel.fit(series).sample(1, rng=0).shape == (5, 5, 1)


def test_a_constant_node_never_goes_negative() -> None:
    rng = numpy.random.default_rng(1)
    data = numpy.zeros((3, 3, 40))
    data[0, 1] = rng.exponential(5.0, size=40)
    model = LowRankModel.fit(TrafficMatrix(data))
    assert (model.sample(200, rng=0).matrix >= 0).all()


@pytest.mark.parametrize(
    "call",
    [
        lambda: LowRankModel.fit(TrafficMatrix(numpy.ones((2, 2, 2)))),
        lambda: LowRankModel.fit(TrafficMatrix(numpy.ones((2, 2, 5)))),
        lambda: LowRankModel.fit(
            TrafficMatrix(numpy.random.default_rng(0).random((2, 2, 5))), rank=5
        ),
        lambda: LowRankModel.fit(
            TrafficMatrix(numpy.random.default_rng(0).random((2, 2, 5))), rank=0
        ),
        lambda: LowRankModel.fit(
            TrafficMatrix(numpy.random.default_rng(0).random((2, 2, 5))),
            explained_variance=0,
        ),
        lambda: LowRankModel.fit(
            TrafficMatrix(numpy.random.default_rng(0).random((2, 2, 5))),
            explained_variance=1.5,
        ),
    ],
)
def test_fit_validates(call: Any) -> None:
    with pytest.raises(TMgenException):
        call()


def test_sample_validates(series: TrafficMatrix) -> None:
    with pytest.raises(TMgenException):
        LowRankModel.fit(series).sample(0)
