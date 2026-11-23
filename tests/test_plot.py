from collections.abc import Iterator

import numpy
import pytest

matplotlib = pytest.importorskip("matplotlib")  # an optional dependency
matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from tmgen import TrafficMatrix, plot  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.mark.parametrize("epochs", [1, 6])
@pytest.mark.parametrize("have_seaborn", [True, False])
def test_heatmap(
    monkeypatch: pytest.MonkeyPatch, epochs: int, have_seaborn: bool
) -> None:
    if have_seaborn:
        pytest.importorskip("seaborn")
    else:
        monkeypatch.setattr(plot, "seaborn", None)
    plot.heatmap(TrafficMatrix(numpy.random.rand(4, 4, epochs)))


def test_heatmap_single_epoch_of_many(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plot, "seaborn", None)
    plot.heatmap(TrafficMatrix(numpy.random.rand(4, 4, 6)), epoch=2)


def test_timeseries() -> None:
    plot.timeseries(TrafficMatrix(numpy.random.rand(3, 3, 5)))


def test_functions_return_the_axes_they_draw_on() -> None:
    tm = TrafficMatrix(numpy.random.rand(3, 3, 5))
    _, (left, right) = plt.subplots(1, 2)
    assert plot.heatmap(tm, ax=left) is left
    assert plot.timeseries(tm, ax=right) is right
    assert left.get_xlabel() == "Time (epochs)"
    assert right.get_xlabel() == "Epoch"


def test_current_axes_are_the_default() -> None:
    tm = TrafficMatrix(numpy.random.rand(3, 3, 5))
    assert plot.timeseries(tm) is plt.gca()
    assert plot.heatmap(tm, epoch=1).get_title() == "Volume at epoch 1"


def test_timeseries_draws_one_line_per_pair() -> None:
    ax = plot.timeseries(TrafficMatrix(numpy.random.rand(3, 3, 5)))
    assert len(ax.lines) == 9
    assert all(numpy.asarray(line.get_ydata()).size == 5 for line in ax.lines)
