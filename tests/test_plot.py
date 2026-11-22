from collections.abc import Iterator

import matplotlib
import numpy
import pytest

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
