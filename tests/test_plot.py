import subprocess
import sys
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
def test_heatmap_without_seaborn(monkeypatch: pytest.MonkeyPatch, epochs: int) -> None:
    monkeypatch.setattr(plot, "seaborn", None)
    plot.heatmap(TrafficMatrix(numpy.random.rand(4, 4, epochs)))


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


def test_heatmap_of_many_epochs_labels_every_tenth() -> None:
    # seaborn needs an integer step for the tick labels
    pytest.importorskip("seaborn")
    ax = plot.heatmap(TrafficMatrix(numpy.random.rand(3, 3, 24)))
    assert len(ax.get_xticks()) == 12


def _run(code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_importing_plot_leaves_the_matplotlib_style_alone() -> None:
    code = (
        "import matplotlib; before = dict(matplotlib.rcParams); "
        "import tmgen.plot; print(before == dict(matplotlib.rcParams))"
    )
    assert _run(code) == "True"


def test_plot_works_without_seaborn_and_without_warnings() -> None:
    # a None entry in sys.modules makes "import seaborn" fail
    code = (
        "import sys; sys.modules['seaborn'] = None; "
        "import matplotlib; matplotlib.use('Agg'); "
        "import numpy; import tmgen.plot as p; from tmgen import TrafficMatrix; "
        "p.heatmap(TrafficMatrix(numpy.ones((2, 2, 12)))); print(p.seaborn)"
    )
    assert _run(code) == "None"
