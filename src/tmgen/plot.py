import importlib
import warnings
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from tmgen.tm import TrafficMatrix

seaborn: ModuleType | None
try:
    seaborn = importlib.import_module("seaborn")
    seaborn.set(style="white")
except ImportError:
    seaborn = None
    warnings.warn(
        "Please consider installing seaborn package for prettier graphs",
        stacklevel=2,
    )
try:
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
except ImportError as e:
    warnings.warn(
        "Plotting functionality requires the matplotlib library", stacklevel=2
    )
    raise e


def _matplotlib_heatmap(matrix: NDArray[np.float64], ax: Axes) -> None:
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.figure.colorbar(image, ax=ax)


def heatmap(
    tm: TrafficMatrix, epoch: int | None = None, *, ax: Axes | None = None
) -> Axes:
    """
    Plot the traffic matrix as a heatmap using matplotlib.
    If epoch is given or traffic matrix has a single epoch then the graph is a
    heatmap depicting flow volume between each ingress-egress pair.

    If tm has multiple epochs, the graph will have ingress-egress pairs
    enumerated sequentially on the y-axis and epochs on x-axis.

    :param tm: the traffic matrix to visualize
    :param epoch: the epoch to plot. If None, all epochs will be plotted
    :param ax: the matplotlib axes to draw on. Defaults to the current axes
    :return: the axes the heatmap was drawn on

    .. note::

        It is the user's responsibility to create new figures and display (or
        save) the plot

    Example: ::

        from matplotlib import pyplot as plt
        from tmgen import models, plot

        fig, ax = plt.subplots()
        tm = models.exp_tm(10, 500)  # 10 nodes, mean entry of 500
        plot.heatmap(tm, ax=ax)
        plt.show()  # display the graph interactively
    """
    ax = plt.gca() if ax is None else ax
    n = tm.num_nodes
    if epoch is not None or tm.num_epochs == 1:
        shown = epoch if epoch is not None else 0
        if seaborn is not None:
            seaborn.heatmap(tm.at_time(shown), cmap="viridis", ax=ax)
        else:
            _matplotlib_heatmap(tm.at_time(shown), ax)
        ax.set_xlabel("Egress node")
        ax.set_ylabel("Ingress node")
        ax.set_title(f"Volume at epoch {shown}")
    else:
        matrix = tm.matrix.reshape((n * n, tm.num_epochs))
        if seaborn is not None:
            seaborn.heatmap(
                matrix, xticklabels=max(tm.num_epochs // 10, 1), yticklabels=n, ax=ax
            )
        else:
            _matplotlib_heatmap(matrix, ax)
        ax.set_xlabel("Time (epochs)")
        ax.set_ylabel("Ingress-egress pair")
    return ax


def timeseries(tm: TrafficMatrix, *, ax: Axes | None = None) -> Axes:
    """
    Plot the traffic matrix as timeseries. With epochs on the x-axis and volume
    of the y-axis each line represents a single ingress-egress pair.

    :param tm: the traffic matrix to visualize
    :param ax: the matplotlib axes to draw on. Defaults to the current axes
    :return: the axes the lines were drawn on

    .. note::

        It is the user's responsibility to create new figures and display (or
        save) the plot

    Example: ::

        from matplotlib import pyplot as plt
        from tmgen import models, plot

        fig, ax = plt.subplots()
        tm = models.exp_tm(10, 500, num_epochs=24)
        plot.timeseries(tm, ax=ax)
        plt.show()  # display the graph interactively
    """
    ax = plt.gca() if ax is None else ax
    ax.plot(tm.matrix.reshape((tm.num_nodes**2, tm.num_epochs)).T)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Volume")
    return ax
