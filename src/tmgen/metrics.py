"""
Measures of how realistic a generated traffic matrix series is.

Compare a synthetic series with a real one in three separate ways, because a
generator can match one and miss another:

- *marginal*: the distribution of entry values (:py:func:`ks_distance`,
  :py:func:`wasserstein_distance`);
- *spatial*: how traffic is spread over node pairs (:py:func:`singular_spectrum`,
  :py:func:`gravity_error`, :py:func:`heavy_hitter_share`);
- *temporal*: how it changes over time (:py:func:`autocorrelation`).

:py:func:`spatial_comparison` and :py:func:`temporal_comparison` bundle these
for a real and a synthetic series.
"""

import numpy
from numpy.typing import NDArray

from tmgen.exceptions import TMgenException
from tmgen.tm import TrafficMatrix


def _entries(tm: TrafficMatrix) -> NDArray[numpy.float64]:
    values: NDArray[numpy.float64] = numpy.sort(tm.matrix, axis=None)
    return values


def ks_distance(a: TrafficMatrix, b: TrafficMatrix) -> float:
    """
    The Kolmogorov-Smirnov distance between the entry values of two series:
    the largest gap between their cumulative distribution functions, from 0
    (same distribution) to 1 (no overlap).

    The series may have different numbers of nodes and epochs.
    """
    x, y = _entries(a), _entries(b)
    grid = numpy.concatenate([x, y])
    gap = numpy.abs(
        numpy.searchsorted(x, grid, side="right") / x.size
        - numpy.searchsorted(y, grid, side="right") / y.size
    )
    return float(gap.max())


def wasserstein_distance(a: TrafficMatrix, b: TrafficMatrix) -> float:
    """
    The first Wasserstein (earth mover's) distance between the entry values of
    two series, in the units of the entries: how far, on average, a value has
    to move to turn one distribution into the other. Unlike
    :py:func:`ks_distance` it grows with the size of the mismatch in the tail.

    The series may have different numbers of nodes and epochs.
    """
    x, y = _entries(a), _entries(b)
    grid = numpy.sort(numpy.concatenate([x, y]))
    gap = numpy.abs(
        numpy.searchsorted(x, grid[:-1], side="right") / x.size
        - numpy.searchsorted(y, grid[:-1], side="right") / y.size
    )
    return float((gap * numpy.diff(grid)).sum())


def singular_spectrum(
    tm: TrafficMatrix, *, center: bool = True
) -> NDArray[numpy.float64]:
    """
    How much of the variation of a series each spatial pattern explains.

    The series is arranged as a matrix with one row per epoch and one column per
    node pair, and the result holds the squared singular values divided by
    their sum, largest first. Real traffic is low rank: a few values hold most
    of the total. A constant series gives all zeros.

    :param tm: a series with at least two epochs
    :param center: subtract the mean matrix first, so the spectrum describes
        the variation over time rather than the average pattern
    """
    if tm.num_epochs < 2:
        raise TMgenException("a spectrum needs at least 2 epochs")
    series = tm.matrix.reshape(-1, tm.num_epochs).T
    if center:
        series = series - series.mean(axis=0)
    energy = numpy.linalg.svd(series, compute_uv=False) ** 2
    total = energy.sum()
    return energy / total if total > 0 else numpy.zeros_like(energy)


def gravity_error(tm: TrafficMatrix) -> float:
    """
    How far the average matrix of a series is from a gravity model: the
    relative Frobenius error between the mean matrix and the outer product of
    its row and column sums divided by its total. Zero means traffic is
    independent between ingress and egress.
    """
    mean = tm.matrix.mean(axis=2)
    total = mean.sum()
    if total == 0:
        return 0.0
    gravity = numpy.outer(mean.sum(axis=1), mean.sum(axis=0)) / total
    return float(numpy.linalg.norm(mean - gravity) / numpy.linalg.norm(mean))


def heavy_hitter_share(tm: TrafficMatrix, fraction: float = 0.1) -> float:
    """
    The share of the total traffic carried by the busiest *fraction* of node
    pairs, judged on the mean matrix. For perfectly even traffic it equals
    *fraction*; for real networks it is much higher.

    :param fraction: the fraction of node pairs, in ``(0, 1]``
    """
    if not 0 < fraction <= 1:
        raise TMgenException(f"fraction must be in (0, 1], got {fraction!r}")
    pairs = numpy.sort(tm.matrix.mean(axis=2), axis=None)[::-1]
    total = pairs.sum()
    if total == 0:
        return 0.0
    count = max(1, round(fraction * pairs.size))
    return float(pairs[:count].sum() / total)


def autocorrelation(tm: TrafficMatrix, max_lag: int) -> NDArray[numpy.float64]:
    """
    The autocorrelation of the entries over time, averaged over node pairs
    whose traffic varies.

    :param tm: a series with more epochs than *max_lag*
    :param max_lag: the largest lag, in epochs
    :return: an array of ``max_lag + 1`` values for lags 0 to *max_lag*; the
        first is 1. All zeros except the first if no entry varies
    """
    if not isinstance(max_lag, int) or max_lag < 1:
        raise TMgenException(f"max_lag must be a positive integer, got {max_lag!r}")
    if tm.num_epochs <= max_lag:
        raise TMgenException(
            f"max_lag must be below the number of epochs ({tm.num_epochs})"
        )
    series = tm.matrix.reshape(-1, tm.num_epochs)
    series = series - series.mean(axis=1, keepdims=True)
    variance = (series**2).sum(axis=1)
    varying = variance > 0
    result = numpy.zeros(max_lag + 1)
    result[0] = 1.0
    if varying.any():
        series, variance = series[varying], variance[varying]
        for lag in range(1, max_lag + 1):
            result[lag] = numpy.mean(
                (series[:, lag:] * series[:, :-lag]).sum(axis=1) / variance
            )
    return result


def spatial_comparison(
    real: TrafficMatrix, synthetic: TrafficMatrix, *, rank: int = 5
) -> dict[str, float]:
    """
    Compare how a synthetic series spreads traffic over node pairs with a real
    one. Every value is a distance, so 0 is a perfect match:

    - ``ks``, ``wasserstein``: the marginal distances (the Wasserstein
      distance relative to the mean entry of *real*, so it has no units);
    - ``spectrum``: the largest gap between the first *rank* values of the
      two :py:func:`singular_spectrum`\\ s (on the variation over time);
    - ``gravity``: the gap between the two :py:func:`gravity_error`\\ s;
    - ``heavy_hitters``: the gap between the two :py:func:`heavy_hitter_share`\\ s.

    Both series need at least two epochs and the same number of nodes for the
    spectrum; the others work for any sizes.
    """
    if real.num_nodes != synthetic.num_nodes:
        raise TMgenException("both series must have the same number of nodes")
    real_scale = float(real.matrix.mean())
    if real_scale == 0:
        raise TMgenException("the real series has no traffic")
    top = min(rank, real.num_epochs, synthetic.num_epochs)
    return {
        "ks": ks_distance(real, synthetic),
        "wasserstein": wasserstein_distance(real, synthetic) / real_scale,
        "spectrum": float(
            numpy.abs(
                singular_spectrum(real)[:top] - singular_spectrum(synthetic)[:top]
            ).max()
        ),
        "gravity": abs(gravity_error(real) - gravity_error(synthetic)),
        "heavy_hitters": abs(heavy_hitter_share(real) - heavy_hitter_share(synthetic)),
    }


def temporal_comparison(
    real: TrafficMatrix, synthetic: TrafficMatrix, *, max_lag: int | None = None
) -> dict[str, float]:
    """
    Compare how a synthetic series changes over time with a real one. Every
    value is a distance, so 0 is a perfect match:

    - ``autocorrelation``: the mean absolute gap between the two
      :py:func:`autocorrelation` curves over lags 1 to *max_lag*;
    - ``variability``: the gap between the coefficients of variation (standard
      deviation over mean) of the total traffic per epoch.

    :param max_lag: the largest lag, in epochs. Default: a quarter of the
        shorter series, at most 288 (a day of 5-minute epochs)
    """
    shortest = min(real.num_epochs, synthetic.num_epochs)
    if max_lag is None:
        max_lag = min(288, shortest // 4)
    if max_lag < 1:
        raise TMgenException("the series are too short to compare over time")

    def variability(tm: TrafficMatrix) -> float:
        totals = tm.matrix.sum(axis=(0, 1))
        mean = totals.mean()
        return float(totals.std() / mean) if mean > 0 else 0.0

    gap = autocorrelation(real, max_lag)[1:] - autocorrelation(synthetic, max_lag)[1:]
    return {
        "autocorrelation": float(numpy.abs(gap).mean()),
        "variability": abs(variability(real) - variability(synthetic)),
    }
