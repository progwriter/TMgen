"""
Traffic matrix generators.

A matrix has ``num_nodes`` nodes and ``num_epochs`` time epochs. An *entry* is
the traffic volume from an ingress node to an egress node in one epoch.
``total_traffic`` is the sum of the entries of a matrix and ``mean_entry`` the
mean of one entry. Every model generates traffic from a node to itself (the
diagonal); remove it with :py:meth:`~tmgen.tm.TrafficMatrix.zero_diagonal`.
"""

import numpy
from numpy.typing import ArrayLike, NDArray

from tmgen._compat import renamed_arguments
from tmgen.exceptions import TMgenException
from tmgen.tm import TrafficMatrix

RngLike = numpy.random.Generator | int | None
"""Seed or generator accepted by the random models.

``None`` draws from NumPy's global state (so ``numpy.random.seed`` keeps
working); an ``int`` seeds a fresh generator; a ``Generator`` is used as is.
"""


def _count(value: float, name: str) -> int:
    """A positive whole number, such as a number of nodes or epochs."""
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        raise TMgenException(
            f"{name} must be a positive integer, got {value!r}"
        ) from None
    if count != value or count < 1:
        raise TMgenException(f"{name} must be a positive integer, got {value!r}")
    return count


def _number(value: float, name: str, *, minimum: float | None = 0) -> float:
    """A finite number, at least *minimum* (any if None)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TMgenException(f"{name} must be a number, got {value!r}") from None
    if not numpy.isfinite(number) or (minimum is not None and number < minimum):
        bound = "" if minimum is None else f" and at least {minimum:g}"
        raise TMgenException(f"{name} must be finite{bound}, got {value!r}")
    return number


def _resolve_rng(
    rng: RngLike,
) -> numpy.random.Generator | numpy.random.RandomState:
    if rng is None:
        return numpy.random.mtrand._rand
    return numpy.random.default_rng(rng)


def _truncated_normal(
    gen: numpy.random.Generator | numpy.random.RandomState,
    mean: NDArray[numpy.float64],
    std: float,
) -> NDArray[numpy.float64]:
    """
    Sample independent normals with the given means and standard deviation,
    truncated to [0, inf). Note that *mean* and *std* are those of the
    untruncated normal.

    Uses rejection sampling: from the normal itself when most of its mass is
    positive, otherwise Robert's exponential proposal for the tail.
    """
    mean = numpy.asarray(mean, dtype=numpy.float64)
    if std == 0:
        return numpy.clip(mean, 0, None)
    lower = -mean / std  # truncation point of the standardized variable
    out = numpy.empty(mean.shape)
    todo = numpy.ones(mean.shape, dtype=bool)
    while todo.any():
        a = lower[todo]
        count = a.size
        z = numpy.empty(count)
        accept = numpy.empty(count, dtype=bool)
        body = a <= 0
        if body.any():
            z[body] = gen.standard_normal(int(body.sum()))
            accept[body] = z[body] >= a[body]
        tail = ~body
        if tail.any():
            at = a[tail]
            rate = (at + numpy.sqrt(at * at + 4)) / 2
            zt = at + gen.exponential(1.0, size=at.size) / rate
            z[tail] = zt
            accept[tail] = gen.random(at.size) <= numpy.exp(-((zt - rate) ** 2) / 2)
        idx = numpy.flatnonzero(todo)
        out[idx[accept]] = z[accept]
        todo[idx[accept]] = False
    return numpy.clip(mean + std * out, 0, None)


def diurnal_profile(
    num_epochs: int,
    mean: float,
    peak_to_mean: float = 1.5,
    trough_to_mean: float | None = None,
    diurnal_freq: float = 1 / 24,
) -> NDArray[numpy.float64]:
    """
    A daily cycle: a sinusoid that starts at *mean* and rises, with the given
    peak and trough relative to *mean*.

    By default the cycle is symmetric, so a peak-to-mean ratio of 1.5 gives a
    trough-to-mean ratio of 0.5 (values from 0.5 to 1.5 times *mean*, as in
    Fig. 2 of Tune and Roughan, SIGCOMM 2015). Pass *trough_to_mean* for a
    cycle that dips less (or more) than it peaks. The average over whole
    cycles is *mean* only for a symmetric cycle.

    Use it as the *profile* of :py:meth:`~tmgen.tm.TrafficMatrix.modulate`
    (with ``mean=1``), or to build your own total traffic.

    :param num_epochs: number of epochs (samples) to generate
    :param mean: the level the cycle oscillates around
    :param peak_to_mean: the highest value divided by *mean*; at least 1 (1 is
        a constant signal)
    :param trough_to_mean: the lowest value divided by *mean*, between 0 and 1.
        Default ``2 - peak_to_mean``, the symmetric cycle, which requires
        *peak_to_mean* of at most 2
    :param diurnal_freq: cycles per epoch. The default, 1/24, is one cycle per
        24 epochs, i.e. hourly epochs over a day
    :return: an array with one value per epoch
    """
    num_epochs = _count(num_epochs, "num_epochs")
    mean = _number(mean, "mean")
    peak_to_mean = _number(peak_to_mean, "peak_to_mean", minimum=1)
    diurnal_freq = _number(diurnal_freq, "diurnal_freq", minimum=None)
    if trough_to_mean is None:
        if peak_to_mean > 2:
            raise TMgenException(
                "A symmetric cycle needs a peak_to_mean of at most 2 (its trough "
                "would be negative); pass trough_to_mean for a different shape"
            )
        trough_to_mean = 2 - peak_to_mean
    trough_to_mean = _number(trough_to_mean, "trough_to_mean")
    if trough_to_mean > 1:
        raise TMgenException(
            f"trough_to_mean must be at most 1, got {trough_to_mean:g}"
        )
    wave = numpy.sin(2 * numpy.pi * diurnal_freq * numpy.arange(num_epochs))
    amplitude = numpy.where(wave >= 0, peak_to_mean - 1, 1 - trough_to_mean)
    return mean * (1 + amplitude * wave)


@renamed_arguments(
    num_tms="num_epochs",
    mean_traffic="total_traffic",
    pm_ratio="peak_to_mean",
    t_ratio="trough_to_mean",
)
def modulated_gravity_tm(
    num_nodes: int,
    num_epochs: int,
    total_traffic: float,
    peak_to_mean: float = 1.5,
    trough_to_mean: float | None = None,
    diurnal_freq: float = 1 / 24,
    spatial_variance: float = 0,
    temporal_variance: float = 0,
    *,
    symmetric: bool = False,
    rng: RngLike = None,
) -> TrafficMatrix:
    """
    Generate a modulated gravity traffic matrix (Tune and Roughan, SIGCOMM 2015).

    The total traffic follows a daily cycle (see :py:func:`diurnal_profile`)
    around *total_traffic*.
    Each epoch is :math:`X(t) = T\\, u\\, v^T p(t)`, where :math:`T` is the mean
    total, :math:`u` and :math:`v` hold each node's share of the ingress and
    egress traffic and :math:`p(t)` the relative volume at time :math:`t`. They
    are drawn from normal distributions truncated at zero, centered on a random
    split of the traffic and on the cycle. With non-zero variances the total of
    an epoch matches the cycle only in expectation.

    :param num_nodes: number of nodes in the network
    :param num_epochs: number of time epochs (matrices) to generate
    :param total_traffic: the level the total volume of traffic in a matrix
        oscillates around (its average over whole cycles for a symmetric cycle)
    :param peak_to_mean: highest total traffic divided by *total_traffic*; at
        least 1. Default is 1.5
    :param trough_to_mean: lowest total traffic divided by *total_traffic*,
        between 0 and 1. Default is symmetric, ``2 - peak_to_mean``
    :param diurnal_freq: cycles per epoch. The default, 1/24, is one cycle per
        24 epochs, i.e. hourly epochs over a day
    :param spatial_variance: Variance of each node's traffic, in squared
        traffic units. Each share of the mean traffic then has a standard
        deviation of ``sqrt(spatial_variance) / total_traffic``. The default,
        0, gives deterministic shares. The effect depends on the scale of
        *total_traffic* and *num_nodes*: large values relative to
        ``total_traffic / num_nodes`` inflate the total
    :param temporal_variance: Variance of the total traffic around the cycle,
        in squared traffic units. The default, 0, follows the cycle exactly
    :param symmetric: use the same shares for ingress and egress, which makes
        every matrix symmetric. By default they are drawn independently
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: a new :py:class:`~TrafficMatrix`
    """
    num_nodes = _count(num_nodes, "num_nodes")
    num_epochs = _count(num_epochs, "num_epochs")
    total_traffic = _number(total_traffic, "total_traffic")
    if total_traffic == 0:
        raise TMgenException("total_traffic must be positive")
    diurnal_freq = _number(diurnal_freq, "diurnal_freq", minimum=None)
    spatial_variance = _number(spatial_variance, "spatial_variance")
    temporal_variance = _number(temporal_variance, "temporal_variance")
    gen = _resolve_rng(rng)

    total = diurnal_profile(
        num_epochs, total_traffic, peak_to_mean, trough_to_mean, diurnal_freq
    )
    mean_total = float(numpy.mean(total))

    def node_shares() -> NDArray[numpy.float64]:
        shares = gen.random(num_nodes)
        shares /= shares.sum()
        return _truncated_normal(gen, shares, numpy.sqrt(spatial_variance) / mean_total)

    ingress = node_shares()
    egress = ingress if symmetric else node_shares()
    modulation = _truncated_normal(
        gen, total / mean_total, numpy.sqrt(temporal_variance) / mean_total
    )
    gravity = mean_total * numpy.outer(ingress, egress)
    return TrafficMatrix._from_owned(gravity[:, :, None] * modulation)


@renamed_arguments(mean_traffic="total_traffic")
def random_gravity_tm(
    num_nodes: int,
    total_traffic: float,
    *,
    normalize: bool = False,
    symmetric: bool = False,
    rng: RngLike = None,
) -> TrafficMatrix:
    """
    Random gravity model, the maximum entropy model for given total traffic
    (eq. 6 of Tune and Roughan, SIGCOMM 2015; originally from
    http://dl.acm.org/citation.cfm?id=1096551).

    The matrix is :math:`X = (T / N^2)\\, u v^T`, where :math:`T` is
    *total_traffic*, :math:`N` is *num_nodes*, and the ingress weights
    :math:`u_i` and egress weights :math:`v_j` are drawn independently from an
    exponential distribution with mean 1. Its entries sum to :math:`T` on
    average, and to a random value in any one draw; use *normalize* for an
    exact total.

    The matrix includes traffic from each node to itself (the diagonal); use
    :py:meth:`TrafficMatrix.zero_diagonal` to remove it.

    :param num_nodes: number of nodes in the network
    :param total_traffic: the total traffic volume of the matrix, in
        expectation (exactly with *normalize*). The average between a pair
        of nodes is this divided by ``num_nodes ** 2``
    :param normalize: scale the matrix so that its entries sum to exactly
        *total_traffic*
    :param symmetric: use the same weights for ingress and egress, which makes
        the matrix symmetric. Its entries then sum to
        :math:`T (1 + 1/N)` on average
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: a new :py:class:`~TrafficMatrix`
    """
    num_nodes = _count(num_nodes, "num_nodes")
    total_traffic = _number(total_traffic, "total_traffic")
    gen = _resolve_rng(rng)
    ingress = gen.exponential(1.0, size=num_nodes)
    egress = ingress if symmetric else gen.exponential(1.0, size=num_nodes)
    scale = ingress.sum() * egress.sum() if normalize else num_nodes**2
    matrix = numpy.outer(ingress, egress) * (total_traffic / scale)
    return TrafficMatrix._from_owned(matrix.reshape((num_nodes, num_nodes, 1)))


def gravity_tm(populations: ArrayLike, total_traffic: float) -> TrafficMatrix:
    """
    Compute the gravity traffic matrix, based on node populations (sizes).
    The TM will have no randomness and only contains one epoch.

    ..note::
        A possible way of generating populations is to sample from a
        log-normal distribution

    :param populations: array/list with populations (weights) for each node
    :param total_traffic: total volume of traffic in the network
        (divided among all ingress-egress pairs)
    :return: a new :py:class:`~TrafficMatrix`
    """
    total_traffic = _number(total_traffic, "total_traffic")
    try:
        populations = numpy.asarray(populations, dtype=numpy.float64)
    except (TypeError, ValueError):
        raise TMgenException("populations must be numeric") from None
    if populations.ndim != 1 or populations.size == 0:
        raise TMgenException("Expected populations to be a non-empty 1-d array")
    if not numpy.isfinite(populations).all() or (populations < 0).any():
        raise TMgenException("populations must be finite and non-negative")
    if populations.sum() == 0:
        raise TMgenException("populations must not all be zero")
    num_nodes = populations.size
    denom = populations.sum() ** 2
    res = numpy.outer(populations, populations) / denom * total_traffic
    return TrafficMatrix._from_owned(res.reshape((num_nodes, num_nodes, 1)))


def uniform_tm(
    num_nodes: int,
    low: float,
    high: float,
    num_epochs: int = 1,
    *,
    rng: RngLike = None,
) -> TrafficMatrix:
    """
    Return a uniform traffic matrix. Entries are chosen independently from each other,
    uniformly at random, between given values of low and high.

    :param num_nodes: number of nodes in the network
    :param low: lowest allowed volume of an entry
    :param high: highest allowed volume of an entry
    :param num_epochs: number of epochs
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: TrafficMatrix object
    """
    num_nodes = _count(num_nodes, "num_nodes")
    num_epochs = _count(num_epochs, "num_epochs")
    low = _number(low, "low")
    high = _number(high, "high")
    if low > high:
        raise TMgenException(f"low ({low:g}) must not exceed high ({high:g})")
    r = _resolve_rng(rng).random((num_nodes, num_nodes, num_epochs))
    return TrafficMatrix._from_owned(
        numpy.reshape(low + (high - low) * r, (num_nodes, num_nodes, num_epochs)).clip(
            min=0
        )
    )


@renamed_arguments(mean_traffic="mean_entry")
def exp_tm(
    num_nodes: int, mean_entry: float, num_epochs: int = 1, *, rng: RngLike = None
) -> TrafficMatrix:
    """
    Exponential traffic matrix. Values are drawn from an exponential distribution,
    with a mean value of *mean_entry*.

    :param num_nodes:  number of nodes in the network
    :param mean_entry: mean volume of an entry
    :param num_epochs: number of epochs in the traffic matrix
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: TrafficMatrix object
    """
    num_nodes = _count(num_nodes, "num_nodes")
    num_epochs = _count(num_epochs, "num_epochs")
    mean_entry = _number(mean_entry, "mean_entry")
    return TrafficMatrix._from_owned(
        _resolve_rng(rng)
        .exponential(mean_entry, size=(num_nodes, num_nodes, num_epochs))
        .clip(min=0)
    )


def spike_tm(
    num_nodes: int,
    num_spikes: int,
    mean_spike: float,
    num_epochs: int = 1,
    *,
    rng: RngLike = None,
) -> TrafficMatrix:
    """
    Generate a traffic matrix using the spike model.

    Exactly *num_spikes* entries of the whole ``num_nodes x num_nodes x
    num_epochs`` array are non-zero, chosen uniformly at random without
    replacement (so spikes may cluster in some epochs and be absent from
    others). Spike volumes are exponentially distributed, and all other
    entries are zero. See eq. 16 of Tune and Roughan, SIGCOMM 2015.

    :param num_nodes: number of nodes in the network
    :param num_spikes: total number of spikes over all epochs. At most
        ``num_nodes * num_nodes * num_epochs``
    :param mean_spike: average volume of a single spike
    :param num_epochs: number of time epochs
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: TrafficMatrix object
    """
    num_nodes = _count(num_nodes, "num_nodes")
    num_epochs = _count(num_epochs, "num_epochs")
    mean_spike = _number(mean_spike, "mean_spike")
    size = num_nodes * num_nodes * num_epochs
    if num_spikes != int(num_spikes) or not 0 <= num_spikes <= size:
        raise TMgenException(
            "num_spikes must be a whole number between 0 and the number of "
            f"entries (num_nodes * num_nodes * num_epochs = {size}), "
            f"got {num_spikes!r}"
        )
    num_spikes = int(num_spikes)
    gen = _resolve_rng(rng)
    tm = numpy.zeros(size)
    support = gen.choice(size, size=num_spikes, replace=False)
    tm[support] = gen.exponential(mean_spike, size=num_spikes)
    return TrafficMatrix._from_owned(tm.reshape((num_nodes, num_nodes, num_epochs)))


@renamed_arguments(mean_traffic="mu")
def lognormal_tm(
    num_nodes: int,
    mu: float,
    sigma: float = 1,
    num_epochs: int = 1,
    *,
    rng: RngLike = None,
) -> TrafficMatrix:
    """
    Lognormal traffic matrix. Values are drawn from a lognormal distribution.

    :param num_nodes: number of nodes in the network
    :param mu: mean of the underlying normal distribution, *not* the mean of
        the resulting lognormal samples (which is ``exp(mu + sigma**2 / 2)``)
    :param sigma: standard deviation of the underlying normal distribution
    :param num_epochs: number of epochs in the traffic matrix
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: TrafficMatrix object
    """
    num_nodes = _count(num_nodes, "num_nodes")
    num_epochs = _count(num_epochs, "num_epochs")
    mu = _number(mu, "mu", minimum=None)
    sigma = _number(sigma, "sigma")
    return TrafficMatrix._from_owned(
        _resolve_rng(rng)
        .lognormal(mean=mu, sigma=sigma, size=(num_nodes, num_nodes, num_epochs))
        .clip(min=0)
    )


@renamed_arguments(val="value")
def exact_tm(num_nodes: int, value: float, num_epochs: int = 1) -> TrafficMatrix:
    """
    Create a traffic matrix where each entry has exactly the volume *value*.
    Mostly used for unit tests when exact values are needed to check the solution,
    unlikely to be needed in practice.

    :param num_nodes: number of nodes in the network
    :param value: the volume of every entry
    :param num_epochs: number of epochs
    :return: a new TrafficMatrix object
    """
    num_nodes = _count(num_nodes, "num_nodes")
    num_epochs = _count(num_epochs, "num_epochs")
    value = _number(value, "value")
    return TrafficMatrix._from_owned(
        numpy.ones((num_nodes, num_nodes, num_epochs)) * value
    )


__all__ = [
    "RngLike",
    "diurnal_profile",
    "modulated_gravity_tm",
    "random_gravity_tm",
    "gravity_tm",
    "uniform_tm",
    "exp_tm",
    "spike_tm",
    "exact_tm",
    "lognormal_tm",
]
