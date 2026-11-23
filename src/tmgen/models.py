import numpy
from numpy.typing import ArrayLike, NDArray

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


def _peak_mean_cycle(
    freq: float,
    n: int,
    mean: float,
    peak_to_mean: float,
    trough_to_mean: float = numpy.nan,
) -> NDArray[numpy.float64]:
    """
    Generate a simple sinusoid with specified frequency, length, mean,
    peak-to-mean ratio, and trough_to_mean ratio.

    The generated signal has the form
    :math:`x = mean*(peaktomean-1)*sin(2*\\pi*freq*linspace(0,1,N))+mean`

    Note that if the mean m is 0, then
    :math:`x = peakmean*sin(2*\\pi*freq*linspace(0,1,N))`

    Intervals of x below the mean have a drawdown determined by trough_to_mean.

    ..note
        the trough-to-mean ratio must be less than or equal to the
        peak-to-mean ratio.

    :param freq: frequency
    :param n: number of samples
    :param mean: mean value
    :param peak_to_mean: ratio of peak to mean. Must be >= 1 (1 resulting
        in a constant, flatlined signal)
    :param trough_to_mean: ratio of through to mean
    :return: numpy array containing the y-values for the sinusoid
    """
    # sanity checks
    if peak_to_mean < 1:
        raise TMgenException("Peak-to-mean ratio must be greater than or equal to 1")
    # Generate the basic sinusoid
    base = numpy.sin(2 * numpy.pi * freq * numpy.linspace(0, n - 1, n))
    # If mean is 0, simply return scaled sinusoid
    if mean == 0:
        return peak_to_mean * base
    x = mean * (peak_to_mean - 1) * base + mean
    if not numpy.isnan(trough_to_mean):
        if trough_to_mean > peak_to_mean:
            raise TMgenException(
                "Trough-to-mean ratio must be less than or equal to peak-to-mean ratio"
            )
        if trough_to_mean == 1:
            return x  # no effect on trough
        y = mean * (trough_to_mean - 1) * base + mean
        x[x < mean] = y[x < mean]
    return x


def modulated_gravity_tm(
    num_nodes: int,
    num_tms: int,
    mean_traffic: float,
    pm_ratio: float = 1.5,
    t_ratio: float = 0.75,
    diurnal_freq: float = 1 / 24,
    spatial_variance: float = 0,
    temporal_variance: float = 0,
    *,
    rng: RngLike = None,
) -> TrafficMatrix:
    """
    Generate a modulated gravity traffic matrix (Tune and Roughan, SIGCOMM 2015).

    The total traffic follows a sinusoidal cycle with mean *mean_traffic*.
    Each epoch is :math:`X(t) = T\\, u\\, u^T p(t)`, where :math:`T` is the mean
    total, :math:`u` holds each node's share of the traffic and :math:`p(t)`
    the relative volume at time :math:`t`. Both are drawn from normal
    distributions truncated at zero, centered on a random split of the
    traffic and on the cycle. With non-zero variances the total of an epoch
    matches the cycle only in expectation.

    :param num_nodes: number of Points-of-Presence (i.e., origin-destination pairs)
    :param num_tms: total number of traffic matrices to generate (i.e., time epochs)
    :param mean_traffic: the average total volume of traffic
    :param pm_ratio: peak-to-mean ratio. Peak traffic will be larger by
        this much (must be bigger than 1). Default is 1.5
    :param t_ratio: trough-to-mean ratio. Default is 0.75
    :param diurnal_freq: Frequency of modulation. Default is 1/24 (i.e., hourly)
        if you are generating multi-day TMs
    :param spatial_variance: Variance of each node's traffic, in squared
        traffic units. Each share of the mean traffic then has a standard
        deviation of ``sqrt(spatial_variance) / mean_traffic``. The default,
        0, gives deterministic shares. The effect depends on the scale of
        *mean_traffic* and *num_nodes*: large values relative to
        ``mean_traffic / num_nodes`` inflate the total
    :param temporal_variance: Variance of the total traffic around the cycle,
        in squared traffic units. The default, 0, follows the cycle exactly
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: a new :py:class:`~TrafficMatrix`
    """
    num_nodes = _count(num_nodes, "num_nodes")
    num_tms = _count(num_tms, "num_tms")
    mean_traffic = _number(mean_traffic, "mean_traffic")
    if mean_traffic == 0:
        raise TMgenException("mean_traffic must be positive")
    diurnal_freq = _number(diurnal_freq, "diurnal_freq", minimum=None)
    spatial_variance = _number(spatial_variance, "spatial_variance")
    temporal_variance = _number(temporal_variance, "temporal_variance")
    gen = _resolve_rng(rng)

    total = _peak_mean_cycle(diurnal_freq, num_tms, mean_traffic, pm_ratio, t_ratio)
    if numpy.min(total) < 0:
        total = total + numpy.absolute(numpy.min(total))
    mean_total = float(numpy.mean(total))
    if mean_total <= 0:
        raise TMgenException("The mean traffic must be positive")

    shares = gen.random(num_nodes)
    shares /= shares.sum()
    node = _truncated_normal(gen, shares, numpy.sqrt(spatial_variance) / mean_total)
    modulation = _truncated_normal(
        gen, total / mean_total, numpy.sqrt(temporal_variance) / mean_total
    )
    gravity = mean_total * numpy.outer(node, node)
    return TrafficMatrix(gravity[:, :, None] * modulation)


def random_gravity_tm(
    num_nodes: int,
    mean_traffic: float,
    *,
    normalize: bool = False,
    rng: RngLike = None,
) -> TrafficMatrix:
    """
    Random gravity model, the maximum entropy model for given total traffic
    (eq. 6 of Tune and Roughan, SIGCOMM 2015; originally from
    http://dl.acm.org/citation.cfm?id=1096551).

    The matrix is :math:`X = (T / N^2)\\, u u^T` with the node weights
    :math:`u_i` drawn independently from an exponential distribution with
    mean 1, where :math:`T` is *mean_traffic* and :math:`N` is *num_nodes*.
    The weights are shared by rows and columns, so the matrix is symmetric.
    Its entries sum to :math:`T (1 + 1/N)` on average, and to a random value
    in any one draw; use *normalize* for an exact total.

    :param num_nodes: number of nodes in the network
    :param mean_traffic: the total traffic volume of the matrix, in
        expectation (exactly with *normalize*). The average between a pair
        of nodes is this divided by ``num_nodes ** 2``
    :param normalize: scale the matrix so that its entries sum to exactly
        *mean_traffic*
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: a new :py:class:`~TrafficMatrix`
    """
    num_nodes = _count(num_nodes, "num_nodes")
    mean_traffic = _number(mean_traffic, "mean_traffic")
    weights = _resolve_rng(rng).exponential(1.0, size=num_nodes)
    scale = weights.sum() ** 2 if normalize else num_nodes**2
    matrix = numpy.outer(weights, weights) * (mean_traffic / scale)
    return TrafficMatrix(matrix.reshape((num_nodes, num_nodes, 1)))


def gravity_tm(populations: ArrayLike, total_traffic: float) -> TrafficMatrix:
    """
    Compute the gravity traffic matrix, based on node populations (sizes).
    The TM will have no randomness and only contains one epoch.

    ..note::
        A possible way of generating populations is to sample from a
        log-normal distribution

    :param populations: array/list with populations (weights) for each node
    :param total_traffic: total volume of traffic in the network
        (will be divided among all ingres-egress pairs)
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
    return TrafficMatrix(res.reshape((num_nodes, num_nodes, 1)))


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

    :param num_nodes: number of points-of-presence
    :param low: lowest allowed value
    :param high: highest allowed value
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
    return TrafficMatrix(
        numpy.reshape(low + (high - low) * r, (num_nodes, num_nodes, num_epochs)).clip(
            min=0
        )
    )


def exp_tm(
    num_nodes: int, mean_traffic: float, num_epochs: int = 1, *, rng: RngLike = None
) -> TrafficMatrix:
    """
    Exponential traffic matrix. Values are drawn from an exponential distribution,
    with a mean value of *mean_traffic*.

    :param num_nodes:  number of nodes in the network
    :param mean_traffic: mean value of the distribution
    :param num_epochs: number of epochs in the traffic matrix
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: TrafficMatrix object
    """
    num_nodes = _count(num_nodes, "num_nodes")
    num_epochs = _count(num_epochs, "num_epochs")
    mean_traffic = _number(mean_traffic, "mean_traffic")
    return TrafficMatrix(
        _resolve_rng(rng)
        .exponential(mean_traffic, size=(num_nodes, num_nodes, num_epochs))
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
    return TrafficMatrix(tm.reshape((num_nodes, num_nodes, num_epochs)))


def lognormal_tm(
    num_nodes: int,
    mean_traffic: float,
    sigma: float = 1,
    num_epochs: int = 1,
    *,
    rng: RngLike = None,
) -> TrafficMatrix:
    """
    Lognormal traffic matrix. Values are drawn from a lognormal distribution.

    :param num_nodes: number of nodes in the network
    :param mean_traffic: mean (mu) of the underlying normal distribution,
        *not* the mean of the resulting lognormal samples
    :param sigma: standard deviation of the underlying normal distribution
    :param num_epochs: number of epochs in the traffic matrix
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: TrafficMatrix object
    """
    num_nodes = _count(num_nodes, "num_nodes")
    num_epochs = _count(num_epochs, "num_epochs")
    mean_traffic = _number(mean_traffic, "mean_traffic", minimum=None)
    sigma = _number(sigma, "sigma")
    return TrafficMatrix(
        _resolve_rng(rng)
        .lognormal(
            mean=mean_traffic, sigma=sigma, size=(num_nodes, num_nodes, num_epochs)
        )
        .clip(min=0)
    )


def exact_tm(num_nodes: int, val: float, num_epochs: int = 1) -> TrafficMatrix:
    """
    Create a traffic matrix where each value has exact value *val*.
    Mostly used for unit tests when exact values are needed to check the solution,
    unlikely to be needed in practice.

    :param num_nodes: number of nodes in the network
    :param val: the value to use for each entry
    :param num_epochs: number of epochs
    :return: a new TrafficMatrix object
    """
    num_nodes = _count(num_nodes, "num_nodes")
    num_epochs = _count(num_epochs, "num_epochs")
    val = _number(val, "val")
    return TrafficMatrix(numpy.ones((num_nodes, num_nodes, num_epochs)) * val)


__all__ = [
    "RngLike",
    "modulated_gravity_tm",
    "random_gravity_tm",
    "gravity_tm",
    "uniform_tm",
    "exp_tm",
    "spike_tm",
    "exact_tm",
    "lognormal_tm",
]
