"""
Traffic matrix generators.

A matrix has ``num_nodes`` nodes and ``num_epochs`` time epochs. An *entry* is
the traffic volume from an ingress node to an egress node in one epoch.
``total_traffic`` is the sum of the entries of a matrix and ``mean_entry`` the
mean of one entry. Every model generates traffic from a node to itself (the
diagonal); remove it with :py:meth:`~tmgen.tm.TrafficMatrix.zero_diagonal`.
"""

from collections.abc import Sequence
from typing import Any

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


def _resolve_rng(rng: RngLike) -> Any:
    """
    A source of random numbers with the methods of ``numpy.random.Generator``
    that the models use. ``None`` gives the ``numpy.random`` module itself, whose
    functions draw from NumPy's global state (and so follow ``numpy.random.seed``).
    """
    if rng is None:
        return numpy.random
    return numpy.random.default_rng(rng)


def _truncated_normal(
    gen: Any,
    mean: NDArray[numpy.float64],
    std: float | NDArray[numpy.float64],
) -> NDArray[numpy.float64]:
    """
    Sample independent normals with the given means and standard deviation,
    truncated to [0, inf). Note that *mean* and *std* are those of the
    untruncated normal. *std* is a number, or an array of positive numbers
    that broadcasts against *mean*.

    Uses rejection sampling: from the normal itself when most of its mass is
    positive, otherwise Robert's exponential proposal for the tail.
    """
    mean = numpy.asarray(mean, dtype=numpy.float64)
    std = numpy.asarray(std, dtype=numpy.float64)
    if not std.any():
        return numpy.clip(mean, 0, None)
    shape = mean.shape
    std = numpy.broadcast_to(std, shape).ravel()
    mean = mean.ravel()
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
    return numpy.clip(mean + std * out, 0, None).reshape(shape)


def _series(values: ArrayLike, name: str) -> NDArray[numpy.float64]:
    """A non-empty one-dimensional array of finite non-negative numbers."""
    array = numpy.asarray(values, dtype=numpy.float64)
    if array.ndim != 1 or array.size == 0:
        raise TMgenException(f"{name} must be a non-empty one-dimensional array")
    if not numpy.isfinite(array).all() or (array < 0).any():
        raise TMgenException(f"{name} must be finite and non-negative")
    return array


def _node_traffic(
    values: ArrayLike, num_epochs: int, name: str
) -> NDArray[numpy.float64]:
    """Per-node traffic as (num_nodes, num_epochs); a 1-D input is constant in time."""
    array = numpy.asarray(values, dtype=numpy.float64)
    if array.ndim == 1:
        array = numpy.repeat(array[:, None], num_epochs, axis=1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != num_epochs:
        raise TMgenException(
            f"{name} must have shape (num_nodes, {num_epochs}) or (num_nodes,), "
            f"got {numpy.shape(values)}"
        )
    if not numpy.isfinite(array).all() or (array < 0).any():
        raise TMgenException(f"{name} must be finite and non-negative")
    return array


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
    profile: NDArray[numpy.float64] = mean * (1 + amplitude * wave)
    return profile


def fourier_profile(
    num_epochs: int,
    mean: float,
    harmonics: Sequence[tuple[float, float, float]],
) -> NDArray[numpy.float64]:
    """
    A cyclical profile built from a few Fourier components, the temporal
    constraint of eq. 8 of Tune and Roughan (SIGCOMM 2015).

    The value at epoch :math:`t` is :math:`m + \\sum_h a_h \\cos(2\\pi f_h t +
    \\phi_h)`. Choose the components to match the cycles of real traffic, for
    example the strongest ones of a measured total (daily, weekly, ...).

    :param num_epochs: number of epochs (samples) to generate
    :param mean: the level the profile oscillates around, :math:`m`
    :param harmonics: ``(frequency, amplitude, phase)`` triples: the frequency
        in cycles per epoch, the amplitude in the units of *mean* and the
        phase in radians
    :return: an array with one value per epoch
    :raises TMgenException: if the profile would be negative in some epoch
    """
    num_epochs = _count(num_epochs, "num_epochs")
    mean = _number(mean, "mean")
    epochs = numpy.arange(num_epochs)
    profile: NDArray[numpy.float64] = numpy.full(num_epochs, mean)
    for frequency, amplitude, phase in harmonics:
        frequency = _number(frequency, "frequency", minimum=None)
        amplitude = _number(amplitude, "amplitude", minimum=None)
        phase = _number(phase, "phase", minimum=None)
        profile = profile + amplitude * numpy.cos(
            2 * numpy.pi * frequency * epochs + phase
        )
    if profile.min() < 0:
        raise TMgenException(
            f"The profile goes negative (minimum {profile.min():g}); "
            "lower the amplitudes or raise mean"
        )
    return profile


def time_zone_profiles(
    profile: ArrayLike, shifts: Sequence[int]
) -> NDArray[numpy.float64]:
    """
    One copy of a cyclical *profile* per node, each delayed by a number of
    epochs. It models nodes in different time zones whose traffic follows the
    same daily cycle at different local times (Tune and Roughan, SIGCOMM 2015,
    section 3.4).

    Use the result as the *ingress_traffic* or *egress_traffic* of
    :py:func:`ncim_tm`. The profile wraps around, so it should cover whole
    cycles.

    :param profile: one value per epoch, such as from :py:func:`diurnal_profile`
    :param shifts: for each node, how many epochs its cycle lags behind
        *profile* (negative to lead)
    :return: an array of shape ``(len(shifts), len(profile))``; row ``i`` is
        *profile* delayed by ``shifts[i]``
    """
    values = numpy.asarray(profile, dtype=numpy.float64)
    if values.ndim != 1 or values.size == 0:
        raise TMgenException("profile must be a non-empty one-dimensional array")
    if len(shifts) == 0:
        raise TMgenException("shifts must not be empty")
    for shift in shifts:
        if int(shift) != shift:
            raise TMgenException(f"shifts must be whole epochs, got {shift!r}")
    return numpy.stack([numpy.roll(values, int(shift)) for shift in shifts])


def ncim_tm(
    total_traffic: ArrayLike,
    num_nodes: int | None = None,
    *,
    ingress_traffic: ArrayLike | None = None,
    egress_traffic: ArrayLike | None = None,
    spatial_variance: float = 0,
    rng: RngLike = None,
) -> TrafficMatrix:
    """
    Non-stationary conditionally independent model (NCIM, eqs. 14 and 15 of
    Tune and Roughan, SIGCOMM 2015): every epoch is drawn independently, with
    the traffic of each epoch following what you know about it.

    Knowing only the mean total of each epoch, :math:`s(t)`, every entry is
    exponential with mean :math:`s(t) / N^2`. Knowing in addition the mean
    traffic entering and leaving each node, :math:`r_i(t)` and :math:`c_j(t)`,
    the epoch is :math:`X(t) = s(t)\\, u(t) v(t)^T` with
    :math:`u_i \\sim \\mathrm{Exp}(r_i / s)` and
    :math:`v_j \\sim \\mathrm{Exp}(c_j / s)` (means), a gravity model with
    random node weights whose mean is :math:`r_i c_j / s`. Shifting the
    per-node cycles models nodes in different time zones, see
    :py:func:`time_zone_profiles`.

    The matrices include traffic from each node to itself (the diagonal).

    :param total_traffic: the mean total traffic of each epoch, one
        non-negative value per epoch; this sets the number of epochs
    :param num_nodes: number of nodes. Needed only without *ingress_traffic*
        and *egress_traffic*, otherwise it is read from them
    :param ingress_traffic: the mean traffic entering at each node, shape
        ``(num_nodes, num_epochs)``, or ``(num_nodes,)`` for a constant. Give
        it together with *egress_traffic*
    :param egress_traffic: the mean traffic leaving at each node, as above
    :param spatial_variance: with node traffics given, the variance of each
        node's traffic in squared traffic units: the weights then follow normal
        distributions truncated at zero, as in :py:func:`modulated_gravity_tm`.
        The default, 0, gives exponential weights
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: a new :py:class:`~TrafficMatrix`
    """
    total = _series(total_traffic, "total_traffic")
    num_epochs = total.size
    spatial_variance = _number(spatial_variance, "spatial_variance")
    gen = _resolve_rng(rng)

    if ingress_traffic is None and egress_traffic is None:
        if num_nodes is None:
            raise TMgenException(
                "Give num_nodes, or ingress_traffic and egress_traffic"
            )
        if spatial_variance:
            raise TMgenException(
                "spatial_variance needs ingress_traffic and egress_traffic"
            )
        num_nodes = _count(num_nodes, "num_nodes")
        scale = total / num_nodes**2
        entries = gen.exponential(1.0, size=(num_nodes, num_nodes, num_epochs))
        return TrafficMatrix._from_owned(entries * scale)

    if ingress_traffic is None or egress_traffic is None:
        raise TMgenException("Give both ingress_traffic and egress_traffic, or neither")
    ingress = _node_traffic(ingress_traffic, num_epochs, "ingress_traffic")
    egress = _node_traffic(egress_traffic, num_epochs, "egress_traffic")
    if num_nodes is not None:
        num_nodes = _count(num_nodes, "num_nodes")
        if num_nodes != ingress.shape[0]:
            raise TMgenException(
                f"num_nodes is {num_nodes} but ingress_traffic has "
                f"{ingress.shape[0]} nodes"
            )
    if egress.shape[0] != ingress.shape[0]:
        raise TMgenException(
            "ingress_traffic and egress_traffic must cover the same nodes"
        )

    scale = numpy.where(total > 0, total, 1.0)

    def weights(node_traffic: NDArray[numpy.float64]) -> NDArray[numpy.float64]:
        share = node_traffic / scale
        if spatial_variance:
            return _truncated_normal(gen, share, numpy.sqrt(spatial_variance) / scale)
        return numpy.asarray(gen.exponential(1.0, size=share.shape) * share)

    u = weights(ingress)
    v = weights(egress)
    return TrafficMatrix._from_owned(numpy.einsum("it,jt->ijt", u, v) * total)


def traffic_class_tm(
    total_traffic: ArrayLike,
    node_classes: ArrayLike,
    class_fractions: ArrayLike,
    *,
    rng: RngLike = None,
) -> TrafficMatrix:
    """
    Traffic classes (eq. 17 of Tune and Roughan, SIGCOMM 2015): nodes belong to
    classes, such as access and peering nodes, and you say what fraction of the
    traffic goes from each class to each class. Within a pair of classes
    traffic follows a random gravity model, so ingress and egress nodes are
    independent only inside their class.

    Epoch :math:`t` is :math:`X(t) = T(t) \\sum_{m,n} q_{mn}(t)\\, u_m(t) v_n(t)^T`,
    where :math:`T(t)` is *total_traffic*, :math:`q_{mn}` the fraction of
    traffic from class :math:`m` to class :math:`n`, and :math:`u_m`
    (:math:`v_n`) holds exponential weights, scaled to sum to 1, for the nodes
    of class :math:`m` as ingress (class :math:`n` as egress) and zero for the
    others. The traffic from class :math:`m` to class :math:`n` in an epoch is
    therefore exactly :math:`T(t) q_{mn}(t)`.

    The matrices include traffic from each node to itself (the diagonal).

    :param total_traffic: the total traffic of each epoch, one non-negative
        value per epoch; this sets the number of epochs
    :param node_classes: the class of each node, a whole number from 0 to
        ``num_classes - 1``; every class needs at least one node
    :param class_fractions: the fractions :math:`q`: shape
        ``(num_classes, num_classes)`` where entry ``[m, n]`` is the share of
        traffic from class ``m`` to class ``n``, or
        ``(num_classes, num_classes, num_epochs)`` to vary it over time. Non-negative
        and summing to 1 in every epoch
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: a new :py:class:`~TrafficMatrix`
    """
    total = _series(total_traffic, "total_traffic")
    num_epochs = total.size
    labels = numpy.asarray(node_classes)
    if (
        labels.ndim != 1
        or labels.size == 0
        or not numpy.issubdtype(labels.dtype, numpy.integer)
    ):
        raise TMgenException("node_classes must be a non-empty list of whole numbers")
    num_classes = int(labels.max()) + 1
    if labels.min() < 0 or numpy.bincount(labels, minlength=num_classes).min() == 0:
        raise TMgenException(
            "node_classes must number the classes from 0 and use every class"
        )
    fractions = numpy.asarray(class_fractions, dtype=numpy.float64)
    if fractions.shape == (num_classes, num_classes):
        fractions = numpy.repeat(fractions[:, :, None], num_epochs, axis=2)
    if fractions.shape != (num_classes, num_classes, num_epochs):
        raise TMgenException(
            f"class_fractions must have shape ({num_classes}, {num_classes}) or "
            f"({num_classes}, {num_classes}, {num_epochs}), got {fractions.shape}"
        )
    if (
        not numpy.isfinite(fractions).all()
        or (fractions < 0).any()
        or not numpy.allclose(fractions.sum(axis=(0, 1)), 1.0)
    ):
        raise TMgenException(
            "class_fractions must be non-negative and sum to 1 in every epoch"
        )
    gen = _resolve_rng(rng)

    def weights() -> NDArray[numpy.float64]:
        raw = gen.exponential(1.0, size=(labels.size, num_epochs))
        for c in range(num_classes):
            members = labels == c
            raw[members] /= raw[members].sum(axis=0)
        return numpy.asarray(raw)

    pair_fraction = fractions[labels][:, labels]
    matrix = numpy.einsum("it,jt->ijt", weights(), weights()) * pair_fraction * total
    return TrafficMatrix._from_owned(matrix)


def datacenter_tm(
    num_nodes: int,
    total_traffic: float,
    *,
    hot_node_fraction: float = 0.2,
    hot_load_fraction: float = 0.55,
    hot_nodes: Sequence[int] | None = None,
    num_racks: int = 1,
    inter_rack_fraction: float | None = None,
    rng: RngLike = None,
) -> TrafficMatrix:
    """
    A data centre traffic matrix with hot nodes and racks, following the
    node distributions of TrafPy (Parsonson, Benjamin and Zervas, *Traffic
    generation for benchmarking data centre networks*, Optical Switching and
    Networking, 2022).

    A fraction of the nodes is *hot*: together they send (and receive)
    *hot_load_fraction* of the traffic, split evenly among them, and the other
    nodes split the rest. Nodes can also sit in racks, and *inter_rack_fraction*
    sets the share of the traffic that crosses racks. Of all matrices with
    these properties the result is the one of maximum entropy (found by
    iterative proportional fitting), so no pair is favored beyond what you
    asked for.

    There is no traffic from a node to itself (as in TrafPy), and the matrix
    is deterministic given the hot nodes; *rng* only chooses them. It has one
    epoch: use :py:meth:`~tmgen.tm.TrafficMatrix.modulate` for a time series.

    :param num_nodes: number of nodes (servers), at least 2
    :param total_traffic: the total traffic of the matrix
    :param hot_node_fraction: fraction of the nodes that are hot, between 0
        and 1; at least one node is hot. The default 0.2 with *hot_load_fraction*
        0.55 is TrafPy's university data centre. Ignored with *hot_nodes*
    :param hot_load_fraction: share of the ingress and of the egress traffic
        of the hot nodes together, between 0 and 1. Each node must send less
        than half of the traffic, as it cannot send to itself
    :param hot_nodes: the indices of the hot nodes instead of random ones
    :param num_racks: number of racks. Nodes are assigned to racks in order,
        as evenly as possible
    :param inter_rack_fraction: share of the traffic between nodes of
        different racks, between 0 and 1. Required with more than one rack
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: a new :py:class:`~TrafficMatrix`
    :raises TMgenException: if the requested properties cannot hold together
    """
    num_nodes = _count(num_nodes, "num_nodes")
    if num_nodes < 2:
        raise TMgenException("num_nodes must be at least 2")
    total_traffic = _number(total_traffic, "total_traffic")
    hot_load_fraction = _number(hot_load_fraction, "hot_load_fraction")
    if hot_load_fraction > 1:
        raise TMgenException("hot_load_fraction must be at most 1")
    num_racks = _count(num_racks, "num_racks")
    if num_racks > num_nodes:
        raise TMgenException("num_racks must not exceed num_nodes")
    if (inter_rack_fraction is None) != (num_racks == 1):
        raise TMgenException(
            "inter_rack_fraction is required with, and only with, more than one rack"
        )
    gen = _resolve_rng(rng)

    if hot_nodes is None:
        hot_node_fraction = _number(hot_node_fraction, "hot_node_fraction")
        if hot_node_fraction > 1:
            raise TMgenException("hot_node_fraction must be at most 1")
        count = max(1, round(hot_node_fraction * num_nodes))
        hot = numpy.asarray(gen.permutation(num_nodes)[:count])
    else:
        hot = numpy.unique(numpy.asarray(hot_nodes, dtype=int))
        if hot.size == 0 or hot.min() < 0 or hot.max() >= num_nodes:
            raise TMgenException(
                f"hot_nodes must be non-empty node indices below {num_nodes}"
            )
    if hot.size == num_nodes and hot_load_fraction != 1:
        raise TMgenException("hot_load_fraction must be 1 when every node is hot")
    share = numpy.full(num_nodes, 0.0)
    cold = num_nodes - hot.size
    share[hot] = hot_load_fraction / hot.size
    if cold:
        share[numpy.setdiff1d(numpy.arange(num_nodes), hot)] = (
            1 - hot_load_fraction
        ) / cold
    if (2 * share > 1 + 1e-12).any():
        raise TMgenException(
            "A node cannot send more than half of the traffic; lower "
            "hot_load_fraction or hot more nodes"
        )

    off_diagonal = ~numpy.eye(num_nodes, dtype=bool)
    mass = off_diagonal.astype(numpy.float64)
    racks = numpy.arange(num_nodes) * num_racks // num_nodes
    crossing = racks[:, None] != racks[None, :]
    within = off_diagonal & ~crossing
    inter = 0.0
    if inter_rack_fraction is not None:
        inter = _number(inter_rack_fraction, "inter_rack_fraction")
        if inter > 1:
            raise TMgenException("inter_rack_fraction must be at most 1")
        if not within.any() and inter < 1:
            raise TMgenException("racks of one node have no traffic inside them")
    for _ in range(20000):
        mass *= (share / mass.sum(axis=1))[:, None]
        mass *= (share / mass.sum(axis=0))[None, :]
        if num_racks > 1:
            crossing_mass = mass[crossing].sum()
            mass[crossing] *= inter / crossing_mass if crossing_mass else 1.0
            mass[within] *= (1 - inter) / mass[within].sum() if inter < 1 else 0.0
        error = max(
            numpy.abs(mass.sum(axis=1) - share).max(),
            numpy.abs(mass.sum(axis=0) - share).max(),
            abs(mass[crossing].sum() - inter) if num_racks > 1 else 0.0,
        )
        if error < 1e-12:
            break
    else:
        raise TMgenException(
            "The requested hot nodes, racks and inter-rack fraction cannot hold "
            "together; relax one of them"
        )
    return TrafficMatrix._from_owned((mass * total_traffic)[:, :, None])


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
    profile: ArrayLike | None = None,
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
    :param profile: the mean total traffic of each epoch, one value per epoch
        (for example from :py:func:`fourier_profile`). It replaces the daily
        cycle, so *total_traffic*, *peak_to_mean*, *trough_to_mean* and
        *diurnal_freq* are ignored
    :param rng: seed or :py:class:`numpy.random.Generator`; see :py:data:`RngLike`
    :return: a new :py:class:`~TrafficMatrix`
    """
    num_nodes = _count(num_nodes, "num_nodes")
    num_epochs = _count(num_epochs, "num_epochs")
    total_traffic = _number(total_traffic, "total_traffic")
    if total_traffic == 0 and profile is None:
        raise TMgenException("total_traffic must be positive")
    diurnal_freq = _number(diurnal_freq, "diurnal_freq", minimum=None)
    spatial_variance = _number(spatial_variance, "spatial_variance")
    temporal_variance = _number(temporal_variance, "temporal_variance")
    gen = _resolve_rng(rng)

    if profile is None:
        total = diurnal_profile(
            num_epochs, total_traffic, peak_to_mean, trough_to_mean, diurnal_freq
        )
    else:
        total = numpy.asarray(profile, dtype=numpy.float64)
        if total.shape != (num_epochs,):
            raise TMgenException(
                f"profile needs one value per epoch ({num_epochs}), got shape "
                f"{total.shape}"
            )
        if not numpy.isfinite(total).all() or (total < 0).any():
            raise TMgenException("profile must be finite and non-negative")
    mean_total = float(numpy.mean(total))
    if mean_total == 0:
        raise TMgenException("profile must have a positive mean")

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
