# coding=utf-8
from __future__ import division

cimport numpy
import numpy
from six.moves import range
from tmgen.exceptions import TMgenException
from tmgen.tm cimport TrafficMatrix

cdef numpy.ndarray _peak_mean_cycle(double freq, int n, double mean,
                                    double peak_to_mean,
                                    double trough_to_mean=numpy.nan):
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
        raise TMgenException(
            'Peak-to-mean ratio must be greater than or equal to 1')
    # Generate the basic sinusoid
    cdef numpy.ndarray base = numpy.sin(
        2 * numpy.pi * freq * numpy.linspace(0, n - 1, n))
    # If mean is 0, simply return scaled sinusoid
    if mean == 0:
        return peak_to_mean * base
    cdef numpy.ndarray x, y
    cdef int i
    x = mean * (peak_to_mean - 1) * base + mean
    if not numpy.isnan(trough_to_mean):
        if trough_to_mean > peak_to_mean:
            raise TMgenException(
                'Trough-to-mean ratio must be less than or equal to '
                'peak-to-mean ratio')
        if trough_to_mean == 1:
            return x  # no effect on trough
        y = mean * (trough_to_mean - 1) * base + mean
        x[x < mean] = y[x < mean]
    return x

cpdef TrafficMatrix modulated_gravity_tm(int num_nodes, int num_tms,
                                         double mean_traffic,
                                         double pm_ratio=1.5,
                                         double t_ratio=.75,
                                         double diurnal_freq=1 / 24,
                                         double spatial_variance=100,
                                         double temporal_variance=0.01):
    """
    Generate a modulated gravity traffic matrix with the given parameters

    :param num_nodes: number of Points-of-Presence (i.e., origin-destination pairs)
    :param num_tms: total number of traffic matrices to generate (i.e., time epochs)
    :param mean_traffic: the average total volume of traffic
    :param pm_ratio: peak-to-mean ratio. Peak traffic will be larger by
        this much (must be bigger than 1). Default is 1.5
    :param t_ratio: trough-to-mean ratio. Default is 0.75
    :param diurnal_freq: Frequency of modulation. Default is 1/24 (i.e., hourly)
        if you are generating multi-day TMs
    :param spatial_variance: Variance on the volume of traffic between
        origin-destination pairs.
        Pick something reasonable with respect to your mean_traffic.
        Default is 100
    :param temporal_variance: Variance on the volume in time
    :return:
    """

    # generate total traffic
    cdef numpy.ndarray sinusoid = _peak_mean_cycle(diurnal_freq, num_tms,
                                                   mean_traffic, pm_ratio,
                                                   t_ratio)
    if numpy.min(sinusoid) < 0:
        # rescale
        sinusoid = sinusoid + numpy.absolute(numpy.min(sinusoid))
        mean_traffic = numpy.mean(sinusoid)

    base_random_gravity_tm = random_gravity_tm(num_nodes,
                                               mean_traffic / num_nodes)
    tm = numpy.concatenate(
        [base_random_gravity_tm.matrix * x for x in sinusoid], axis=2)
    return TrafficMatrix(tm)

cpdef TrafficMatrix random_gravity_tm(int num_nodes, double mean_traffic):
    """
    Random gravity model, parametrized by the mean traffic per ingress-egress
    pair.
    See http://dl.acm.org/citation.cfm?id=1096551 for full description

    :param num_nodes: number of nodes in the network
    :param mean_traffic: average traffic volume between a pair of nodes
    :return: a new :py:class:`~TrafficMatrix`
    """
    dist = numpy.random.rand(num_nodes, 1)
    dist = dist / sum(dist)
    matrix = numpy.matmul(dist, dist.T).clip(min=0) * mean_traffic
    matrix = matrix.reshape((num_nodes, num_nodes, 1))

    return TrafficMatrix(matrix)

cpdef TrafficMatrix gravity_tm(populations, double total_traffic):
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
    if not isinstance(populations, numpy.ndarray):
        populations = numpy.array(populations)
    if populations.ndim != 1:
        raise TMgenException('Expected populations to be 1-d numpy array')
    cdef int num_nodes = populations.size
    res = numpy.zeros((num_nodes, num_nodes))
    cdef double denom = populations.sum() ** 2
    cdef int i, j = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            res[i, j] = (populations[i] * populations[
                j] / denom) * total_traffic
    # Conform to the 3d shape
    res = numpy.reshape(res, (num_nodes, num_nodes, 1))
    return TrafficMatrix(res)

cpdef TrafficMatrix uniform_tm(int num_nodes, double low, double high,
                               int num_epochs=1):
    """
    Return a uniform traffic matrix. Entries are chosen independently from each other,
    uniformly at random, between given values of low and high.

    :param num_nodes: number of points-of-presence
    :param low: lowest allowed value
    :param high: highest allowed value
    :param num_epochs: number of
    :return: TrafficMatrix object
    """
    if num_nodes < 0 or low < 0 or high < 0:
        raise TMgenException("All values must be greater than 0")
    if low >= high:
        # Swap the values
        low, high = high, low
    # Get random tm
    cdef numpy.ndarray r = numpy.random.rand(num_nodes, num_nodes, num_epochs)
    return TrafficMatrix(numpy.reshape(low + (high - low) * r,
                                       (num_nodes, num_nodes, num_epochs)).clip(
        min=0))

cpdef TrafficMatrix exp_tm(int num_nodes, double mean_traffic,
                           int num_epochs=1):
    """
    Exponential traffic matrix. Values are drawn from an exponential distribution,
    with a mean value of *mean_traffic*.

    :param num_nodes:  number of nodes in the network
    :param mean_traffic: mean value of the distribution
    :param num_epochs: number of epochs in the traffic matrix
    :return: TrafficMatrix object
    """
    return \
        TrafficMatrix(numpy.random.exponential(mean_traffic,
                                               size=(num_nodes, num_nodes,
                                                     num_epochs)).clip(min=0))

cpdef TrafficMatrix spike_tm(int num_nodes, int num_spikes, double mean_spike,
                             int num_epochs=1):
    """
    Generate a traffic matrix using the spike model.

    :param num_nodes: number of nodes in the network
    :param num_spikes: number of ingress-egress spikes. Must be fewer than :math:`numpops^2`
    :param mean_spike: average volume of a single spike
    :param num_epochs: number of time epochs
    :return: TrafficMatrix object
    """
    if not num_spikes < num_nodes * num_nodes:
        raise TMgenException('Number of spikes cannot be larger than number of '
                             'ingress-egress pairs')
    # Start by generating a non-negative exponential matrix
    tm = numpy.random.exponential(mean_spike,
                                  size=(num_nodes, num_nodes, num_epochs)) \
        .clip(min=0)

    # This generates a "mask" for which values to zero out. We need all values
    # except num_spikes to be 0s. So we generete an 1-d array of num_spikes ones
    # and all other zeros.
    # We continue to shuffle it, and reshape it into a 2-d array of n x n nodes
    # to use it as a boolean index into the traffic matrix and set values to 0.
    mask = numpy.concatenate((numpy.ones(num_spikes),
                              numpy.zeros(num_nodes * num_nodes - num_spikes)))
    for e in range(num_epochs):
        numpy.random.shuffle(mask)
        reshaped_mask = mask.reshape((num_nodes, num_nodes))
        tm[:, :, e][reshaped_mask == 0] = 0

    return TrafficMatrix(tm)

cpdef TrafficMatrix lognormal_tm(int num_nodes, double mean_traffic,
                                 double sigma=1,
                                 int num_epochs=1):
    """
    Lognormal traffic matrix. Values are drawn from a lognormal distribution.

    :param num_nodes: number of nodes in the network
    :param mean_traffic: mean of the underlying normal distribution
    :param sigma: standard deviation of the underlying normal distribution
    :param num_epochs: number of epochs in the traffic matrix
    :return: TrafficMatrix object
    """
    return TrafficMatrix(numpy.random.lognormal(mean=mean_traffic, sigma=sigma,
                                                size=(num_nodes, num_nodes,
                                                      num_epochs)).clip(min=0))

cpdef TrafficMatrix exact_tm(int num_nodes, double val, int num_epochs=1):
    """
    Create a traffic matrix where each value has exact value *val*.

    Mostly used for unit tests when exact values are needed to check the solution,
    unlikely to be needed in practice.

    :param num_nodes: number of nodes in the network
    :param val: the value to use for each entry
    :param num_epochs: number of epochs
    :return: a new TrafficMatrix object
    """
    return TrafficMatrix(numpy.ones((num_nodes, num_nodes, num_epochs)) * val)

__all__ = ['modulated_gravity_tm', 'random_gravity_tm', 'gravity_tm',
           'uniform_tm', 'exp_tm', 'spike_tm', 'exact_tm', 'lognormal_tm']
