# coding=utf-8
from __future__ import division

cimport numpy
import numpy
from cpython cimport bool
from six.moves import range
from tmgen.hmc cimport hmc_exact
from tmgen.tm cimport TrafficMatrix

cdef numpy.ndarray _peak_mean_cycle(double freq, double n, double mean,
                                    double peak_to_mean,
                                    double trough_to_mean=numpy.nan):
    """
    Generate a simple sinusoid with specified frequency, length, mean,
    peak-to-mean ratio, and trough_to_mean ratio.

    The generated signal has the form
    :math:`x = mean*(peaktomean-1)*sin(2*\pi*freq*linspace(0,1,N))+mean`

    Note that if the mean m is 0, then
    :math:`x = peakmean*sin(2*\pi*freq*linspace(0,1,N))`

    Intervals of x below the mean have a drawdown determined by troughmean.

    ..note
        the trough-to-mean ratio must be less than or equal to the
        peak-to-mean ratio.

    :param freq: frequency
    :param n: number of samples
    :param mean: mean value
    :param peak_to_mean: ratio of peak to mean. Must be >= 1 (1 resulting
        in a constant, flatlined signal)
    :param trough_to_mean: ratio of through to mean
    :return: 
    """
    # sanity checks
    if peak_to_mean < 1:
        raise ValueError(
            'Peak-to-mean ratio must be greater than or equal to 1')
    # Generate the basic sinusoid
    cdef numpy.ndarray base = numpy.sin(2 * numpy.pi * freq * numpy.linspace(0, n - 1, n))
    # If mean is 0, simply return scaled sinusoid
    if mean == 0:
        return peak_to_mean * base
    cdef numpy.ndarray x, y
    cdef int i
    x = mean * (peak_to_mean - 1) * base + mean
    if not numpy.isnan(trough_to_mean):
        if trough_to_mean > peak_to_mean:
            raise ValueError(
                'Trough-to-mean ratio must be less than or equal to peak-to-mean ratio')
        if trough_to_mean == 1:
            return x  # no effect on trough
        y = mean * (trough_to_mean - 1) * base + mean
        x[x<mean] = y[y<mean]
    return x

cdef tuple _modulated_gravity(numpy.ndarray mean_row, numpy.ndarray mean_col,
                              numpy.ndarray modulated_total,
                              double sigmasq,
                              double sigmasq_temporal=numpy.nan,
                              bool only_gravity=False):
    if numpy.isnan(sigmasq_temporal):
        sigmasq_temporal = sigmasq

    if numpy.min(mean_row) < 0:
        raise ValueError('Row means must be non-negative')

    if numpy.min(mean_col) < 0:
        raise ValueError('Column means must be non-negative')

    cdef int num_nodes = mean_row.size
    if not num_nodes == mean_col.size:
        raise ValueError('Column means length mismatch to row means length')

    if sigmasq < 0 or sigmasq_temporal < 0:
        raise ValueError('Noise variance must be non-negative')

    if numpy.min(modulated_total) < 0:
        raise ValueError('Total traffic must be non-negative')

    # Setup necessary auxiliary parameters: number of TMs to generate and the
    # mean of the total traffic from the modulated_mean total traffic signal
    cdef int num_tms = modulated_total.size
    cdef double mean_total = numpy.mean(modulated_total)

    # average fanout
    cdef numpy.ndarray pu = mean_row / mean_total
    cdef numpy.ndarray pv = mean_col / mean_total

    # Synthesize
    # generate truncated normal random variables: we generate two samples
    # each time because hmc_exact needs to burn-in
    cdef numpy.ndarray u = hmc_exact(numpy.eye(num_nodes), numpy.zeros(num_nodes),
                                     numpy.eye(
                                         num_nodes) * sigmasq / mean_total ** 2,
                                     pu, True, 2, pu)
    cdef numpy.ndarray v = hmc_exact(numpy.eye(num_nodes), numpy.zeros(num_nodes),
                                     numpy.eye(
                                         num_nodes) * sigmasq / mean_total ** 2,
                                     pv, True, 2, pv)
    # print(u, v)
    cdef numpy.ndarray normalized_mean = modulated_total / mean_total

    # gravity model
    gravity_model = mean_total * numpy.matmul(numpy.asmatrix(u[:, 1]).T,
                                              *numpy.asmatrix(v[:, 1]))
    if only_gravity:
        return gravity_model

    # modulate mean with modulated_mean reference total
    cdef numpy.ndarray modulated_mean = \
        hmc_exact(numpy.eye(num_tms),
                  numpy.zeros(num_tms),
                  numpy.eye(num_tms) * sigmasq_temporal / mean_total ** 2,
                  normalized_mean.T,
                  True, 2, normalized_mean.T)

    # construct modulated_mean gravity model
    # in 3D array form
    cdef numpy.ndarray traffic_matrix = numpy.matmul(
        numpy.reshape(gravity_model, (num_nodes ** 2, 1)),
        numpy.asmatrix(modulated_mean[:, 1]))
    # print(traffic_matrix.shape[0], traffic_matrix.shape[1])
    traffic_matrix = numpy.reshape(numpy.asarray(traffic_matrix),
                                   (num_nodes, num_nodes, num_tms))
    return traffic_matrix, gravity_model

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
        Pick someting reasonable with respect to your mean_traffic. Default is 100
    :param temporal_variance: Variance on the volume in time
    :return:
    """

    # generate total traffic
    cdef numpy.ndarray total_traffic = _peak_mean_cycle(diurnal_freq, num_tms,
                                                        mean_traffic, pm_ratio,
                                                        t_ratio)
    if numpy.min(total_traffic) < 0:
        # rescale
        total_traffic = total_traffic + numpy.absolute(numpy.min(total_traffic))
        mean_traffic = numpy.mean(total_traffic)
    # print('total traffic', total_traffic)
    # print('mean traffic', mean_traffic)

    # randomly generate incoming and outgoing total PoP traffic
    # here we take the ratio of uniform random variables which turns out to be
    # equivalent to a Beta distribution
    # outgoing
    fraction = numpy.random.rand(num_nodes)
    fraction = fraction / (numpy.sum(fraction))
    mean_row = fraction * mean_traffic
    # print ('mean_row', mean_row)

    # incoming
    # fraction = numpy.random.rand(num_nodes)
    fraction = fraction / (numpy.sum(fraction))
    mean_col = fraction * mean_traffic

    # note: rank of G must be 1
    (traffic_matrix, g) = _modulated_gravity(mean_row, mean_col, total_traffic,
                                             spatial_variance,
                                             temporal_variance)
    return TrafficMatrix(traffic_matrix)
    # print(traffic_matrix.shape)

cpdef TrafficMatrix random_gravity_tm(int num_nodes, double mean_traffic):
    """
    Random gravity model, parametrized by the mean traffic per ingress-egress pair
    
    See http://dl.acm.org/citation.cfm?id=1096551 for full description
    
    :param num_nodes: 
    :param mean_traffic: 
    :return: 
    """


    dist = numpy.random.rand(num_nodes, 1)
    dist = dist / sum(dist)
    matrix = numpy.matmul(dist, dist.T).clip(min=0) * mean_traffic
    matrix = matrix.reshape((num_nodes, num_nodes, 1))

    return TrafficMatrix(matrix)


cpdef TrafficMatrix gravity_tm(populations, double total_traffic):
    """
    Compute the gravity traffic matrix, based on node populations. The TM will have no randomness
    and only contains one epoch.

    ..note::
        A possible way of generation populations is to sample from a log-normal distribution

    :param populations: array/list with populations (weights) for each PoP
    :param total_traffic: total volume of traffic in the network (will be divided among all ingres-egress pairs)
    :return: TraffixMatrix object
    """
    if not isinstance(populations, numpy.ndarray):
        populations = numpy.array(populations)
    assert populations.ndim == 1
    cdef int num_nodes = populations.size
    res = numpy.zeros((num_nodes, num_nodes))
    cdef double denom = populations.sum() ** 2
    cdef int i, j = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            res[i, j] = (populations[i] * populations[j] / denom) * total_traffic
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
        raise ValueError("All values must be greater than 0")
    if low >= high:
        low, high = high, low  #  swap
    # Get random tm
    cdef numpy.ndarray r = numpy.random.rand(num_nodes, num_nodes, num_epochs)
    return TrafficMatrix(numpy.reshape(low + (high - low) * r,
                                       (num_nodes, num_nodes, num_epochs)).clip(min=0))

cpdef TrafficMatrix exp_tm(int num_nodes, double mean_traffic, int num_epochs=1):
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
        raise ValueError('Number of spikes cannot be larger than number of '
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

cpdef TrafficMatrix lognormal_tm(int num_nodes, double mean_traffic, double sigma=1,
                                 int num_epochs=1):
    return TrafficMatrix(numpy.random.lognormal(mean_traffic, sigma=sigma,
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
           'uniform_tm', 'exp_tm', 'spike_tm', 'exact_tm']
