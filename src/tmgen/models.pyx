from __future__ import division

cimport numpy
from cpython cimport bool
from tmgen.hmc cimport hmc_exact
from tmgen.tm cimport TrafficMatrix

cpdef numpy.ndarray _peak_mean_cycle(double freq, double n, double mean,
                                     double peak_to_mean,
                                     double trough_to_mean=numpy.nan):
    """
    Generate a simple sinusoid with specified frequency, length, mean,
    peak-to-mean ratio, and trough_to_mean ratio.

    The generated signal has the form
    $x = mean*(peaktomean-1)*sin(2*\pi*freq*linspace(0,1,N))+mean$

    Note that if the mean m is 0, then
    $x = peakmean*sin(2*\pi*freq*linspace(0,1,N))$

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
    if peak_to_mean < 1:
        raise ValueError(
            'Peak-to-mean ratio must be greater than or equal to 1')
    cdef numpy.ndarray base = numpy.sin(
        2 * numpy.pi * freq * numpy.linspace(0, n - 1, n))
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
        for i in numpy.arange(x.size):
            if x[i] < mean and y[i] < mean:
                x[i] = y[i]
    return x

cpdef tuple _modulated_gravity(numpy.ndarray mean_row, numpy.ndarray mean_col,
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

    cdef int num_pops = mean_row.size
    if not num_pops == mean_col.size:
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
    cdef numpy.ndarray u = hmc_exact(numpy.eye(num_pops), numpy.zeros(num_pops),
                                     numpy.eye(
                                         num_pops) * sigmasq / mean_total ** 2,
                                     pu, True, 2, pu)
    cdef numpy.ndarray v = hmc_exact(numpy.eye(num_pops), numpy.zeros(num_pops),
                                     numpy.eye(
                                         num_pops) * sigmasq / mean_total ** 2,
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
        numpy.reshape(gravity_model, (num_pops ** 2, 1)),
        numpy.asmatrix(modulated_mean[:, 1]))
    # print(traffic_matrix.shape[0], traffic_matrix.shape[1])
    traffic_matrix = numpy.reshape(numpy.asarray(traffic_matrix),
                                   (num_pops, num_pops, num_tms))
    return traffic_matrix, gravity_model

cpdef TrafficMatrix modulated_gravity_tm(int num_pops, int num_tms,
                                         double mean_traffic,
                                         double pm_ratio=2, double t_ratio=.5,
                                         double diurnal_freq=1 / 24,
                                         double spatial_variance=100,
                                         double temporal_variance=0.01):
    """
    Generate a modulated gravity traffic matrix with the given parameters

    :param num_pops: number of Points-of-Presence (i.e., origin-destination pairs)
    :param num_tms: total number of traffic matrices to generate (i.e., time epochs)
    :param mean_traffic: the average total volume of traffic
    :param pm_ratio: peak-to-mean ratio. Peak traffic will be larger by
        this much (must be bigger than 1)
    :param t_ratio: trough-to-mean ratio. Default is 0.5
    :param diurnal_freq: Frequency of modulation. Default is 1/24 (i.e., hourly) if you are generating multi-day TMs
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
    fraction = numpy.random.rand(num_pops)
    fraction = fraction / (numpy.sum(fraction))
    mean_row = fraction * mean_traffic
    # print ('mean_row', mean_row)

    # incoming
    # fraction = numpy.random.rand(num_pops)
    fraction = fraction / (numpy.sum(fraction))
    mean_col = fraction * mean_traffic

    # note: rank of G must be 1
    (traffic_matrix, g) = _modulated_gravity(mean_row, mean_col, total_traffic,
                                             spatial_variance,
                                             temporal_variance)
    return TrafficMatrix(traffic_matrix)
    # print(traffic_matrix.shape)

cpdef TrafficMatrix random_gravity_tm(int num_pops, double mean_traffic,
                                      double spatial_variance):
    pass

cpdef TrafficMatrix gravity_tm():
    pass

cpdef TrafficMatrix poisson_tm(int num_pops, double mean_traffic):
    pass

cpdef TrafficMatrix lognormal_tm(int num_pops, double mean_traffic):
    pass

cpdef TrafficMatrix uniform_iid(int num_pops, double low, double high):
    """
    Return a uniform traffic matrix. Entries are chosen independently.

    :param num_pops: number of points-of-presence
    :param low:
    :param high:
    :return: TrafficMatrix object
    """
    cdef numpy.ndarray r = numpy.random.rand(num_pops, num_pops)
    return TrafficMatrix(numpy.reshape(low + (high-low)*r,
                                       num_pops, num_pops, 1))
