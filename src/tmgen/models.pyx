from __future__ import division

import numpy
cimport numpy
from tmgen.hmc cimport hmc_exact

cpdef numpy.ndarray peak_mean_cycle(double freq, double n, double mean, double peak_to_mean,
                                    double trough_to_mean=numpy.nan):
    if peak_to_mean < 1:
        raise ValueError('Peak-to-mean ratio must be greater than or equal to 1')
    if mean == 0:
        return peak_to_mean * numpy.sin(2 * numpy.pi * freq * numpy.linspace(0, n - 1, n))
    cdef numpy.ndarray x, y
    cdef numpy.ndarray base = numpy.sin(2 * numpy.pi * freq * numpy.linspace(0, n - 1, n))
    x = mean * (peak_to_mean - 1) * base + mean
    # FIXME: implement trough to mean
    if not numpy.isnan(trough_to_mean):
        raise NotImplementedError('Through-to-mean ratio has not been implemented yet')
        # if troughToMeanRatio > peakToMeanRatio:
        #     raise ValueError('Trough-to-mean ratio must be less than or equal to peak-to-mean ratio')
        # if troughToMeanRatio == 1:
        #     return x  # no effect on trough
        # y = mean * (troughToMeanRatio - 1) * base + mean
        # # print('PMC:', x, y)
        # x[x < mean] = y[y < mean]
    return x

cpdef tuple modulated_gravity(numpy.ndarray mean_row, numpy.ndarray mean_col, numpy.ndarray modulated_total,
                              double sigmasq, double sigmasq_temporal=numpy.nan):
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
                                     numpy.eye(num_pops) * sigmasq / mean_total ** 2, pu, True, 2, pu)
    cdef numpy.ndarray v = hmc_exact(numpy.eye(num_pops), numpy.zeros(num_pops),
                                     numpy.eye(num_pops) * sigmasq / mean_total ** 2, pv, True, 2, pv)
    # print(u, v)
    cdef numpy.ndarray normalized_mean = modulated_total / mean_total

    # modulate mean with modulated_mean reference total
    cdef numpy.ndarray modulated_mean = hmc_exact(numpy.eye(num_tms), numpy.zeros(num_tms),
                                                  numpy.eye(num_tms) * sigmasq_temporal / mean_total ** 2,
                                                  normalized_mean.T,
                                                  True, 2, normalized_mean.T)

    # gravity model
    gravity_model = mean_total * numpy.matmul(numpy.asmatrix(u[:,1]).T, * numpy.asmatrix(v[:,1]))
    # print(gravity_model.shape[0], gravity_model.shape[1])

    # construct modulated_mean gravity model
    # in 3D array form
    cdef numpy.ndarray traffic_matrix = numpy.matmul(numpy.reshape(gravity_model, (num_pops ** 2, 1)),
                                                     numpy.asmatrix(modulated_mean[:, 1]))
    # print(traffic_matrix.shape[0], traffic_matrix.shape[1])
    traffic_matrix = numpy.reshape(numpy.asarray(traffic_matrix), (num_pops, num_pops, num_tms))
    return traffic_matrix, gravity_model

cpdef simple_generator():
    cdef int num_pops = 20  # number of PoPs
    cdef int day = 96
    cdef int num_days = 7  # set number of days
    # cdef int num_tms = num_days * day  # number of traffic matrices
    cdef int num_tms = 50
    cdef double mean_traffic = 100  # average total traffic
    cdef double pm_ratio = 2  # peak-to-mean ratio
    cdef double t_ratio = 0.25 * pm_ratio  # trough-to-mean ratio
    cdef double diurnal_freq = 1 / 96
    cdef double spatial_var = 100  # \sigma^2 parameter variation of traffic
    cdef double temporal_var = 0.01

    # generate total traffic
    cdef numpy.ndarray total_traffic = peak_mean_cycle(diurnal_freq, num_tms, mean_traffic, pm_ratio)
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
    fraction = numpy.random.rand(num_pops)
    fraction = fraction / (numpy.sum(fraction))
    mean_col = fraction * mean_traffic

    # note: rank of G must be 1
    (traffic_matrix, g) = modulated_gravity(mean_row, mean_col, total_traffic, spatial_var, temporal_var)
    print(traffic_matrix.shape)
