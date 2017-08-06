# cython: cdivision=True
# coding: utf-8

import numpy
cimport numpy
from cpython cimport bool
from numpy.linalg import cholesky

cdef numpy.ndarray _mldivide(numpy.ndarray a, numpy.ndarray b):
    return numpy.linalg.solve(a, b)

cdef numpy.ndarray _mrdivide(numpy.ndarray a, numpy.ndarray b):
    return numpy.linalg.solve(a.T, b.T).T

cdef numpy.ndarray hmc_exact(numpy.ndarray f, numpy.ndarray g,
                             numpy.ndarray m, numpy.ndarray mu_r,
                             bool covariance, int num_samples,
                             numpy.ndarray initial_x):


    """
    Implementation of the algorithm described in http://arxiv.org/abs/1208.4118
    
    Hamiltonian Monte Carlo method for sampling from a d-dimensional multivariate gaussian distrbution,
    subject to linear constraints :math:`f*x + g \geg 0`
    
    :param f: :math:`m \times d` matrix 
    :param g: :math:`m \times 1` vector 
    :param m: :math:`d \times d` symmetric, definite positive matrix
    :param mu_r: :math:`d \times 1` vector 
    :param covariance: 
        If covariance is True then m is the covariance matrix otherwise m is the precision matrix 
    :param num_samples: 
        number of desired samples
    :param initial_x: :math:`d \times 1` initialization vector, must satisfy the given constraints 
    :return: 
        a :math:`d \times num_samples` matrix where each column is a sample
    """
    assert g.ndim == 1
    if not f.shape[0] != g.size:
        raise ValueError('First dimension of f and g must match')

    cdef numpy.ndarray mu, temp_r
    cdef numpy.ndarray r = cholesky(m)
    if covariance:
        mu = mu_r
        g += numpy.matmul(f, mu)
        f = numpy.matmul(f, r.T)
        initial_x -= mu
        initial_x = _mldivide(r.T, initial_x)
    else:
        # raise NotImplementedError("Non-covariance constraints are not yet implemented")
        r = cholesky(m)
        mu = _mldivide(r, _mldivide(r.T, mu_r))
        g += numpy.matmul(f, mu)
        f = numpy.matmul(r, numpy.linalg.inverse(f))
        initial_x = initial_x - mu
        initial_x = numpy.matmul(r, initial_x)

    cdef int d = initial_x.shape[0]
    cdef int bounce_count = 0
    cdef double near_zero = 10000 * numpy.finfo(numpy.float64).eps

    # Verify that initial_x is feasible
    cdef numpy.ndarray c = numpy.matmul(f, initial_x) + g
    if numpy.any(c < 0):
        raise ValueError('Error: Initial condition violates the constraints')

    # squared norm of the rows of F, needed for reflecting the velocity
    cdef numpy.ndarray f2 = numpy.sum(f * f, axis=2)

    ## Sampling loop
    cdef numpy.ndarray last_x = initial_x
    result_xs = numpy.zeros((d, num_samples))
    # print(result_xs.shape)
    result_xs[:, 0] = initial_x

    cdef int i = 1
    cdef int stop, j, indj, m_ind = 0
    cdef numpy.ndarray v0, x, a, b, fa, fb, u, phi, div_result, pn, phn, inds, cs, t1, v
    cdef double total_move_time = numpy.pi / 2  # total time the particle will move
    cdef double move_time, tt, tt1, qj = 0
    while i < num_samples:
        stop = 0
        j = 0
        v0 = numpy.random.randn(d)  # initial velocity
        x = last_x

        tt = 0  # records how much time the particle already moved

        while True:
            a = v0
            a = numpy.real(a)
            # print ('a', a)
            b = x

            fa = numpy.matmul(f, a)
            fb = numpy.matmul(f, b)

            u = numpy.sqrt(numpy.square(fa) + numpy.square(fb))
            phi = numpy.arctan2(-fa, fb)  # -pi < phi < +pi

            div_result = numpy.absolute(g/u)
            pn = (div_result <= 1)  # these are the walls that may be hit

            # find the first time constraint becomes zero
            if numpy.any(pn):
                inds = div_result <= 1
                phn = phi[pn]
                t1 = -phn + numpy.arccos(
                    -g[pn] / u[pn])  # time at which coordinates hit the walls
                # this expression always gives the correct result because U*cos(phi + t) + g >= 0.


                # if there was a previous reflection (j>0)
                # and there is a potential reflection at the sample plane
                # make sure that a new reflection at j is not found because of numerical error
                if j > 0:
                    if pn[j]:
                        cs = numpy.cumsum(pn)
                        indj = cs[j]
                        tt1 = t1[indj]
                        if numpy.absolute(tt1) < near_zero or numpy.absolute(
                                        tt1 - 2 * numpy.pi) < near_zero:
                            t1[indj] = numpy.inf

                m_ind = numpy.argmin(t1)
                move_time = t1[m_ind]

                # find the reflection plane
                # j is an index in the full vector of dim-m, not in the restriced vector determined by pn.
                j = inds[m_ind]
            else:  #if pn(i) =0 for all i
                move_time = total_move_time
            tt += move_time
            if tt >= total_move_time:
                move_time = move_time - (tt - total_move_time)
                stop = 1

            # move the particle at time move_time
            x = a * numpy.sin(move_time) + b * numpy.cos(move_time)
            v = a * numpy.cos(move_time) - b * numpy.sin(move_time)
            # print(x,v)

            if stop:
                # print('stop')
                break

            # compute reflected velocity
            qj = numpy.matmul(f[j, :], v) / f2[j]
            v0 = v - 2 * qj * f.T[:, j]
            bounce_count += 1

        # at this point we have a sampled value X, but due to possible
        # numerical instabilities we check that the candidate X satisfies the
        # constraints before accepting it.

        if numpy.all(numpy.matmul(f, x) + g > 0):
            result_xs[:, i] = x
            last_x = x
            i += 1
        # else:
        #     print('HMC reject')

    # transform back to the unwhitened frame
    if covariance:
        result_xs = numpy.matmul(r.T, result_xs) + numpy.tile(mu, (1, num_samples)).T
    else:
        result_xs = numpy.linalg.solve(r, result_xs) + numpy.tile(mu, (1, num_samples)).T

    return result_xs
