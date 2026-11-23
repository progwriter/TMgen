# cython: cdivision=True, boundscheck=False, wraparound=False, language_level=3
# coding: utf-8

import numpy
cimport numpy
from cpython cimport bool
from libc.math cimport acos, atan2, cos, fabs, sin, sqrt, INFINITY, M_PI, isfinite
from numpy.linalg import cholesky

from tmgen.exceptions import TMgenException

# The Gram matrix G = F F^T costs m*m doubles. Above this size the kernel
# falls back to recomputing F @ a and F @ b after every reflection.
MAX_GRAM_BYTES = 268435456

# A candidate is rejected when rounding error leaves it outside the
# constraints. This many rejections in a row means the problem is degenerate
# (for example a constraint that can never be strictly satisfied).
MAX_CONSECUTIVE_REJECTIONS = 1000

cdef double NEAR_ZERO = 10000 * 2.220446049250313e-16


cdef void _matvec(double[:, ::1] f, double[::1] x, double[::1] out) noexcept nogil:
    cdef Py_ssize_t k, j
    cdef double s
    for k in range(f.shape[0]):
        s = 0
        for j in range(f.shape[1]):
            s += f[k, j] * x[j]
        out[k] = s


cdef numpy.ndarray _sample_loop(double[:, ::1] f, double[::1] g,
                                double[::1] x0, int num_samples, object rng,
                                Py_ssize_t max_gram_bytes,
                                Py_ssize_t max_rejections):
    """
    Exact HMC on the whitened problem: sample x ~ N(0, I) with f x + g >= 0.

    Between wall hits the particle moves along x(t) = a sin t + b cos t, so
    f@a and f@b are all that is needed to find the next hit. After a
    reflection a and b change by a rank-one term, and f@a, f@b can be updated
    in O(m) from the Gram matrix instead of recomputed in O(m d).
    """
    cdef Py_ssize_t m = f.shape[0], d = f.shape[1]
    cdef Py_ssize_t i = 1, k, j, jbest, rejections = 0
    cdef bint use_gram = m * m * 8 <= max_gram_bytes
    cdef double[:, ::1] gram
    cdef double[::1] f2 = numpy.empty(m)
    if use_gram:
        gram = numpy.ascontiguousarray(numpy.asarray(f) @ numpy.asarray(f).T)
        for k in range(m):
            f2[k] = gram[k, k]
    else:
        gram = numpy.empty((0, 0))
        for k in range(m):
            f2[k] = numpy.dot(f[k], f[k])

    result = numpy.zeros((d, num_samples))
    cdef double[:, ::1] out = result
    cdef double[::1] a = numpy.zeros(d), b = numpy.zeros(d)
    cdef double[::1] x = numpy.zeros(d), v = numpy.zeros(d)
    cdef double[::1] fa = numpy.zeros(m), fb = numpy.zeros(m)
    cdef double[::1] fx = numpy.zeros(m), fv = numpy.zeros(m)
    cdef double[::1] last = numpy.array(x0), vel
    cdef double u, t, best, move, tt, s, c, qj, ak, bk
    cdef double total = M_PI / 2
    cdef bint any_wall, stop, ok

    for k in range(d):
        out[k, 0] = x0[k]
    while i < num_samples:
        vel = rng.standard_normal(d)
        for k in range(d):
            a[k] = vel[k]
            b[k] = last[k]
        _matvec(f, a, fa)
        _matvec(f, b, fb)
        j = -1
        tt = 0
        while True:
            best = INFINITY
            jbest = -1
            any_wall = False
            for k in range(m):
                ak = fa[k]
                bk = fb[k]
                u = sqrt(ak * ak + bk * bk)
                if fabs(g[k] / u) <= 1.0:
                    any_wall = True
                    t = -atan2(-ak, bk) + acos(-g[k] / u)
                    # the wall just hit would otherwise be found again
                    # because of rounding
                    if k == j and (fabs(t) < NEAR_ZERO
                                   or fabs(t - 2 * M_PI) < NEAR_ZERO):
                        t = INFINITY
                    if t < best:
                        best = t
                        jbest = k
            if any_wall and isfinite(best):
                move = best
                j = jbest
            else:
                move = total
            tt += move
            stop = tt >= total
            if stop:
                move -= tt - total
            s = sin(move)
            c = cos(move)
            for k in range(d):
                x[k] = a[k] * s + b[k] * c
                v[k] = a[k] * c - b[k] * s
            if stop:
                break
            for k in range(m):
                fx[k] = fa[k] * s + fb[k] * c
                fv[k] = fa[k] * c - fb[k] * s
            qj = fv[j] / f2[j]
            for k in range(d):
                a[k] = v[k] - 2 * qj * f[j, k]
                b[k] = x[k]
            if use_gram:
                for k in range(m):
                    fa[k] = fv[k] - 2 * qj * gram[j, k]
                    fb[k] = fx[k]
            else:
                _matvec(f, a, fa)
                _matvec(f, b, fb)

        # numerical error can push the endpoint outside the constraints
        ok = True
        for k in range(m):
            t = g[k]
            for j in range(d):
                t += f[k, j] * x[j]
            if t <= 0:
                ok = False
                break
        if ok:
            for k in range(d):
                out[k, i] = x[k]
                last[k] = x[k]
            i += 1
            rejections = 0
        else:
            rejections += 1
            if rejections >= max_rejections:
                raise TMgenException(
                    f'No valid sample after {max_rejections} consecutive '
                    'rejections; the constraints may be degenerate '
                    '(for example an all-zero row of f with g = 0)')
    return result


cdef numpy.ndarray _mldivide(numpy.ndarray a, numpy.ndarray b):
    return numpy.linalg.solve(a, b)

cdef numpy.ndarray _mrdivide(numpy.ndarray a, numpy.ndarray b):
    return numpy.linalg.solve(a.T, b.T).T

cdef numpy.ndarray hmc_exact(numpy.ndarray f, numpy.ndarray g,
                             numpy.ndarray m, numpy.ndarray mu_r,
                             bool covariance, int num_samples,
                             numpy.ndarray initial_x, object rng):


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
    :param rng: None to draw from the global NumPy state, otherwise a seed or
        Generator
    :return:
        a :math:`d \times num_samples` matrix where each column is a sample
    """
    if num_samples < 1:
        raise TMgenException('num_samples must be at least 1')
    if f.ndim != 2:
        raise TMgenException('f must be a matrix')
    if g.ndim != 1:
        raise TMgenException('g must be a vector')
    if f.shape[0] != g.size:
        raise TMgenException('First dimension of f and g must match')
    cdef Py_ssize_t dim = f.shape[1]
    if (numpy.shape(m) != (dim, dim) or numpy.shape(mu_r) != (dim,)
            or numpy.shape(initial_x) != (dim,)):
        raise TMgenException(
            'm must be d x d, and mu_r and initial_x must have length d, '
            'where d is the second dimension of f')

    # work on copies, the caller's arrays must not be modified
    g = g.copy()
    initial_x = initial_x.astype(numpy.float64, copy=True)

    cdef numpy.ndarray mu
    # numpy's cholesky is lower triangular (m = lower @ lower.T), the
    # reference Matlab implementation uses the upper factor (lower.T)
    cdef numpy.ndarray lower = cholesky(m)
    if covariance:
        mu = mu_r
        g += numpy.matmul(f, mu)
        f = numpy.matmul(f, lower)
        initial_x = _mldivide(lower, initial_x - mu)
    else:
        mu = _mldivide(lower.T, _mldivide(lower, mu_r))
        g += numpy.matmul(f, mu)
        f = _mrdivide(f, lower.T)
        initial_x = numpy.matmul(lower.T, initial_x - mu)

    cdef int d = initial_x.shape[0]

    # Verify that initial_x is feasible
    cdef numpy.ndarray c = numpy.matmul(f, initial_x) + g
    if numpy.any(c < 0):
        raise TMgenException('Initial condition violates the constraints')

    if rng is None:
        # the global state keeps numpy.random.seed working
        sampler = numpy.random.mtrand._rand
    else:
        sampler = numpy.random.default_rng(rng)

    result_xs = _sample_loop(numpy.ascontiguousarray(f),
                             numpy.ascontiguousarray(g),
                             numpy.ascontiguousarray(initial_x),
                             num_samples, sampler, MAX_GRAM_BYTES,
                             MAX_CONSECUTIVE_REJECTIONS)

    # transform back to the unwhitened frame
    if covariance:
        result_xs = numpy.matmul(lower, result_xs) + mu[:, None]
    else:
        result_xs = _mldivide(lower.T, result_xs) + mu[:, None]

    return result_xs


def hmc_sample(f, g, m, mu_r, initial_x, int num_samples, covariance=True,
               *, rng=None):
    """
    Python-visible wrapper around :py:func:`hmc_exact`.
    Samples a Gaussian subject to :math:`f x + g \\geq 0`.

    :param rng: seed or :py:class:`numpy.random.Generator`; ``None`` draws
        from NumPy's global state
    :return: a :math:`d \\times num\\_samples` array, one sample per column
    """
    return hmc_exact(numpy.asarray(f, dtype=numpy.float64),
                     numpy.asarray(g, dtype=numpy.float64),
                     numpy.asarray(m, dtype=numpy.float64),
                     numpy.asarray(mu_r, dtype=numpy.float64),
                     bool(covariance), num_samples,
                     numpy.asarray(initial_x, dtype=numpy.float64), rng)
