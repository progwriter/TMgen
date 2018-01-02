# coding=utf-8
import numpy
cimport numpy
from cpython cimport bool

# cdef numpy.ndarray mldivide(numpy.ndarray a, numpy.ndarray b)
# cdef numpy.ndarray mrdivide(numpy.ndarray a, numpy.ndarray b)
cdef numpy.ndarray hmc_exact(numpy.ndarray f, numpy.ndarray g,
                             numpy.ndarray constraint_m, numpy.ndarray mu_r,
                             bool cov,
                             int num_samples, numpy.ndarray initial_x)
