# coding=utf-8
import numpy
cimport numpy

cdef class TrafficMatrix:
    cdef public numpy.ndarray matrix
    cpdef numpy.ndarray at_time(self, int t)
    cpdef between(self, int o, int d, str modestr=*)
    cpdef to_pickle(self, fname)
    cpdef TrafficMatrix worst_case(self)
    cpdef int num_pops(self)
    cpdef int num_epochs(self)