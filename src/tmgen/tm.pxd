# coding=utf-8
import numpy
cimport numpy

cdef class TrafficMatrix:
    """
        Spatio-temporal traffix matrix object
    """
    cdef public numpy.ndarray matrix
    cpdef numpy.ndarray at_time(self, int t)
    cpdef TrafficMatrix worst_case(self)
    cpdef TrafficMatrix mean(self)
    cpdef int num_nodes(self)
    cpdef int num_epochs(self)

    # I/O functions
    cpdef between(self, int o, int d, modestr=*)
    cpdef to_pickle(self, fname)
    cpdef to_csv(self, fname)
