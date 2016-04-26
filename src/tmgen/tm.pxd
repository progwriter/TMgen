import numpy
cimport numpy

ctypedef enum entry_mode: ALL, MIN, MAX, MEAN

cdef class TrafficMatrix:
    cdef public numpy.ndarray matrix
    cpdef at_time(self, int t)
    cpdef between(self, int o, int d, entry_mode mode)
    cpdef to_pickle(self, fname)
    cpdef worst_case(self)