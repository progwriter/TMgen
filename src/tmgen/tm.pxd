import numpy
cimport numpy

cdef class TrafficMatrix:
    cdef public numpy.ndarray matrix