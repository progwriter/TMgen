cimport numpy
import cPickle

from tm cimport entry_mode

cdef class TrafficMatrix:
    def __init__(self, numpy.ndarray tm):
        self.matrix = tm

    cpdef at_time(self, int t):
        return self.matrix[:, :, t]

    cpdef between(self, int o, int d, entry_mode mode):
        if mode == ALL:
            return self.matrix[o, d, :]
        elif mode == MIN:
            return numpy.min(self.matrix[o, d, :])
        elif mode == MAX:
            return numpy.max(self.matrix[o, d, :])
        elif mode == MEAN:
            return numpy.mean(self.matrix[o, d, :])

    cpdef to_pickle(self, fname):
        with open(fname, 'w') as f:
            cPickle.dump(self.matrix, f)

    cpdef worst_case(self):
        return self.matrix.max(axis=2)

    @staticmethod
    def from_pickle(self, fname):
        with open(fname, 'r') as f:
            return TrafficMatrix(cPickle.load(f))

    def __repr__(self):
        return repr(self.matrix)
