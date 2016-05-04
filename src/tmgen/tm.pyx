# coding=utf-8
import numpy
cimport numpy
from six.moves import cPickle

ctypedef enum EntryMode:
    ALL, MIN, MAX, MEAN

cdef EntryMode _mode_to_enum(str m):
    m = m.lower()
    if m == 'all' or m == '*':
        return ALL
    elif m == 'min' or m == 'mininum':
        return MIN
    elif m == 'max' or m == 'maximum':
        return MAX
    elif m == 'mean' or m == 'average':
        return MEAN
    else:
        raise ValueError("Unknown entry choice mode")

cdef class TrafficMatrix:
    """
    Represents a traffic matrix.
    """
    def __init__(self, numpy.ndarray tm):
        """
        Create a new traffic matrix from the given array
        :param tm: the Numpy ndarray. Must have three dimetions, n x n x m.
            If the matrix has no variability in time, them m=1
            First two dimensions must match, as a single traffic matrix is
            always n x n.
        :return:
        """
        self.matrix = tm
        assert tm.shape[0] == tm.shape[1]  # num_pops both ways
        assert tm.ndim == 3  # 2d + time

    cpdef numpy.ndarray at_time(self, int t):
        """
        Return a numpy ndarray reprenseting a single TM at epoch t.
        :param t: the number of the epoch (0-indexed).
        :return: numpy ndarray
        """
        return self.matrix[:, :, t]

    cpdef between(self, int o, int d, str modestr='all'):
        mode = _mode_to_enum(modestr)
        if mode == ALL:
            return self.matrix[o, d, :]
        elif mode == MIN:
            return numpy.min(self.matrix[o, d, :])
        elif mode == MAX:
            return numpy.max(self.matrix[o, d, :])
        elif mode == MEAN:
            return numpy.mean(self.matrix[o, d, :])

    cpdef to_pickle(self, fname):
        """
        Save to a python pickle file
        :param fname: the file name
        """
        with open(fname, 'w') as f:
            cPickle.dump(self.matrix, f)

    cpdef TrafficMatrix worst_case(self):
        """
        Return a new traffic matrix that chooses maximum traffic through time
        :return: a new TrafficMatrix
        """
        return TrafficMatrix(numpy.reshape(self.matrix.max(axis=2),
                                           (self.num_pops(), self.num_pops(),
                                            1)))

    cpdef int num_pops(self):
        """
        :return: The number of PoPs in this traffic matrix
        """
        return self.matrix.shape[0]

    cpdef int num_epochs(self):
        return self.matrix.shape[2]

    @staticmethod
    def from_pickle(fname):
        """
        Load a TrafficMatrix object from a file
        :param fname: the file name on disk
        :return: new TraffixMatrix
        """
        with open(fname, 'r') as f:
            return TrafficMatrix(cPickle.load(f))

    def __repr__(self):
        return repr(self.matrix)
