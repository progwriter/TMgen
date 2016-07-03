# coding=utf-8
cimport numpy
from six.moves import cPickle # for py 2/3 compat

ctypedef enum EntryMode:
    ALL, MIN, MAX, MEAN

cdef EntryMode _mode_to_enum(str m):
    # Convert the string to enum
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
        if not tm.ndim == 3:  # 2d + time
            raise ValueError('Traffic matrix must have 3 dimensions: n x n x m,'
                             'where n is number of nodes and m is number of '
                             'epochs (at least 1)')
        if not tm.shape[0] == tm.shape[1]:  # num_pops both ways
            raise ValueError('Traffic matrix dimensions must match')


    cpdef numpy.ndarray at_time(self, int t):
        """
        Return a numpy ndarray reprenseting a single TM at epoch t.
        :param t: the number of the epoch (0-indexed).
        :return: numpy ndarray
        """
        return self.matrix[:, :, t]

    cpdef between(self, int o, int d, str modestr='all'):
        """
        Return the traffic matrix between the given ingress and egress nodes.
        This method supports multiple temporal modes: 'all', 'min', 'max', and 'mean'
        :param o:
        :param d:
        :param modestr:
        :return: numpy ndarray if modestr=='all', double otherwise
        """
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
        Save the matrix to a python pickle file
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
    def from_pickle(str fname):
        """
        Load a TrafficMatrix object from a file
        :param fname: the file name on disk
        :return: new TraffixMatrix
        """
        with open(fname, 'r') as f:
            return TrafficMatrix(cPickle.load(f))

    def __repr__(self):
        return repr(self.matrix)

    def __len__(self):
        return self.num_epochs()

    def __add__(self, other):
        return TrafficMatrix(self.matrix + other.matrix)
