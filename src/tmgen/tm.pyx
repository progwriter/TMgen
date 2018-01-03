# coding=utf-8

import numpy
cimport numpy
from six.moves import cPickle  # for py 2/3 compat

from tmgen.exceptions import TMgenException

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
        raise TMgenException("Unknown entry choice mode")

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
        """
        if not tm.ndim == 3:  # 2d + time
            raise TMgenException(
                'Traffic matrix must have 3 dimensions: n x n x m,'
                'where n is number of nodes and m is number of '
                'epochs (at least 1)')
        if not tm.shape[0] == tm.shape[1]:  # num_nodes both ways
            raise TMgenException(
                'First two dimentions of the traffic matrix must match')
        self.matrix = tm

    cpdef numpy.ndarray at_time(self, int t):
        """
        Return a numpy array reprenseting a single TM at epoch t.

        :param t: the number of the epoch (0-indexed).
        :rtype: numpy array
        """
        return self.matrix[:, :, t]

    cpdef between(self, int o, int d, modestr='all'):
        """
        Return the traffic matrix between the given ingress and egress nodes.
        This method supports multiple temporal modes: 'all', 'min', 'max', and
        'mean'

        :param o: source node (origin)
        :param d: destination node
        :param modestr: temporal mode
        :return: Numpy array if modestr=='all', double otherwise
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
        with open(fname, 'wb') as f:
            cPickle.dump(self.matrix, f)

    cpdef to_csv(self, fname):
        """
        Save the matrix to a CSV file.

        .. warning:
            Note that if the matrix has more than one epoch, it will be "flattened".
            In this case each line will have :math:`n^2 entries`.
            Each row is a single epoch with a flattened array of ingress-egress nodes.
        """
        if self.num_epochs() > 1:
            m = self.matrix.reshape((self.num_nodes() ** 2, self.num_epochs()))
            numpy.savetxt(fname, m, delimiter=',')
        else:
            numpy.savetxt(fname, self.matrix, delimiter=',')

    cpdef TrafficMatrix mean(self):
        """
        Returns a new traffic matrix that is the average across all epochs.

        """
        return TrafficMatrix(numpy.reshape(self.matrix.max(axis=2),
                                           (self.num_nodes(), self.num_nodes(),
                                            1)))

    cpdef TrafficMatrix worst_case(self):
        """
        Return a new, single-epoch traffic matrix that chooses maximum volume
        per OD pair (in time)

        :return: a new TrafficMatrix
        """
        return TrafficMatrix(numpy.reshape(self.matrix.max(axis=2),
                                           (self.num_nodes(), self.num_nodes(),
                                            1)))

    cpdef int num_nodes(self):
        """
        :return: The number of nodes in this traffic matrix
        """
        return self.matrix.shape[0]

    cpdef int num_epochs(self):
        """
        :return:  The number of epochs in this traffic matrix
        """
        return self.matrix.shape[2]

    @staticmethod
    def from_pickle(fname):
        """
        Load a TrafficMatrix object from a file

        :param fname: the file name on disk
        :return: new TrafficMatrix
        """
        with open(fname, 'rb') as f:
            return TrafficMatrix(cPickle.load(f))

    def __repr__(self):
        return repr(self.matrix)

    def __len__(self):
        return self.num_epochs()

    def __add__(self, other):
        return TrafficMatrix(self.matrix + other.matrix)
