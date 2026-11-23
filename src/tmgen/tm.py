import os
import pickle

import numpy
from numpy.typing import ArrayLike, NDArray

from tmgen.exceptions import TMgenException

_Path = str | os.PathLike[str]


def _canonical_mode(m: str) -> str:
    m = m.lower()
    if m in ("all", "*"):
        return "all"
    if m in ("min", "mininum"):
        return "min"
    if m in ("max", "maximum"):
        return "max"
    if m in ("mean", "average"):
        return "mean"
    raise TMgenException("Unknown entry choice mode")


class TrafficMatrix:
    """
    Represents a traffic matrix: non-negative traffic volumes between
    ``num_nodes`` nodes over ``num_epochs`` time epochs.

    The data is copied on construction and exposed read-only through
    :py:attr:`matrix`, so a matrix cannot change after it is created. Arrays
    returned by the accessors are read-only views of that data; copy them
    before modifying.
    """

    def __init__(self, tm: ArrayLike) -> None:
        """
        Create a new traffic matrix from the given array

        :param tm: array-like of non-negative, finite numbers with three
            dimensions, n x n x m. If the matrix has no variability in time,
            then m=1. First two dimensions must match, as a single traffic
            matrix is always n x n. The data is converted to float64 and
            copied.
        """
        try:
            matrix = numpy.array(tm, dtype=numpy.float64)
        except (TypeError, ValueError):
            raise TMgenException("Traffic matrix must be an array of numbers") from None
        self._set_matrix(matrix)
        if not numpy.isfinite(matrix).all() or (matrix < 0).any():
            raise TMgenException("Traffic volumes must be finite and non-negative")

    @classmethod
    def _from_owned(cls, matrix: NDArray[numpy.float64]) -> "TrafficMatrix":
        """Wrap a float64 array that nothing else references, without copying."""
        self = cls.__new__(cls)
        self._set_matrix(matrix)
        return self

    def _set_matrix(self, matrix: NDArray[numpy.float64]) -> None:
        if matrix.ndim != 3:  # 2d + time
            raise TMgenException(
                "Traffic matrix must have 3 dimensions: n x n x m, "
                "where n is number of nodes and m is number of "
                "epochs (at least 1)"
            )
        if matrix.shape[0] != matrix.shape[1]:  # num_nodes both ways
            raise TMgenException(
                "First two dimensions of the traffic matrix must match"
            )
        if matrix.size == 0:
            raise TMgenException("Traffic matrix needs at least one node and epoch")
        matrix.flags.writeable = False
        self._matrix = matrix

    @property
    def matrix(self) -> NDArray[numpy.float64]:
        """The traffic volumes as a read-only n x n x m array."""
        return self._matrix

    def at_time(self, t: int) -> NDArray[numpy.float64]:
        """
        Return a numpy array reprenseting a single TM at epoch t.

        :param t: the number of the epoch (0-indexed).
        :rtype: numpy array
        """
        return self.matrix[:, :, t]

    def between(
        self, o: int, d: int, modestr: str = "all"
    ) -> NDArray[numpy.float64] | float:
        """
        Return the traffic matrix between the given ingress and egress nodes.
        This method supports multiple temporal modes: 'all', 'min', 'max', and
        'mean'

        :param o: source node (origin)
        :param d: destination node
        :param modestr: temporal mode
        :return: Numpy array if modestr=='all', double otherwise
        """
        mode = _canonical_mode(modestr)
        series = self.matrix[o, d, :]
        if mode == "all":
            return series
        if mode == "min":
            return float(numpy.min(series))
        if mode == "max":
            return float(numpy.max(series))
        return float(numpy.mean(series))

    def to_pickle(self, fname: _Path) -> None:
        """
        Save the matrix to a python pickle file

        :param fname: the file name
        """
        with open(fname, "wb") as f:
            pickle.dump(self.matrix, f)

    def to_csv(self, fname: _Path) -> None:
        """
        Save the matrix to a CSV file.

        .. warning::
            Note that if the matrix has more than one epoch, it will be "flattened".
            In this case each line will have :math:`n^2` entries.
            Each row is a single epoch with a flattened array of ingress-egress nodes.
        """
        if self.num_epochs() > 1:
            m = self.matrix.reshape((self.num_nodes() ** 2, self.num_epochs())).T
            numpy.savetxt(fname, m, delimiter=",")
        else:
            numpy.savetxt(fname, self.matrix[:, :, 0], delimiter=",")

    def mean(self) -> "TrafficMatrix":
        """
        Returns a new traffic matrix that is the average across all epochs.

        """
        return TrafficMatrix._from_owned(self.matrix.mean(axis=2, keepdims=True))

    def worst_case(self) -> "TrafficMatrix":
        """
        Return a new, single-epoch traffic matrix that chooses maximum volume
        per OD pair (in time)

        :return: a new TrafficMatrix
        """
        return TrafficMatrix._from_owned(self.matrix.max(axis=2, keepdims=True))

    def num_nodes(self) -> int:
        """
        :return: The number of nodes in this traffic matrix
        """
        return int(self.matrix.shape[0])

    def num_epochs(self) -> int:
        """
        :return:  The number of epochs in this traffic matrix
        """
        return int(self.matrix.shape[2])

    @staticmethod
    def from_pickle(fname: _Path) -> "TrafficMatrix":
        """
        Load a TrafficMatrix object from a file

        :param fname: the file name on disk
        :return: new TrafficMatrix
        """
        with open(fname, "rb") as f:
            return TrafficMatrix(pickle.load(f))

    def __repr__(self) -> str:
        return repr(self.matrix)

    def __len__(self) -> int:
        return self.num_epochs()

    def __add__(self, other: "TrafficMatrix") -> "TrafficMatrix":
        if not isinstance(other, TrafficMatrix):
            return NotImplemented
        if self.matrix.shape != other.matrix.shape:
            raise TMgenException(
                f"Cannot add traffic matrices of shapes {self.matrix.shape} "
                f"and {other.matrix.shape}"
            )
        return TrafficMatrix._from_owned(self.matrix + other.matrix)
