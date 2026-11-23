import os
import pickle

import numpy
from numpy.typing import NDArray

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
    Represents a traffic matrix.
    """

    matrix: NDArray[numpy.float64]

    def __init__(self, tm: NDArray[numpy.float64]) -> None:
        """
        Create a new traffic matrix from the given array

        :param tm: the Numpy ndarray. Must have three dimetions, n x n x m.
            If the matrix has no variability in time, them m=1
            First two dimensions must match, as a single traffic matrix is
            always n x n.
        """
        if not isinstance(tm, numpy.ndarray):
            raise TypeError(f"Expected a numpy array, got {type(tm).__name__}")
        if not tm.ndim == 3:  # 2d + time
            raise TMgenException(
                "Traffic matrix must have 3 dimensions: n x n x m,"
                "where n is number of nodes and m is number of "
                "epochs (at least 1)"
            )
        if not tm.shape[0] == tm.shape[1]:  # num_nodes both ways
            raise TMgenException(
                "First two dimentions of the traffic matrix must match"
            )
        self.matrix = tm

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
        return TrafficMatrix(
            numpy.reshape(
                self.matrix.mean(axis=2), (self.num_nodes(), self.num_nodes(), 1)
            )
        )

    def worst_case(self) -> "TrafficMatrix":
        """
        Return a new, single-epoch traffic matrix that chooses maximum volume
        per OD pair (in time)

        :return: a new TrafficMatrix
        """
        return TrafficMatrix(
            numpy.reshape(
                self.matrix.max(axis=2), (self.num_nodes(), self.num_nodes(), 1)
            )
        )

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
        return TrafficMatrix(self.matrix + other.matrix)
