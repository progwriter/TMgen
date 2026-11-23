import os
import pickle
import warnings
import zipfile
from typing import Literal, overload

import numpy
from numpy.typing import ArrayLike, NDArray

from tmgen._compat import renamed_arguments
from tmgen.exceptions import TMgenException

_Path = str | os.PathLike[str]

FORMAT_VERSION = 1
_CSV_COLUMNS = ("epoch", "ingress", "egress", "volume")


_LEGACY_AGGREGATIONS: dict[str, str | None] = {
    "all": None,
    "*": None,
    "min": "min",
    "mininum": "min",
    "max": "max",
    "maximum": "max",
    "mean": "mean",
    "average": "mean",
}


def _aggregation(agg: str | None) -> str | None:
    if agg is None or agg in ("min", "max", "mean"):
        return agg
    if isinstance(agg, str) and agg.lower() in _LEGACY_AGGREGATIONS:
        warnings.warn(
            f"agg={agg!r} is deprecated, use None, 'min', 'max' or 'mean'",
            DeprecationWarning,
            stacklevel=4,  # the caller of between(), which is wrapped
        )
        return _LEGACY_AGGREGATIONS[agg.lower()]
    raise TMgenException(f"agg must be None, 'min', 'max' or 'mean', got {agg!r}")


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

    @overload
    def between(
        self, ingress: int, egress: int, agg: None = None
    ) -> NDArray[numpy.float64]: ...

    @overload
    def between(
        self, ingress: int, egress: int, agg: Literal["min", "max", "mean"]
    ) -> float: ...

    @renamed_arguments(o="ingress", d="egress", modestr="agg")
    def between(
        self, ingress: int, egress: int, agg: str | None = None
    ) -> NDArray[numpy.float64] | float:
        """
        Return the traffic volumes from the ingress node to the egress node.

        :param ingress: ingress (source) node
        :param egress: egress (destination) node
        :param agg: ``None`` for the volume at each epoch (a read-only array),
            or ``"min"``, ``"max"`` or ``"mean"`` to aggregate over the epochs
            into a single number
        :return: an array with one volume per epoch, or the aggregate
        """
        mode = _aggregation(agg)
        series = self.matrix[ingress, egress, :]
        if mode is None:
            return series
        if mode == "min":
            return float(numpy.min(series))
        if mode == "max":
            return float(numpy.max(series))
        return float(numpy.mean(series))

    def save(self, fname: _Path) -> None:
        """
        Save the matrix to a compressed NumPy ``.npz`` file, which
        :py:meth:`load` reads back. Unlike a pickle, loading it cannot run
        code. The file name is used as given, so include the ``.npz``
        extension.

        :param fname: the file name
        """
        with open(fname, "wb") as f:
            numpy.savez_compressed(
                f,
                matrix=self.matrix,
                format_version=numpy.int64(FORMAT_VERSION),
            )

    @staticmethod
    def load(fname: _Path) -> "TrafficMatrix":
        """
        Load a traffic matrix saved with :py:meth:`save`. Pickled data is
        never loaded, so files from untrusted sources are safe to read.

        :param fname: the file name on disk
        :return: new TrafficMatrix
        :raises TMgenException: if the file is not a TMgen file, or was written
            by a newer version of TMgen
        """
        not_ours = TMgenException(f"{fname} is not a TMgen traffic matrix file")
        try:
            data = numpy.load(fname, allow_pickle=False)
        except (ValueError, zipfile.BadZipFile) as err:
            raise not_ours from err
        if not isinstance(data, numpy.lib.npyio.NpzFile):
            raise not_ours
        with data:
            if "matrix" not in data.files or "format_version" not in data.files:
                raise not_ours
            try:
                version = int(data["format_version"])
                matrix = data["matrix"]
            except (TypeError, ValueError) as err:  # e.g. object arrays
                raise not_ours from err
            if version > FORMAT_VERSION:
                raise TMgenException(
                    f"{fname} uses file format version {version}, but this "
                    f"version of TMgen reads up to {FORMAT_VERSION}"
                )
        return TrafficMatrix(matrix)  # validates data from outside

    def to_pickle(self, fname: _Path) -> None:
        """
        Save the matrix to a python pickle file.

        .. deprecated:: 0.2
            Use :py:meth:`save`, which writes a file that is safe to load.

        :param fname: the file name
        """
        warnings.warn(
            "to_pickle is deprecated, use save", DeprecationWarning, stacklevel=2
        )
        with open(fname, "wb") as f:
            pickle.dump(self.matrix, f)

    def to_csv(self, fname: _Path) -> None:
        """
        Save the matrix to a CSV file in long format: a header line
        ``epoch,ingress,egress,volume`` and one line per entry, which
        :py:meth:`from_csv` reads back.

        :param fname: the file name
        """
        epochs, ingress, egress = numpy.meshgrid(
            numpy.arange(self.num_epochs()),
            numpy.arange(self.num_nodes()),
            numpy.arange(self.num_nodes()),
            indexing="ij",
        )
        rows = numpy.column_stack(
            [
                epochs.ravel(),
                ingress.ravel(),
                egress.ravel(),
                self.matrix.transpose(2, 0, 1).ravel(),
            ]
        )
        numpy.savetxt(
            fname,
            rows,
            delimiter=",",
            header=",".join(_CSV_COLUMNS),
            comments="",
            fmt=["%d", "%d", "%d", "%.17g"],
        )

    @staticmethod
    def from_csv(fname: _Path) -> "TrafficMatrix":
        """
        Load a traffic matrix from a long-format CSV file, as written by
        :py:meth:`to_csv`. Every combination of epoch, ingress and egress
        must appear exactly once.

        :param fname: the file name on disk
        :return: new TrafficMatrix
        """
        bad = TMgenException(
            f"{fname} is not a traffic matrix CSV with the columns "
            f"{','.join(_CSV_COLUMNS)}, one line per entry"
        )
        with open(fname) as f:
            if f.readline().strip().split(",") != list(_CSV_COLUMNS):
                raise bad
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # empty file
                rows = numpy.loadtxt(fname, delimiter=",", skiprows=1, ndmin=2)
        except ValueError as err:
            raise bad from err
        if rows.shape[1] != 4 or rows.shape[0] == 0:
            raise bad
        index = rows[:, :3]
        if (index < 0).any() or (index != numpy.floor(index)).any():
            raise bad
        epoch, ingress, egress = index.astype(numpy.int64).T
        num_nodes = int(max(ingress.max(), egress.max())) + 1
        num_epochs = int(epoch.max()) + 1
        entries = num_nodes * num_nodes * num_epochs
        position = (epoch * num_nodes + ingress) * num_nodes + egress
        if rows.shape[0] != entries or numpy.unique(position).size != entries:
            raise TMgenException(
                f"{fname} must have exactly one line for each of the {entries} "
                f"combinations of {num_epochs} epochs and {num_nodes} nodes"
            )
        matrix = numpy.empty(entries)
        matrix[position] = rows[:, 3]
        return TrafficMatrix(
            matrix.reshape((num_epochs, num_nodes, num_nodes)).transpose(1, 2, 0)
        )

    def mean(self) -> "TrafficMatrix":
        """
        Returns a new traffic matrix that is the average across all epochs.

        """
        return TrafficMatrix._from_owned(self.matrix.mean(axis=2, keepdims=True))

    def worst_case(self) -> "TrafficMatrix":
        """
        Return a new, single-epoch traffic matrix that chooses maximum volume
        per ingress-egress pair (over time)

        :return: a new TrafficMatrix
        """
        return TrafficMatrix._from_owned(self.matrix.max(axis=2, keepdims=True))

    def zero_diagonal(self) -> "TrafficMatrix":
        """
        Return a new traffic matrix with the diagonal (traffic from a node to
        itself) set to zero in every epoch. The totals decrease by the volume
        removed.

        :return: a new TrafficMatrix
        """
        matrix = self.matrix.copy()
        nodes = numpy.arange(self.num_nodes())
        matrix[nodes, nodes, :] = 0
        return TrafficMatrix._from_owned(matrix)

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
        Load a TrafficMatrix object from a pickle file.

        .. deprecated:: 0.2
            Use :py:meth:`load`. Unpickling can run arbitrary code, so only
            load pickle files you created yourself.

        :param fname: the file name on disk
        :return: new TrafficMatrix
        """
        warnings.warn(
            "from_pickle is deprecated, use load (pickles can run code when loaded)",
            DeprecationWarning,
            stacklevel=2,
        )
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
