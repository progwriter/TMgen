from pathlib import Path

import numpy
import pytest

from tmgen import TrafficMatrix, models
from tmgen.exceptions import TMgenException


def test_empty_constructor() -> None:
    with pytest.raises(TypeError):
        TrafficMatrix()  # type: ignore[call-arg]


def test_mean_and_worst_case() -> None:
    m = numpy.stack([numpy.ones((2, 2)), 3 * numpy.ones((2, 2))], axis=2)
    tm = TrafficMatrix(m)
    assert (tm.mean().matrix == 2).all()
    assert (tm.worst_case().matrix == 3).all()


@pytest.mark.parametrize("epochs", [1, 3])
def test_csv_shape(tmp_path: Path, epochs: int) -> None:
    tm = TrafficMatrix(numpy.random.rand(2, 2, epochs))
    f = tmp_path / "tm.csv"
    tm.to_csv(str(f))
    loaded = numpy.loadtxt(str(f), delimiter=",", ndmin=2)
    assert loaded.shape == ((2, 2) if epochs == 1 else (epochs, 4))


def test_constructor_copies_its_input() -> None:
    source = numpy.ones((2, 2, 1))
    tm = TrafficMatrix(source)
    source[:] = 5
    assert (tm.matrix == 1).all()
    assert not numpy.shares_memory(tm.matrix, source)


def test_matrix_is_read_only() -> None:
    tm = TrafficMatrix(numpy.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="read-only"):
        tm.matrix[0, 0, 0] = 2
    with pytest.raises(AttributeError):
        tm.matrix = numpy.zeros(3)  # type: ignore[misc]


def test_accessors_return_read_only_data() -> None:
    tm = TrafficMatrix(numpy.ones((2, 2, 3)))
    for view in (tm.at_time(0), tm.between(0, 1), tm.mean().matrix):
        assert isinstance(view, numpy.ndarray)
        assert not view.flags.writeable
    assert not tm.worst_case().matrix.flags.writeable
    assert not (tm + tm).matrix.flags.writeable


def test_accepts_lists_and_integers_as_float64() -> None:
    tm = TrafficMatrix([[[1], [2]], [[3], [4]]])
    assert tm.matrix.dtype == numpy.float64
    assert tm.matrix.shape == (2, 2, 1)
    assert tm.matrix[1, 0, 0] == 3.0


@pytest.mark.parametrize(
    "bad",
    [
        numpy.ones((2, 2)),
        numpy.ones((2, 3, 1)),
        numpy.ones((2, 2, 2, 2)),
        numpy.ones((0, 0, 1)),
        numpy.ones((2, 2, 0)),
        numpy.full((2, 2, 1), numpy.nan),
        numpy.full((2, 2, 1), numpy.inf),
        -numpy.ones((2, 2, 1)),
        [["a"]],
    ],
)
def test_invalid_matrix_raises(bad: object) -> None:
    with pytest.raises(TMgenException):
        TrafficMatrix(bad)  # type: ignore[arg-type]


def test_add() -> None:
    a = TrafficMatrix(numpy.ones((2, 2, 1)))
    b = TrafficMatrix(2 * numpy.ones((2, 2, 1)))
    assert ((a + b).matrix == 3).all()
    with pytest.raises(TMgenException):
        a + TrafficMatrix(numpy.ones((3, 3, 1)))
    with pytest.raises(TypeError):
        a + 1  # type: ignore[operator]


def test_models_return_read_only_matrices() -> None:
    assert not models.exp_tm(3, 1.0, num_epochs=2, rng=0).matrix.flags.writeable
    assert not models.gravity_tm([1, 2, 3], 6.0).matrix.flags.writeable


def test_pickle_round_trip_is_deprecated(tmp_path: Path) -> None:
    tm = TrafficMatrix(numpy.random.default_rng(0).random((3, 3, 2)))
    path = tmp_path / "tm.pickle"
    with pytest.warns(DeprecationWarning, match="to_pickle is deprecated"):
        tm.to_pickle(path)
    with pytest.warns(DeprecationWarning, match="from_pickle is deprecated"):
        loaded = TrafficMatrix.from_pickle(path)
    numpy.testing.assert_array_equal(loaded.matrix, tm.matrix)
    assert not loaded.matrix.flags.writeable


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    tm = TrafficMatrix(numpy.random.default_rng(0).random((3, 3, 2)))
    path = tmp_path / "tm.npz"
    tm.save(path)
    loaded = TrafficMatrix.load(path)
    numpy.testing.assert_array_equal(loaded.matrix, tm.matrix)
    assert not loaded.matrix.flags.writeable


def test_save_uses_the_given_file_name(tmp_path: Path) -> None:
    path = tmp_path / "no_extension"
    TrafficMatrix(numpy.ones((2, 2, 1))).save(path)
    assert path.exists()
    assert TrafficMatrix.load(path).num_nodes() == 2


def test_load_rejects_pickled_data(tmp_path: Path) -> None:
    path = tmp_path / "evil.npz"
    numpy.savez(path, matrix=numpy.array([object()], dtype=object), format_version=1)
    with pytest.raises(TMgenException):
        TrafficMatrix.load(path)


def test_load_rejects_a_pickle_file(tmp_path: Path) -> None:
    path = tmp_path / "tm.pickle"
    with pytest.warns(DeprecationWarning):
        TrafficMatrix(numpy.ones((2, 2, 1))).to_pickle(path)
    with pytest.raises(TMgenException):
        TrafficMatrix.load(path)


def test_load_rejects_other_numpy_files(tmp_path: Path) -> None:
    plain = tmp_path / "plain.npy"
    numpy.save(plain, numpy.ones((2, 2, 1)))
    with pytest.raises(TMgenException, match="not a TMgen"):
        TrafficMatrix.load(plain)
    other = tmp_path / "other.npz"
    numpy.savez(other, something=numpy.ones(3))
    with pytest.raises(TMgenException, match="not a TMgen"):
        TrafficMatrix.load(other)


def test_load_rejects_newer_format_versions(tmp_path: Path) -> None:
    path = tmp_path / "future.npz"
    numpy.savez(path, matrix=numpy.ones((2, 2, 1)), format_version=99)
    with pytest.raises(TMgenException, match="version 99"):
        TrafficMatrix.load(path)


def test_load_validates_the_data(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    numpy.savez(path, matrix=-numpy.ones((2, 2, 1)), format_version=1)
    with pytest.raises(TMgenException):
        TrafficMatrix.load(path)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TrafficMatrix.load(tmp_path / "missing.npz")
