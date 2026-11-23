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


def test_csv_is_long_format(tmp_path: Path) -> None:
    values = numpy.arange(8.0).reshape((2, 2, 2))
    path = tmp_path / "tm.csv"
    TrafficMatrix(values).to_csv(path)
    lines = path.read_text().splitlines()
    assert lines[0] == "epoch,ingress,egress,volume"
    assert len(lines) == 1 + 8
    assert lines[1] == "0,0,0,0"
    assert lines[2] == "0,0,1,2"  # matrix[0, 1, 0]
    assert lines[-1] == "1,1,1,7"


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 2, 1), (3, 3, 4), (4, 4, 2)])
def test_csv_round_trip(tmp_path: Path, shape: tuple[int, int, int]) -> None:
    tm = TrafficMatrix(numpy.random.default_rng(0).random(shape))
    path = tmp_path / "tm.csv"
    tm.to_csv(path)
    numpy.testing.assert_array_equal(TrafficMatrix.from_csv(path).matrix, tm.matrix)


def test_csv_rows_in_any_order(tmp_path: Path) -> None:
    path = tmp_path / "tm.csv"
    path.write_text("epoch,ingress,egress,volume\n1,0,0,5\n0,0,0,3\n")
    assert TrafficMatrix.from_csv(path).matrix.tolist() == [[[3.0, 5.0]]]


@pytest.mark.parametrize(
    "content",
    [
        "",
        "a,b,c,d\n0,0,0,1\n",
        "epoch,ingress,egress,volume\n",
        "epoch,ingress,egress,volume\n0,0,0,x\n",
        "epoch,ingress,egress,volume\n0,0,0\n",
        "epoch,ingress,egress,volume\n0,0,0.5,1\n",
        "epoch,ingress,egress,volume\n0,0,-1,1\n",
        "epoch,ingress,egress,volume\n0,0,0,1\n0,0,0,2\n",
        "epoch,ingress,egress,volume\n0,0,0,1\n0,1,1,2\n",
        "epoch,ingress,egress,volume\n0,0,0,-1\n",
    ],
)
def test_csv_rejects_malformed_files(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(TMgenException):
        TrafficMatrix.from_csv(path)


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
    assert TrafficMatrix.load(path).num_nodes == 2


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


def test_size_properties() -> None:
    tm = TrafficMatrix(numpy.ones((3, 3, 5)))
    assert tm.num_nodes == 3
    assert tm.num_epochs == 5
    assert tm.shape == (3, 3, 5)
    assert len(tm) == 5


def test_calling_the_size_properties_still_works_but_warns() -> None:
    tm = TrafficMatrix(numpy.ones((3, 3, 5)))
    with pytest.warns(DeprecationWarning, match="drop the parentheses") as record:
        assert tm.num_nodes() == 3  # type: ignore[operator]
        assert tm.num_epochs() == 5  # type: ignore[operator]
    assert all(r.filename == __file__ for r in record)


def _tm() -> TrafficMatrix:
    return TrafficMatrix(numpy.arange(8.0).reshape((2, 2, 2)))


def test_numpy_functions_accept_a_traffic_matrix() -> None:
    tm = _tm()
    assert numpy.asarray(tm) is tm.matrix
    assert numpy.sum(tm) == 28.0
    assert numpy.asarray(tm, dtype=numpy.float32).dtype == numpy.float32
    copy = numpy.array(tm)
    copy[0, 0, 0] = 99  # a copy is writable
    assert tm.matrix[0, 0, 0] == 0.0
    with pytest.raises(ValueError, match="copy"):
        numpy.asarray(tm, dtype=numpy.float32, copy=False)


def test_indexing_follows_the_array() -> None:
    tm = _tm()
    assert tm[0, 1, 1] == 3.0
    numpy.testing.assert_array_equal(tm[0, 1], [2.0, 3.0])
    numpy.testing.assert_array_equal(tm[:, :, 1], tm.at_time(1))
    assert not tm[0].flags.writeable


def test_iteration_yields_the_epochs() -> None:
    tm = _tm()
    epochs = list(tm)
    assert len(epochs) == len(tm) == 2
    numpy.testing.assert_array_equal(epochs[1], tm.at_time(1))


def test_equality() -> None:
    assert _tm() == _tm()
    assert _tm() != TrafficMatrix(numpy.ones((2, 2, 2)))
    assert _tm() != TrafficMatrix(numpy.ones((3, 3, 1)))
    assert _tm() != "not a matrix"
    with pytest.raises(TypeError):
        hash(_tm())


def test_scaling() -> None:
    tm = _tm()
    assert (2 * tm).matrix.sum() == 56.0
    assert (tm * 0.5) == TrafficMatrix(tm.matrix / 2)
    assert (tm / 4).matrix.max() == 7 / 4
    assert (tm * numpy.float64(2)) == 2 * tm
    assert not (tm * 2).matrix.flags.writeable


@pytest.mark.parametrize("factor", [-1, float("nan"), float("inf")])
def test_scaling_rejects_invalid_factors(factor: float) -> None:
    with pytest.raises(TMgenException):
        _tm() * factor
    with pytest.raises(TMgenException):
        _tm() / factor


def test_division_by_zero_raises() -> None:
    with pytest.raises(TMgenException, match="positive"):
        _tm() / 0


def test_scaling_by_non_numbers_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        _tm() * "a"  # type: ignore[operator]
    with pytest.raises(TypeError):
        _tm() * True
    with pytest.raises(TypeError):
        _tm() * _tm()  # type: ignore[operator]


def test_modulate_scales_the_matrix_per_epoch() -> None:
    base = TrafficMatrix(numpy.array([[1.0, 2.0], [3.0, 4.0]])[:, :, None])
    result = base.modulate([0.0, 1.0, 2.5])
    assert result.shape == (2, 2, 3)
    numpy.testing.assert_array_equal(result.at_time(0), numpy.zeros((2, 2)))
    numpy.testing.assert_array_equal(result.at_time(1), base.at_time(0))
    numpy.testing.assert_array_equal(result.at_time(2), 2.5 * base.at_time(0))
    assert not result.matrix.flags.writeable


def test_modulate_composes_with_models() -> None:
    totals = numpy.array([2.0, 6.0, 4.0])
    spatial = models.gravity_tm([1, 2, 3], totals.mean())
    result = spatial.modulate(totals / totals.mean())
    numpy.testing.assert_allclose(result.matrix.sum(axis=(0, 1)), totals)


def test_modulate_rejects_multi_epoch_matrices() -> None:
    with pytest.raises(TMgenException, match="single-epoch"):
        TrafficMatrix(numpy.ones((2, 2, 2))).modulate([1.0])


@pytest.mark.parametrize(
    "profile", [[], [[1.0]], [-1.0], [float("nan")], [float("inf")], ["a"], 1.0]
)
def test_modulate_rejects_invalid_profiles(profile: object) -> None:
    with pytest.raises(TMgenException):
        TrafficMatrix(numpy.ones((2, 2, 1))).modulate(profile)  # type: ignore[arg-type]


def test_repr_is_short_and_str_shows_the_values() -> None:
    tm = TrafficMatrix(numpy.arange(8.0).reshape((2, 2, 2)))
    assert repr(tm) == "TrafficMatrix(num_nodes=2, num_epochs=2)"
    assert str(tm) == str(tm.matrix)
    big = TrafficMatrix(numpy.ones((300, 300, 100)))
    assert len(repr(big)) < 80


def test_constructor_and_at_time_accept_the_old_keywords() -> None:
    with pytest.warns(DeprecationWarning, match="'tm' is deprecated, use 'matrix'"):
        tm = TrafficMatrix(tm=numpy.arange(8.0).reshape((2, 2, 2)))  # type: ignore[call-arg]
    assert TrafficMatrix(matrix=tm.matrix) == tm
    numpy.testing.assert_array_equal(tm.at_time(epoch=1), tm.at_time(1))
    with pytest.warns(DeprecationWarning, match="'t' is deprecated, use 'epoch'"):
        numpy.testing.assert_array_equal(tm.at_time(t=1), tm.at_time(1))  # type: ignore[call-arg]
    numpy.testing.assert_array_equal(tm.at_time(-1), tm.at_time(1))
