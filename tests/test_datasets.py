import gzip
import hashlib
import os
from pathlib import Path

import numpy
import pytest

from tmgen import TMgenException
from tmgen.datasets import (
    ABILENE_NODES,
    _fetch,
    _parse_abilene,
    data_dir,
    load_abilene,
)

PAIRS = len(ABILENE_NODES) ** 2


def _week_text(epochs: int) -> str:
    """Rows in the real file's layout: five values per flow, real traffic first."""
    rows = []
    for epoch in range(epochs):
        values = numpy.zeros(PAIRS * 5)
        values[0::5] = numpy.arange(PAIRS) + 1000 * epoch
        values[1::5] = -1
        rows.append(" ".join(f"{v:.7e}" for v in values))
    return "\n".join(rows) + "\n"


def _write_week(path: Path, epochs: int = 3) -> None:
    with gzip.open(path, "wt") as stream:
        stream.write(_week_text(epochs))


def test_parse_keeps_the_real_traffic_in_bytes_per_second(tmp_path: Path) -> None:
    path = tmp_path / "week.gz"
    _write_week(path)
    rates = _parse_abilene(path)
    assert rates.shape == (12, 12, 3)
    # 100 bytes per unit, every 300 seconds; flows are ingress-major
    assert rates[0, 0, 0] == pytest.approx(0)
    assert rates[0, 1, 0] == pytest.approx(1 * 100 / 300)
    assert rates[1, 0, 0] == pytest.approx(12 * 100 / 300)
    assert rates[11, 11, 2] == pytest.approx((143 + 2000) * 100 / 300)
    assert (rates >= 0).all()


def test_parse_a_single_epoch(tmp_path: Path) -> None:
    path = tmp_path / "week.gz"
    _write_week(path, epochs=1)
    assert _parse_abilene(path).shape == (12, 12, 1)


@pytest.mark.parametrize("content", ["1 2 3\n", "abc\n", ""])
def test_parse_rejects_other_files(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(content)
    with pytest.raises(TMgenException):
        _parse_abilene(path)


def test_parse_rejects_a_file_that_is_not_gzip(tmp_path: Path) -> None:
    path = tmp_path / "plain"
    path.write_text("1 2 3")
    with pytest.raises(TMgenException):
        _parse_abilene(path)


def test_fetch_keeps_a_file_with_the_right_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"traffic" * 1000)
    destination = tmp_path / "cache" / "copy.bin"
    _fetch(
        source.as_uri(), destination, hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert destination.read_bytes() == source.read_bytes()
    assert [p.name for p in destination.parent.iterdir()] == ["copy.bin"]


def test_fetch_discards_a_file_with_the_wrong_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"traffic")
    destination = tmp_path / "cache" / "copy.bin"
    with pytest.raises(TMgenException, match="checksum"):
        _fetch(source.as_uri(), destination, "0" * 64)
    assert list((tmp_path / "cache").iterdir()) == []


def test_fetch_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TMgenException, match="could not download"):
        _fetch((tmp_path / "missing").as_uri(), tmp_path / "copy", "0" * 64)
    assert not (tmp_path / "copy").exists()


def test_cached_weeks_are_joined_in_the_order_requested(tmp_path: Path) -> None:
    _write_week(tmp_path / "abilene-X03.gz", epochs=2)
    _write_week(tmp_path / "abilene-X01.gz", epochs=3)
    offline = (tmp_path / "none").as_uri()
    tm = load_abilene([3, 1], cache_dir=tmp_path, base_url=offline)
    assert tm.shape == (12, 12, 5)
    expected = (143 + numpy.array([0, 1000, 0, 1000, 2000])) * 100 / 300
    numpy.testing.assert_allclose(tm.matrix[11, 11], expected)


@pytest.mark.parametrize("weeks", [[], 0, 25, [1, 99]])
def test_load_abilene_rejects_unknown_weeks(tmp_path: Path, weeks: object) -> None:
    with pytest.raises(TMgenException):
        load_abilene(weeks, cache_dir=tmp_path)  # type: ignore[arg-type]


def test_load_abilene_reports_a_failed_download(tmp_path: Path) -> None:
    with pytest.raises(TMgenException, match="could not download"):
        load_abilene(
            1, cache_dir=tmp_path / "cache", base_url=(tmp_path / "none").as_uri()
        )


def test_data_dir_follows_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMGEN_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path
    monkeypatch.delenv("TMGEN_DATA_DIR")
    assert data_dir() == Path.home() / ".cache" / "tmgen"


@pytest.mark.skipif(
    not os.environ.get("TMGEN_NETWORK_TESTS"),
    reason="downloads 7 MB; set TMGEN_NETWORK_TESTS=1 to run",
)
def test_load_abilene_downloads_the_real_week(tmp_path: Path) -> None:
    tm = load_abilene(1, cache_dir=tmp_path)
    assert tm.shape == (12, 12, 2016)
    assert tm.matrix.sum(axis=(0, 1)).mean() > 1e8
