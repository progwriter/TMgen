"""
Public traffic matrix datasets.

Datasets are downloaded from their original hosts on first use and cached on
disk; nothing is bundled with TMgen. The cache directory is
``$TMGEN_DATA_DIR`` if set, else ``~/.cache/tmgen``.
"""

import gzip
import hashlib
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy
from numpy.typing import NDArray

from tmgen.exceptions import TMgenException
from tmgen.tm import TrafficMatrix

logger = logging.getLogger(__name__)

ABILENE_URL = "https://www.cs.utexas.edu/~yzhang/research/AbileneTM"
"""Where the Abilene weeks are downloaded from (Yin Zhang's page)."""

ABILENE_NODES = (
    "ATLA-M5",
    "ATLAng",
    "CHINng",
    "DNVRng",
    "HSTNng",
    "IPLSng",
    "KSCYng",
    "LOSAng",
    "NYCMng",
    "SNVAng",
    "STTLng",
    "WASHng",
)
"""The 12 nodes of the Abilene matrices, in row and column order."""

ABILENE_WEEKS = {
    1: "2004-03-01",
    2: "2004-03-08",
    3: "2004-04-02",
    4: "2004-04-09",
    5: "2004-04-22",
    6: "2004-05-01",
    7: "2004-05-08",
    8: "2004-05-15",
    9: "2004-05-22",
    10: "2004-05-29",
    11: "2004-06-05",
    12: "2004-06-12",
    13: "2004-06-19",
    14: "2004-06-26",
    15: "2004-07-03",
    16: "2004-07-10",
    17: "2004-07-17",
    18: "2004-07-24",
    19: "2004-07-31",
    20: "2004-08-07",
    21: "2004-08-13",
    22: "2004-08-21",
    23: "2004-08-28",
    24: "2004-09-04",
}
"""The first day (a Monday) of each of the 24 weeks, 2004."""

_ABILENE_SHA256 = {
    1: "d996a9115d2815eec158d5055452deb389f5546a96e0dfcbfae5c22a40372558",
    2: "e4646c3020e5904b3e4f3388899c15940b8156ade9fdaf9dd96b06a83f8066a5",
    3: "b90f058def353a78db4e5d65d329de867659add90b7ef4d918acb7ba88a8f040",
    4: "3aab7fe56b84f2469888a142c2af313c865d4c3d41c955cdc199ccea4c66cce0",
    5: "e1cf3f3a79345ffde4e03a77a9c8017fe72fbc9584cb86f8428023efe8203f24",
    6: "770a8b0cb3468e212128954fc56d1cad365f3675a9e34da1e15f4d1ace9b43ff",
    7: "4b8a88af6b358562f356472ff1ae76d1b482333d4cde91666da0237a5c5ae56d",
    8: "5474cc2944c42cb826c6a0aa53da4d382e38bd7d07a5d21342bafcc85d6003b4",
    9: "005511d8b64c1fef598472026d23f7b454db19e5595955b3fbd9e0fe55a4dbbe",
    10: "c3517f34552a42e809f233a6352f9f04b9128ea98fdc319ede3672774967ecda",
    11: "cb382563cd2df8c05ce90ca18abf425862eb4260c614a64d32ff232dccd446f1",
    12: "13d45cc20fef1578c81c888939d719ca6b5cca1029e3f54d00781f561299442f",
    13: "b4f3c1d1861db52967c64a0a800c80775088e68bb66215a4a7c53578e574e9c7",
    14: "44b76c63febe1fee35efe35d5534fad72369e591945d2786e9bfa9d7ba9cef81",
    15: "17e3b2a4f8efb4a12868e25603c820635c6f3159b8a7cbac0781aafaac2198cc",
    16: "dc46c339f2f334b6c67f6219d395761590088e356c02807e0d52082f8019070e",
    17: "f0fd95217ed12b613f3e3a7d2134ed5a240823c7ef71b39e0e676c7891557cb0",
    18: "15ed6b8a9ea78ef22be8137e203142daa7660bfae481d096931456c4805795c0",
    19: "2ca01978bd973c0853ca0d2d716a7fd80b89470f269522fb58dfec68e42cfa65",
    20: "c7ef802f8e9b364f40fd010a9fa76a0022ad10fcaea7273c2b1c528a4c29d1a9",
    21: "0ff08972ccba812f42a16a9cab1f7333dd0228a61193c2ae28e86abdc0b470dc",
    22: "484b56d04b27c605c5eb6eea89c3983677f526a647fb889d816f91c5dcf4dbb4",
    23: "9fb081bf5f6320ddc337ec516839a329245f1550af8b4c161bbe12f15d7f5051",
    24: "f963c23f5446194f3dd37725638bd1523762e504da8d9baa3c8201200b682b92",
}
_ABILENE_EPOCHS_PER_WEEK = 12 * 24 * 7
_ABILENE_PAIRS = len(ABILENE_NODES) ** 2
_ABILENE_COLUMNS_PER_PAIR = 5
_ABILENE_BYTES_PER_UNIT = 100  # 100 is the packet sampling rate
_ABILENE_EPOCH_SECONDS = 300


def data_dir() -> Path:
    """The directory where downloaded datasets are cached."""
    configured = os.environ.get("TMGEN_DATA_DIR")
    return Path(configured) if configured else Path.home() / ".cache" / "tmgen"


def _fetch(url: str, destination: Path, sha256: str) -> None:
    """Download *url* to *destination*, keeping nothing unless the checksum matches."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name, delete=False
    ) as partial:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                for block in iter(lambda: response.read(1 << 20), b""):
                    digest.update(block)
                    partial.write(block)
        except (urllib.error.URLError, OSError) as error:
            Path(partial.name).unlink(missing_ok=True)
            raise TMgenException(f"could not download {url}: {error}") from error
    if digest.hexdigest() != sha256:
        Path(partial.name).unlink(missing_ok=True)
        raise TMgenException(
            f"{url} does not have the expected checksum (got {digest.hexdigest()}); "
            "the dataset may have changed or the download was cut short"
        )
    shutil.move(partial.name, destination)


def _parse_abilene(path: Path) -> NDArray[numpy.float64]:
    """Read one week file as bytes per second, shape (ingress, egress, epoch)."""
    columns = _ABILENE_PAIRS * _ABILENE_COLUMNS_PER_PAIR
    try:
        with gzip.open(path, "rt") as stream, warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="loadtxt: input contained no data"
            )
            table = numpy.loadtxt(
                stream,
                usecols=range(0, columns, _ABILENE_COLUMNS_PER_PAIR),
                ndmin=2,
            )
    except (OSError, ValueError) as error:
        raise TMgenException(f"{path} is not an Abilene week file: {error}") from error
    if table.shape[0] == 0 or table.shape[1] != _ABILENE_PAIRS:
        raise TMgenException(
            f"{path} does not have {_ABILENE_PAIRS} origin-destination flows"
        )
    rates = table * _ABILENE_BYTES_PER_UNIT / _ABILENE_EPOCH_SECONDS
    size = len(ABILENE_NODES)
    return numpy.ascontiguousarray(rates.reshape(-1, size, size).transpose(1, 2, 0))


def load_abilene(
    weeks: int | Sequence[int] = 1,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    base_url: str = ABILENE_URL,
) -> TrafficMatrix:
    """
    Load real traffic matrices of the Abilene backbone network (Internet2,
    2004), measured by Yin Zhang et al. from NetFlow with the destination AS
    mapped to egress routers.

    There are 24 weeks (see :py:data:`ABILENE_WEEKS`) of 2016 matrices of 12
    nodes (:py:data:`ABILENE_NODES`), one every 5 minutes. A week is
    downloaded (about 7 MB) on first use, checked against a checksum and
    cached. Traffic to the Abilene network itself is not included in the
    data, and the diagonal is the traffic that enters and leaves at the same
    node.

    :param weeks: the week number (1 to 24) or a sequence of them, joined in
        the order given
    :param cache_dir: where to keep downloads; default :py:func:`data_dir`
    :param base_url: where to download from
    :return: a new :py:class:`~tmgen.TrafficMatrix` in bytes per second with
        2016 epochs per week
    :raises TMgenException: for an unknown week, a failed download or a file
        that does not match its checksum
    """
    requested = [weeks] if isinstance(weeks, int) else list(weeks)
    if not requested:
        raise TMgenException("weeks must not be empty")
    for week in requested:
        if week not in _ABILENE_SHA256:
            raise TMgenException(f"weeks must be between 1 and 24, got {week!r}")
    directory = Path(cache_dir) if cache_dir is not None else data_dir()
    parts = []
    for week in requested:
        parsed = directory / f"abilene-X{week:02d}.npy"
        if parsed.exists():
            parts.append(numpy.load(parsed, allow_pickle=False))
            continue
        raw = directory / f"abilene-X{week:02d}.gz"
        if not raw.exists():
            logger.info("downloading Abilene week %d to %s", week, raw)
            _fetch(f"{base_url}/X{week:02d}.gz", raw, _ABILENE_SHA256[week])
        rates = _parse_abilene(raw)
        numpy.save(parsed, rates)
        parts.append(rates)
    return TrafficMatrix(numpy.concatenate(parts, axis=2))
