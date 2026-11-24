import importlib as _importlib
from importlib import metadata as _metadata
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

from tmgen import datasets, hmc, lowrank, models
from tmgen.exceptions import TMgenException
from tmgen.tm import TrafficMatrix

if _TYPE_CHECKING:
    from tmgen import plot

__version__ = _metadata.version("tmgen")

__all__ = [
    "TMgenException",
    "TrafficMatrix",
    "__version__",
    "datasets",
    "hmc",
    "lowrank",
    "models",
    "plot",
]


def __dir__() -> list[str]:
    return sorted(__all__)


def __getattr__(name: str) -> _Any:
    # plotting needs the optional matplotlib dependency, so import it on use
    if name == "plot":
        return _importlib.import_module("tmgen.plot")
    raise AttributeError(f"module 'tmgen' has no attribute {name!r}")
