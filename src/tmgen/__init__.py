import importlib
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from tmgen import hmc, models
from tmgen.exceptions import TMgenException
from tmgen.tm import TrafficMatrix

if TYPE_CHECKING:
    from tmgen import plot

__version__ = version("tmgen")

__all__ = [
    "TMgenException",
    "TrafficMatrix",
    "__version__",
    "hmc",
    "models",
    "plot",
]


def __getattr__(name: str) -> Any:
    # plotting needs the optional matplotlib dependency, so import it on use
    if name == "plot":
        return importlib.import_module("tmgen.plot")
    raise AttributeError(f"module 'tmgen' has no attribute {name!r}")
