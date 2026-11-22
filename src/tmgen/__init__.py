from importlib.metadata import version

from tmgen import models, tm

from .tm import TrafficMatrix

__version__ = version("tmgen")

__all__ = ["TrafficMatrix", "__version__", "models", "plot", "tm"]
