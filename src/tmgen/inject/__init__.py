import os
from abc import ABC, abstractmethod

from tmgen import TrafficMatrix


class InjectorBase(ABC):
    def __init__(
        self,
        tm_fname: str | os.PathLike[str],
        epoch_length: float,
        ID: int,
        destinations: dict[str, list[str]],
    ) -> None:
        """
        Create a new traffic injector

        :param tm_fname: the traffic matrix filename
        :param epoch_length: length of a single epoch, in seconds
        :param ID: the integer ID of this injector
            (to determine correct flow volumes)
        :param destinations: a dictionary mapping integer node IDs to IP
            addresses (as strings)
        """
        self.epoch_length = epoch_length
        self.ID = ID
        self.destinations = destinations
        self._load_tm(tm_fname)

    def _load_tm(self, tm_fname: str | os.PathLike[str]) -> None:
        """
        Load the traffic matrix from disk
        :param tm_fname: the TM file name (a file written by
            :py:meth:`TrafficMatrix.save`)
        """
        self.tm = TrafficMatrix.load(tm_fname)

    @abstractmethod
    def run(self) -> None:
        pass
