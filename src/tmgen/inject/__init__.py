# coding=utf-8
from abc import abstractmethod, ABCMeta

from tmgen import TrafficMatrix


class InjectorBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, tm_fname, epoch_length, ID, destinations):
        """
        Create a new traffic injector

        :param tm_fname: the traffic matrix filename
        :param epoch_length: length of a single epoch, in seconds
        :param ID: the integer ID of this injector
            (to determine correct flow volumes)
        :param destinations: a dictionary mapping integer node IDs to IP addresses (as strings)
        """
        self.epoch_length = epoch_length
        self.ID = ID
        self.destinations = destinations
        self._load_tm(tm_fname)

    def _load_tm(self, tm_fname):
        """
        Load the traffic matrix from disk
        :param tm_fname: the TM fileame (file must be in pickle format)
        """
        self.tm = TrafficMatrix.from_pickle(tm_fname)

    @abstractmethod
    def run(self):
        pass
