
# Class to calculate some kind of metric on a component result or dataset
from abc import ABCMeta, abstractmethod


class MeasureMeter:
    __metaclass__ = ABCMeta

    @abstractmethod
    def compute(self, **kwargs):
        pass


# Base class to hold the calculated metric
class Metric:

    def __init__(self, name, data):
        self._name = name
        self._data = data

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self.name
