
# Class to calculate some kind of metric on a component result or dataset
from abc import ABCMeta


class MeasureMeter(object):
    __metaclass__ = ABCMeta

    def compute(self, **kwargs):
        pass


# Base class to hold the calculated metric
class Metric(object):
    pass

