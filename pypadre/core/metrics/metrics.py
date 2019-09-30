
# Class to calculate some kind of metric on a component result or dataset
from abc import ABCMeta, abstractmethod

from pypadre.core.model.computation.computation import Computation


# Base class to hold the calculated metric
class Metric(Computation):

    COMPUTATION_ID = "computation_id"

    def __init__(self, *, name, computation, result, **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {}), **{self.COMPUTATION_ID: computation.id, "name": name}}

        super().__init__(component=computation.component, run=computation.run, result=result, metadata=metadata, **kwargs)
        self._name = name

    @property
    def name(self):
        return self.name


class MeasureMeter:
    __metaclass__ = ABCMeta

    @abstractmethod
    def compute(self, **kwargs) -> Metric:
        pass
