
# Class to calculate some kind of metric on a component result or dataset
from abc import ABCMeta, abstractmethod

from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.generic.custom_code import ICodeManagedObject, ICustomCodeSupport
from pypadre.core.model.generic.i_executable_mixin import IExecuteable
from pypadre.core.model.generic.i_model_mixins import ILoggable


class Metric(Computation):
    """
    Base class to hold the calculated metric
    """

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


class MeasureMeter(ICodeManagedObject, ICustomCodeSupport, IExecuteable, ILoggable):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _execute_helper(self, *args, computation: Computation, **kwargs) -> Metric:
        raise NotImplementedError()
