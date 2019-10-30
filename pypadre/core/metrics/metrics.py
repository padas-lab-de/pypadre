
# Class to calculate some kind of metric on a component result or dataset
from abc import ABCMeta, abstractmethod
from typing import List

from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.generic.custom_code import ICodeManagedObject, CustomCodeHolder
from pypadre.core.model.generic.i_executable_mixin import IExecuteable
from pypadre.core.model.generic.i_model_mixins import ILoggable
from pypadre.core.model.pipeline.components import IConsumer, IProvider


class Metric(Computation):
    """
    Base class to hold the calculated metric
    """

    COMPUTATION_ID = "computation_id"
    RUN_ID = "run_id"

    def __init__(self, *, name, computation, result, **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {}), **{self.COMPUTATION_ID: computation.id, self.RUN_ID: computation.run.id, "name": name}}

        super().__init__(component=computation.component, run=computation.run, result=result, metadata=metadata, **kwargs)
        self._name = name

    @property
    def name(self):
        return self._name


class IMetricProvider(ICodeManagedObject, IConsumer, IProvider, CustomCodeHolder, IExecuteable, ILoggable):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def provides(self) -> List[str]:
        # TODO link with owl semantic
        return [str(self.__class__)]
