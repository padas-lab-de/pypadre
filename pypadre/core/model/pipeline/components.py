# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from types import GeneratorType
from typing import List, Callable, Optional

from pypadre.core.base import MetadataEntity
from pypadre.core.model.execution import Execution
from pypadre.core.model.pipeline.computation import Computation
from pypadre.core.model.split.split import Split, Splitter


class PipelineComponent(MetadataEntity):
    __metaclass__ = ABCMeta

    def __init__(self, *, name: str, metadata: Optional[dict]=None, **kwargs):
        if metadata is None:
            metadata = {}
        # TODO name via enum or removal and name via owlread2
        super().__init__(metadata=metadata, **kwargs)
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def hash(self):
        """
        This method should return a hash used to reference the version of the current code or if code was changed
        :return:
        """
        pass

    def execute(self, *, execution: Execution, data, **kwargs):
        # Trigger event started
        results = self._execute(data=data, **kwargs)
        computation = Computation(component=self, execution=execution, result=results)
        # Trigger component result event for metrics and visualization
        # Trigger event finished
        return computation

    @abstractmethod
    def _execute(self, *, data, **kwargs):
        # Black box execution
        pass

    # TODO Overwrite for no schema validation for now
    def validate(self, **kwargs):
        pass


class BranchingComponent(PipelineComponent):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self, *, execution: Execution, data, **kwargs):
        computation = super().execute(execution=execution, data=data, **kwargs)
        if not isinstance(computation.result, GeneratorType) and not isinstance(computation.result, list):
            raise ValueError("Can only branch if the computation produces a list or generator of data")
        return computation


class PythonCodeComponent(PipelineComponent):

    def hash(self):
        return hash(self.code)

    def __init__(self, code: Callable, **kwargs):
        super().__init__(**kwargs)
        self._code = code

    @property
    def code(self):
        return self._code

    def _execute(self, *, data, **kwargs):
        # TODO maybe data is itself just another kwargs?
        return self.code(data=data, **kwargs)


class SplitComponent(BranchingComponent, PipelineComponent):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(name="splitter", **kwargs)

    def execute(self, *, data, **kwargs):
        # TODO create run?
        computation = super().execute(data=data, **kwargs)
        # TODO do we need the enumerate? @See EstimatorComponent data tuple
        computation.result = enumerate(computation.result)
        return computation


class EstimatorComponent(PipelineComponent):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(name="estimator", **kwargs)
        # TODO check if code is

    def execute(self, *, data, **kwargs):
        # TODO check data format
        split, (train_idx, val_idx, test_idx) = data
        # TODO create split and reference run?
        # Split(self, split, train_idx, val_idx, test_idx)
        return super().execute(data=data, **kwargs)


class SplitPythonComponent(SplitComponent, PythonCodeComponent):
    def __init__(self, *, code: Callable, **kwargs):
        # TODO splitter shouldn't hold a dataset but receive data in the splits function
        splitter = Splitter(fn=code, **kwargs.get("splitter", {}))
        super().__init__(code=splitter.splits, **kwargs)


class EstimatorPythonComponent(EstimatorComponent, PythonCodeComponent):
    def __init__(self, *, code: Callable, **kwargs):
        # TODO make pipeline instead
        super().__init__(code=code, **kwargs)
