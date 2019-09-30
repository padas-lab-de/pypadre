# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from types import GeneratorType
from typing import Callable, Optional, Tuple, Iterable

from pypadre.core.base import MetadataEntity
from pypadre.core.model.code.code import Code
from pypadre.core.model.code.function import Function
from pypadre.core.model.computation.run import Run
from pypadre.core.model.computation.training import Training
from pypadre.core.model.computation.evaluation import Evaluation
from pypadre.core.model.execution import Execution
from pypadre.core.model.generic.i_model_mixins import IExecuteable
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.split.split import Split
from pypadre.core.model.split.splitter import Splitter


class PipelineComponent(MetadataEntity, IExecuteable):
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
        raise NotImplementedError

    def _execute(self, *, run: Run, data, **kwargs):
        kwargs["component"] = self
        results = self._execute_(data=data, run=run, **kwargs)
        if not isinstance(results, Computation):
            results = Computation(component=self, run=run, result=results)
        # TODO Trigger component result event for metrics and visualization
        return results

    @abstractmethod
    def _execute_(self, *, data, **kwargs):
        # Black box execution
        raise NotImplementedError

    # TODO Overwrite for no schema validation for now
    def validate(self, **kwargs):
        pass


class BranchingComponent(PipelineComponent):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _execute(self, *, run: Run, data, **kwargs):
        computation = super()._execute(run=run, data=data, **kwargs)
        if not isinstance(computation.result, GeneratorType) and not isinstance(computation.result, list):
            raise ValueError("Can only branch if the computation produces a list or generator of data")
        return computation


class PythonCodeComponent(PipelineComponent):

    def hash(self):
        return hash(self.code)

    def __init__(self, code: Optional[Code, Callable], **kwargs):
        if isinstance(code, Callable):
            code = Function(fn=code)
        super().__init__(**kwargs)
        self._code = code

    @property
    def code(self):
        return self._code

    def _execute_(self, *, data, **kwargs):
        return self.code.call(data=data, **kwargs)


# def _unpack_computation(cls, computation: Computation):
#     # TODO don't build all splits here. How to handle generators?
#     if isinstance(computation.result, GeneratorType):
#         computation.result = [_unpack_computation_(cls, tuple([i]) + result, computation) if isinstance(result, Tuple) else _unpack_computation_(cls, result, computation) for i, result in enumerate(computation.result)]
#     else:
#         computation.result = _unpack_computation_(cls, computation.result, computation)
#     return computation
#
#
# def _unpack_computation_(cls, result, computation):
#     if isinstance(result, Tuple):
#         return cls(*result, component=computation.component, execution=computation.execution)
#     elif isinstance(result, dict):
#         return cls(**result, component=computation.component, execution=computation.execution)
#     else:
#         return cls(result, component=computation.component, execution=computation.execution)


class SplitComponent(BranchingComponent, PipelineComponent):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, name="splitter", **kwargs):
        super().__init__(name=name, **kwargs)

    def _execute(self, *, data, **kwargs):
        return super()._execute(data=data, **kwargs)


class EstimatorComponent(PipelineComponent):

    @abstractmethod
    def __init__(self, name="estimator", **kwargs):
        super().__init__(name=name, **kwargs)

    # def _execute(self, *, data, **kwargs):
    #     return _unpack_computation(Training, super()._execute(data=data, **kwargs))


class EvaluatorComponent(PipelineComponent):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, name="evaluator", **kwargs):
        super().__init__(name=name, **kwargs)

    # def _execute(self, *, data, **kwargs):
    #     return _unpack_computation(Evaluation, super()._execute(data=data, **kwargs))


class SplitPythonComponent(SplitComponent, PythonCodeComponent):
    def __init__(self, *, code: Optional[Code, Callable], **kwargs):
        splitter = Splitter(code=code, **kwargs.get("splitter", {}))
        super().__init__(code=splitter.splits, **kwargs)


class EstimatorPythonComponent(EstimatorComponent, PythonCodeComponent):
    def __init__(self, *, code: Optional[Code, Callable], **kwargs):
        super().__init__(code=code, **kwargs)


class EvaluatorPythonComponent(EvaluatorComponent, PythonCodeComponent):
    def __init__(self, *, code: Optional[Code, Callable], **kwargs):
        super().__init__(code=code, **kwargs)
