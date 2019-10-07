# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Union, Iterable

from pypadre.core.base import MetadataEntity
from pypadre.core.model.code.code import Code
from pypadre.core.model.code.function import Function
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.execution import Execution
from pypadre.core.model.generic.i_executable_mixin import IExecuteable
from pypadre.core.model.split.splitter import Splitter


class PipelineComponent(MetadataEntity, IExecuteable):
    __metaclass__ = ABCMeta

    def __init__(self, *, name: str, metadata: Optional[dict]=None, **kwargs):
        if metadata is None:
            metadata = {}
        # TODO name via enum or name via owlready2
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

    def _execute(self, *, execution: Execution, data, parameters, predecessor: Computation=None, branch=False, **kwargs):
        kwargs["component"] = self
        results = self._execute_(data=data, execution=execution, predecessor=predecessor, parameters=parameters, **kwargs)
        if not isinstance(results, Computation):
            results = Computation(component=self, parameters=parameters, execution=execution, predecessor=predecessor, branch=branch, result=results)
        # TODO Trigger component result event for metrics and visualization
        return results

    @abstractmethod
    def _execute_(self, *, data, parameters, **kwargs):
        # Black box execution
        raise NotImplementedError

    # TODO Overwrite for no schema validation for now
    def validate(self, **kwargs):
        pass


class ParameterizedPipelineComponent(PipelineComponent):
    __metaclass__ = ABCMeta

    def __init__(self, *, parameters: Iterable, **kwargs):
        # TODO name via enum or name via owlready2
        super().__init__(**kwargs)
        self._parameters = parameters


# class BranchingComponent(PipelineComponent):
#     __metaclass__ = ABCMeta
#
#     @abstractmethod
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def _execute(self, *, run: Run, data, **kwargs):
#         computation = super()._execute(execution=execution, data=data, **kwargs)
#         if not isinstance(computation.result, GeneratorType) and not isinstance(computation.result, list):
#             raise ValueError("Can only branch if the computation produces a list or generator of data")
#         return computation
#
#
# class GridSearchComponent(BranchingComponent):
#
#     def __init__(self, for_component: PipelineComponent, **kwargs):
#         super().__init__(**kwargs)
#         self._for_component = for_component
#
#     @property
#     def for_component(self):
#         return self._for_component
#
#     def hash(self):
#         self.__hash__()
#
#     def _execute_(self, *, data, parameters, **kwargs):
#         # TODO check for parameter type to decide if this is a iterable or not and start doing a grid search here
#         # TODO Get all combinations generator
#         # TODO how do we aggregate the data afterwards? Maybe a run looks for all it's results in different Grid
#         return data


class PythonCodeComponent(PipelineComponent):

    def hash(self):
        return hash(self.code)

    def __init__(self, code: Union[Code, Callable], **kwargs):
        if isinstance(code, Callable):
            code = Function(fn=code)
        super().__init__(**kwargs)
        self._code = code

    @property
    def code(self):
        return self._code

    def _execute_(self, *, data, parameters, **kwargs):
        return self.code.call(data=data, parameters=parameters, **kwargs)


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


# class SplitComponent(BranchingComponent, PipelineComponent):
class SplitComponent(PipelineComponent):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, name="splitter", **kwargs):
        super().__init__(name=name, **kwargs)

    def _execute(self, *, data, branch=True, **kwargs):
        return super()._execute(data=data, branch=branch, **kwargs)


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
    def __init__(self, *, code: Union[Code, Callable], **kwargs):
        splitter = Splitter(code=code, **kwargs.get("splitter", {}))
        super().__init__(code=splitter.splits, **kwargs)


class EstimatorPythonComponent(EstimatorComponent, PythonCodeComponent):
    def __init__(self, *, code: Union[Code, Callable], **kwargs):
        super().__init__(code=code, **kwargs)


class EvaluatorPythonComponent(EvaluatorComponent, PythonCodeComponent):
    def __init__(self, *, code: Union[Code, Callable], **kwargs):
        super().__init__(code=code, **kwargs)
