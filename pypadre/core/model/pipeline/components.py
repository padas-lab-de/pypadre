# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Iterable, List

from pypadre.core.base import MetadataEntity
from pypadre.core.model.code.icode import Function
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.computation.run import Run
from pypadre.core.model.generic.custom_code import CustomCodeHolder, IProvidedCode
from pypadre.core.model.generic.i_executable_mixin import IExecuteable
from pypadre.core.model.pipeline.gridsearch import GridSearch
from pypadre.core.model.pipeline.parameters import IParameterProvider, ParameterMap
from pypadre.core.model.split.split import Split
from pypadre.core.model.split.splitter import Splitter
from pypadre.core.pickling.pickle_base import Pickleable
from pypadre.core.util.inheritance import SuperStop
from pypadre.core.util.utils import unpack
from pypadre.core.validation.validation import ValidateParameters


class IConsumer(SuperStop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def consumes(self) -> str:
        raise NotImplementedError()


class IProvider(SuperStop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def provides(self) -> List[str]:
        raise NotImplementedError()


class PipelineComponent(CustomCodeHolder, IConsumer, IProvider, IExecuteable, MetadataEntity):
    __metaclass__ = ABCMeta

    def __init__(self, *, name: str, metadata: Optional[dict] = None, **kwargs):
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

    def _execute_helper(self, *, run: Run, data,
                        predecessor: Computation = None, branch=False, **kwargs):

        # TODO find the problem in the loop
        results = self._execute_component_code(data=data, run=run, predecessor=predecessor, **kwargs)
        if not isinstance(results, Computation):
            results = Computation(component=self, run=run, predecessor=predecessor,
                                  branch=branch, result=results)

        results.send_put()
        # TODO Trigger component result event for metrics and visualization
        return results

    # @abstractmethod
    # def _execute_component_code(self, ctx, **kwargs):
    #     parameters = kwargs.pop("parameters", {})
    #     # kwargs are the padre context to be used
    #     return self._execute_component_code_help(ctx, **parameters)

    def _execute_component_code(self, **kwargs):
        return self.code.call(component=self, **kwargs)


class ParameterizedPipelineComponent(PipelineComponent, ValidateParameters, Pickleable):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parameter_schema: Iterable = None, parameter_provider: IParameterProvider = None, **kwargs):
        # TODO name via enum or name via owlready2
        # TODO implement parameter schema via owlready2 / mapping
        super().__init__(**kwargs)
        if parameter_provider is None:
            parameter_provider = GridSearch()
        self._parameter_schema = parameter_schema
        self._parameter_provider = parameter_provider

    def transient_fields(self):
        return ["_parameter_provider"]

    @property
    def parameter_provider(self):
        return self._parameter_provider

    def _execute_helper(self, *, run: Run, data, parameters=None, predecessor: Computation = None,
                        branch=False,
                        **kwargs):
        if parameters is None:
            parameters = {}
        self._validate_parameters(parameters)
        return super()._execute_helper(run=run, data=data, parameters=parameters, predecessor=predecessor,
                                       branch=branch, **kwargs)

    def combinations(self, *, run, predecessor, parameter_map: ParameterMap):
        combinations = self._parameter_provider.execute(run=run, component=self,
                                                             predecessor=predecessor, parameter_map=parameter_map)
        combinations.send_put()
        return combinations


class IProvidedComponent(IProvidedCode, PipelineComponent):

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def hash(self):
        return hash(self.__class__)

    def _execute_component_code(self, **kwargs):
        return self.call(component=self, **kwargs)


# class CodeComponent(CustomCodeHolder, PipelineComponent):
#
#     def hash(self):
#         return hash(self.code)
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def _execute_component_code(self, **kwargs):
#         return self.code.call(component=self, **kwargs)


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
    def __init__(self, name="splitter", code=None, **kwargs):
        if code is None:
            code = Splitter()
        if code is Callable:
            code = Function(fn=code)
        super().__init__(name=name, code=code, **kwargs)

    def _execute_helper(self, *, data, branch=True, **kwargs):
        return super()._execute_helper(data=data, branch=branch, **kwargs)


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


class CustomSplit(Function):

    def __init__(self, *, fn: Callable):
        def custom_split(ctx, **kwargs):

            def splitting_iterator():
                num = -1
                train_idx, test_idx, val_idx = fn(ctx, **kwargs)
                (data, run, component, predecessor) = unpack(ctx, "data", "run", "component", ("predecessor", None))
                yield Split(run=run, num=++num, train_idx=train_idx, test_idx=test_idx,
                            val_idx=val_idx, component=component, predecessor=predecessor)
            return splitting_iterator()
        super().__init__(fn=custom_split)
