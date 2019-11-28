# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Iterable, List

from pypadre.core.base import MetadataMixin
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.computation.run import Run
from pypadre.core.model.generic.custom_code import CustomCodeHolder, CodeManagedMixin
from pypadre.core.model.generic.i_executable_mixin import ValidateableExecutableMixin
from pypadre.core.model.pipeline.components.component_interfaces import IConsumer, IProvider
from pypadre.core.model.pipeline.parameter_providers.gridsearch import default_parameter_provider
from pypadre.core.model.pipeline.parameter_providers.parameters import ParameterProviderMixin, ParameterMap
from pypadre.core.util.utils import persistent_hash
from pypadre.core.validation.validation import ValidateParameters


class PipelineComponentMixin(CodeManagedMixin, CustomCodeHolder, IConsumer, IProvider, ValidateableExecutableMixin, MetadataMixin):
    """
    This class is a component of the pipeline. It can consume and provide data as well as validate it's metadata.
    PipelineComponents can be represented by custom code.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, name: str, metadata: Optional[dict] = None, **kwargs):
        if metadata is None:
            metadata = {}

        if name is None:
            name = self.__class__.__name__

        metadata = {**metadata, **{"name": name}}
        # TODO name via enum or name via owlready2
        # TODO validation model?
        super().__init__(metadata=metadata, **kwargs)

        self.metadata["id"] = persistent_hash((self.name, self.code.id, self.reference.id))

    def _execute_helper(self, *, run: Run, data,
                        predecessor: Computation = None, branch=False, intermediate_results=True, store_results=False, **kwargs):

        # TODO find the problem in the loop
        results = self._execute_component_code(data=data, run=run, predecessor=predecessor, **kwargs)
        if not isinstance(results, Computation):
            results = Computation(component=self, run=run, predecessor=predecessor,
                                  branch=branch, result=results, parameters=kwargs.get("parameters", {}),
                                  initial_hyperparameters=kwargs.get('initial_hyperparameters', {}))

        if intermediate_results:
            results.send_put(store_results=store_results)
        # TODO Trigger component result event for metrics and visualization
        return results

    def _execute_component_code(self, **kwargs):
        return self.code.call(component=self, **kwargs)


class ParameterizedPipelineComponentMixin(PipelineComponentMixin, ValidateParameters):
    """
    This is an extension of the PipelineComponentMixin allowing for parameters and their validation.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parameter_schema: Iterable = None, parameter_provider: ParameterProviderMixin = None, **kwargs):
        # TODO name via enum or name via owlready2
        # TODO implement parameter schema via owlready2 / mapping
        super().__init__(**kwargs)
        if parameter_provider is None:
            parameter_provider = default_parameter_provider
        self._parameter_schema = parameter_schema
        self._parameter_provider = parameter_provider

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


class SplitComponentMixin(PipelineComponentMixin):
    """ This component is used to generate splits from a dataset. """

    @abstractmethod
    def __init__(self, *, name="splitter", **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def consumes(self) -> str:
        # TODO types from rdf
        return "dataset"

    @property
    def provides(self) -> List[str]:
        # TODO types from rdf
        return ["split"]

    def _execute_helper(self, *, data, branch=True, **kwargs):
        return super()._execute_helper(data=data, branch=branch, **kwargs)


class EstimatorComponentMixin(PipelineComponentMixin):
    """ This component is used to train a model. """

    @abstractmethod
    def __init__(self, name="estimator", **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def consumes(self) -> str:
        # TODO types from rdf
        return "split"

    @property
    def provides(self) -> List[str]:
        # TODO types from rdf
        return ["model"]


class EvaluatorComponentMixin(PipelineComponentMixin):
    """ This component is used to fit a model. """
    __metaclass__ = ABCMeta
    TRUTH = "truth"
    PREDICTED = "predicted"
    PROBABILITIES = "probabilities"
    PREDICTIONS = "predictions"

    @abstractmethod
    def __init__(self, name="evaluator", **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def consumes(self) -> str:
        # TODO types from rdf
        return "model"

    @property
    def provides(self) -> List[str]:
        # TODO types from rdf
        return ["classification", "regression"]
