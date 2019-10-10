from typing import Callable, Optional, Union

import networkx
from networkx import DiGraph, is_directed_acyclic_graph

from pypadre.core.model.code.code import Code
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.computation.hyper_parameter_search import HyperParameterSearch
from pypadre.core.model.computation.run import Run
from pypadre.core.model.execution import Execution
from pypadre.core.model.generic.i_model_mixins import IStoreable, IProgressable
from pypadre.core.model.generic.i_executable_mixin import IExecuteable
from pypadre.core.model.pipeline.components import PythonCodeComponent, SplitPythonComponent, \
    EstimatorPythonComponent, EstimatorComponent, EvaluatorComponent, PipelineComponent
from pypadre.core.model.pipeline.parameters import ParameterMap
from pypadre.core.model.split.split import Split
from pypadre.core.validation.validation import Validateable


class Pipeline(IStoreable, IProgressable, IExecuteable, DiGraph, Validateable):
    def __init__(self, **attr):
        super().__init__(**attr)

    def hash(self):
        # TODO this has may have to include if the pipeline structure was changed etc
        return hash(",".join([str(pc.hash()) for pc in self.nodes]))

    def _execute(self, *, pipeline_parameters: Union[ParameterMap, dict]=None, parameter_map: ParameterMap=None, execution: Execution, data, **kwargs):
        if parameter_map is None:
            if pipeline_parameters is None:
                parameter_map = ParameterMap({})
            if not isinstance(pipeline_parameters, ParameterMap):
                parameter_map = ParameterMap(pipeline_parameters)

        # TODO currently we don't allow for merging in a pipeline again. To solve this a successor can only execute as soon as it gets all data from all predecessors (Computation pipelines etc...)
        # TODO each component should maybe have a own kwargs list for the execute call to allow for the same parameter name on different components

        # validate the current state
        self.validate()

        entries = self.get_entries()

        for entry in entries:
            self._execute_(entry, parameter_map=parameter_map, execution=execution, data=data, **kwargs)

    def _execute_(self, node: PipelineComponent, *, data, parameter_map: ParameterMap, execution: Execution, **kwargs):
        # TODO do some more sophisticated result analysis in the grid search
        # Grid search if we have multiple combinations
        parameters = parameter_map.combinations(execution=execution, component=node, predecessor=kwargs.get("predecessor", None))

        if isinstance(parameters, HyperParameterSearch):
            if parameters.branch:
                for parameters in parameters.result:
                    # If the parameter map returns a generator or other iterable and should branch we have to execute
                    #  for each item
                    self._execute__(node, data=data, parameters=parameters, parameter_map=parameter_map, execution=execution, predecessor=kwargs.get("predecessor", None))
            else:
                # If the parameter map returns a search with a single item without branch we can just use it
                self._execute__(node, data=data, parameters=parameters.result, parameter_map=parameter_map, execution=execution, predecessor=kwargs.get("predecessor", None))
        else:
            # Todo don't force the user to provide a hyper parameter search in a parameter_map?
            raise NotImplementedError("A hyper parameter search has to be returned by the parameter_map")
            #self._execute__(node, data=data, parameters=parameters, parameter_map=parameter_map, execution=execution)

    def _execute__(self, node: PipelineComponent, *, data, parameters, parameter_map: ParameterMap, execution: Execution, **kwargs):
        computation = node.execute(execution=execution, parameters=parameters, data=data,
                                   predecessor=kwargs.pop("predecessor", None), **kwargs)
        # TODO add metric calculation
        if computation.branch:
            for res in computation.result:
                self._execute_successors(node, execution=execution, predecessor=computation,
                                         parameter_map=parameter_map, data=res)
        else:
            self._execute_successors(node, execution=execution, predecessor=computation, parameter_map=parameter_map,
                                     data=computation.result)

        # Check if we are a end node
        if self.out_degree(node) == 0:
            print("we are at the end of the pipeline / store results?")
            # TODO we are at the end of the pipeline / store results?
            run = Run.from_computation(computation)
            run.send_put()

    def _execute_successors(self, node: PipelineComponent, *, data, parameter_map: ParameterMap, execution: Execution, predecessor: Computation=None, **kwargs):
        successors = self.successors(node)
        for successor in successors:
            self._execute_(successor, data=data, execution=execution, predecessor=predecessor, parameter_map=parameter_map, **kwargs)

    def is_acyclic(self):
        return is_directed_acyclic_graph(self)

    # Overwrite for no schema validation for now
    def validate(self, **kwargs):
        if not self.is_acyclic():
            raise ValueError("Pipelines need to be acyclic")

    def get_exits(self):
        return [node for node, out_degree in self.out_degree() if out_degree == 0]

    def get_entries(self):
        return [node for node, out_degree in self.in_degree() if out_degree == 0]


class DefaultPythonExperimentPipeline(Pipeline):

    # TODO add source entity instead of callable (if only callable is given how to persist?)
    def __init__(self, *, preprocessing_fn: Optional[Union[Code, Callable]] = None,
                 splitting: Optional[Union[Code, Callable]],
                 estimator: Union[Callable, EstimatorComponent],
                 evaluator: Union[Callable, EvaluatorComponent], **attr):
        super().__init__(**attr)
        if not networkx.is_empty(self):
            # TODO attrs could include some network initialization for the components
            raise NotImplementedError("Preinitializing a pipeline is not implemented.")

        self._preprocessor = PythonCodeComponent(name="preprocessor", code=preprocessing_fn,
                                                 **attr) if preprocessing_fn else None

        self._splitter = SplitPythonComponent(code=splitting, predecessors=self._preprocessor, **attr)
        self.add_node(self._splitter)

        self._estimator = EstimatorPythonComponent(code=estimator.execute if isinstance(estimator,
                                                                                        EstimatorComponent) else estimator,
                                                   predecessors=[self._splitter], **attr)
        self.add_node(self._estimator)

        self._evaluator = PythonCodeComponent(name="evaluator",
                                              code=evaluator.execute if isinstance(evaluator,
                                                                                   EvaluatorComponent) else evaluator,
                                              predecessors=[self._estimator], **attr)
        self.add_node(self._evaluator)

        # Build pipeline grap
        if self._preprocessor:
            self.add_edge(self._preprocessor, self._splitter)
        self.add_edge(self._splitter, self._estimator)
        self.add_edge(self._estimator, self._evaluator)

    @property
    def preprocessor(self):
        return self._preprocessor

    @property
    def splitter(self):
        return self._splitter

    @property
    def estimator(self):
        return self._estimator

    # @property
    # def evaluator(self):
    #     return self._evaluator
