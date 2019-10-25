from typing import Callable, Optional, Union, Type

import networkx
from networkx import DiGraph, is_directed_acyclic_graph

from pypadre.core.metrics.MetricRegistry import metric_registry
from pypadre.core.model.code.icode import ICode
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.computation.pipeline_output import PipelineOutput
from pypadre.core.model.computation.run import Run
from pypadre.core.model.generic.i_executable_mixin import IExecuteable
from pypadre.core.model.generic.i_model_mixins import IProgressable
from pypadre.core.model.pipeline.components import EstimatorComponent, EvaluatorComponent, PipelineComponent, \
    ParameterizedPipelineComponent, SplitComponent
from pypadre.core.model.pipeline.parameters import ParameterMap
from pypadre.core.validation.validation import Validateable


class Pipeline(IProgressable, IExecuteable, DiGraph, Validateable):
    def __init__(self, allow_metrics = True, **attr):
        self._allow_metrics = allow_metrics
        super().__init__(**attr)

    def hash(self):
        # TODO this has may have to include if the pipeline structure was changed etc
        return hash(",".join([str(pc.hash()) for pc in self.nodes]))

    def get_component(self, id):
        # TODO make this defensive
        for node in self.nodes:
            if node.id == id:
                return node
        return None

    def _execute_helper(self, *, pipeline_parameters: Union[ParameterMap, dict] = None,
                        parameter_map: ParameterMap = None, run: Run, data, **kwargs):
        if parameter_map is None:
            if pipeline_parameters is None:
                parameter_map = ParameterMap({})
            if not isinstance(pipeline_parameters, ParameterMap):
                parameter_map = ParameterMap(pipeline_parameters)

        # TODO currently we don't allow for merging in a pipeline again.
        #  To solve this a successor can only execute as soon as it gets all data from all predecessors
        #  (Computation pipelines etc...)
        # TODO each component should maybe have a own kwargs list for the execute call
        #  to allow for the same parameter name on different components

        # validate the current state
        self.validate()

        entries = self.get_entries()

        for entry in entries:
            self._execute_pipeline(entry, parameter_map=parameter_map, run=run, data=data, **kwargs)

    def _execute_pipeline(self, node: PipelineComponent, *, data, parameter_map: ParameterMap, run: Run,
                          **kwargs):
        # TODO do some more sophisticated result analysis in the grid search
        # Grid search if we have multiple combinations

        if isinstance(node, ParameterizedPipelineComponent):
            # extract all combinations of parameters we have to execute
            parameter_grid = node.combinations(run=run, predecessor=kwargs.get("predecessor", None),
                                               parameter_map=parameter_map)

            # branch if we have multiple parameter settings
            for parameters in parameter_grid.iter_result():
                # If the parameter map returns a generator or other iterable and should branch we have to
                # execute for each item
                self._execute_pipeline_helper(node, data=data, parameters=parameters,
                                              parameter_map=parameter_map, run=run,
                                              predecessor=kwargs.get("predecessor", None))
        else:
            # If we don't need parameters we don't extract them from the map but only pass the map to the following
            # components
            self._execute_pipeline_helper(node, data=data, parameter_map=parameter_map, run=run,
                                          predecessor=kwargs.get("predecessor", None))

    def _execute_pipeline_helper(self, node: PipelineComponent, *, data, parameter_map: ParameterMap,
                                 run: Run, **kwargs):
        computation = node.execute(run=run, data=data,
                                   predecessor=kwargs.pop("predecessor", None), **kwargs)

        self.send_log(message="Following metrics would be available for " + str(computation) + ": " + ', '.join(str(p) for p in metric_registry.available_providers(computation)))

        # calculate measures
        if self.allow_metrics:
            metrics = metric_registry.calculate_measures(computation, run=run, node=node, **kwargs)
            for metric in metrics:
                metric.send_put()

        # branch results now if needed (for example for splits)
        for res in computation.iter_result():
            self._execute_successors(node, run=run, predecessor=computation,
                                     parameter_map=parameter_map, data=res)

        # Check if we are a end node
        if self.out_degree(node) == 0:
            print("we are at the end of the pipeline / store results?")
            # TODO we are at the end of the pipeline / store results?
            output = PipelineOutput.from_computation(computation)
            output.send_put()

    def _execute_successors(self, node: PipelineComponent, *, data, parameter_map: ParameterMap, run: Run,
                            predecessor: Computation = None, **kwargs):
        successors = self.successors(node)
        for successor in successors:
            self._execute_pipeline(successor, data=data, run=run, predecessor=predecessor,
                                   parameter_map=parameter_map, **kwargs)

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

    def __getstate__(self):
        state = dict(self.__dict__)
        return state

    @property
    def allow_metrics(self):
        return self._allow_metrics


class DefaultPythonExperimentPipeline(Pipeline):

    # TODO add source entity instead of callable (if only callable is given how to persist?)
    def __init__(self, *, preprocessing_fn: Optional[Union[ICode, Callable]] = None,
                 splitting: Optional[Union[Type[ICode], Callable]],
                 estimator: Union[Callable, EstimatorComponent],
                 evaluator: Union[Callable, EvaluatorComponent], **attr):
        super().__init__(**attr)
        if not networkx.is_empty(self):
            # TODO attrs could include some network initialization for the components
            raise NotImplementedError("Preinitializing a pipeline is not implemented.")

        self._preprocessor = PipelineComponent(name="preprocessor", code=preprocessing_fn,
                                           **attr) if preprocessing_fn else None

        self._splitter = SplitComponent(code=splitting, predecessors=self._preprocessor, **attr)
        self.add_node(self._splitter)

        self._estimator = estimator if isinstance(estimator, EstimatorComponent) else EstimatorComponent(
            code=estimator, predecessors=[self._splitter], **attr)

        self.add_node(self._estimator)

        self._evaluator = evaluator if isinstance(evaluator, EvaluatorComponent) else EvaluatorComponent(
            code=evaluator, predecessors=[self._estimator], **attr)
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

    @property
    def evaluator(self):
        return self._evaluator
