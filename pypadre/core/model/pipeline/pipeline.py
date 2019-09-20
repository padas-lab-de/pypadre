from abc import abstractmethod
from typing import Callable, List, Optional, Union

import networkx
from networkx import DiGraph, is_directed_acyclic_graph

from pypadre.core.model.execution import Execution
from pypadre.core.model.pipeline.components import PythonCodeComponent, BranchingComponent, PipelineComponent, \
    SplitPythonComponent, EstimatorPythonComponent
from pypadre.core.validation.validation import Validateable


# TODO wrapper for sklearn pipelines and estimators should be linked with owlready2, add own workflow / pipeline definition?
class Pipeline(DiGraph, Validateable):
    def __init__(self, **attr):
        super().__init__(**attr)

    def hash(self):
        # TODO this has may have to include if the pipeline structure was changed etc
        return hash(",".join([str(pc.hash()) for pc in self.nodes]))

    def execute(self, *, execution: Execution, data, **kwargs):
        # TODO currently we don't allow for merging in a pipeline again. To solve this a successor can only execute as soon as it gets all data from all predecessors (Computation pipelines etc...)
        # TODO each component should maybe have a own kwargs list for the execute call to allow for the same parameter name on different components
        # validate the current state
        self.validate()

        entries = self.get_entries()

        for entry in entries:
            self._execute(entry, execution=execution, data=data, **kwargs)

    def _execute(self, node, *, data, execution: Execution, **kwargs):
        computation = node.execute(execution=execution, data=data, **kwargs)
        if isinstance(node, BranchingComponent):
            for res in computation.result:
                self._execute_successors(node, execution=execution, data=res, **kwargs)
        else:
            self._execute_successors(node, execution=execution, data=computation.result, **kwargs)

    def _execute_successors(self, node, *, data, execution: Execution, **kwargs):
        successors = self.successors(node)
        for successor in successors:
            self._execute(successor, execution=execution, data=data, **kwargs)

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


class Estimator:
    @abstractmethod
    def _fit(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _infer(self, *args, **kwargs):
        raise NotImplementedError()

    def fit_infer(self, *args, **kwargs):
        self._fit(self, *args, **kwargs)
        return self._infer(self, *args, **kwargs)


# class Evaluator:
#     @abstractmethod
#     def infer(self, *args, **kwargs):
#         raise NotImplementedError()


class DefaultPythonExperimentPipeline(Pipeline):

    def __init__(self, *, preproccessing_fn: Optional[Callable] = None, splitting: Callable, estimator: Union[Callable, Estimator], **attr):
        super().__init__(**attr)
        if not networkx.is_empty(self):
            # TODO attrs could include some network initialization for the components
            raise NotImplementedError("Preinitializing a pipeline is not implemented.")

        self._preprocessor = PythonCodeComponent(name="preprocessor", code=preproccessing_fn,
                                                 **attr) if preproccessing_fn else None

        self._splitter = SplitPythonComponent(code=splitting, predecessors=self._preprocessor, **attr)
        self.add_node(self._splitter)

        self._estimator = EstimatorPythonComponent(code=estimator.fit_infer if isinstance(estimator, Estimator) else estimator, predecessors=[self._splitter],
                                                   **attr)
        self.add_node(self._estimator)

        # TODO evaluator?
        # self._evaluator = PythonCodeComponent(name="evaluator", code=evaluator.infer if isinstance(evaluator, Evaluator) else evaluator, predecessors=[self._estimator],
        #                                       **attr)
        # self.add_node(self._evaluator)

        # Build pipeline grap
        if self._preprocessor:
            self.add_edge(self._preprocessor, self._splitter)
        self.add_edge(self._splitter, self._estimator)

        # TODO evaluator?
        #self.add_edge(self._estimator, self._evaluator)

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
