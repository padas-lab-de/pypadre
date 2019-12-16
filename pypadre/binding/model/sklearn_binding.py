from collections import Callable
from typing import Union, Type, Optional

from sklearn.pipeline import Pipeline

from pypadre.binding.model.sklearn_estimator import SKLearnEstimator
from pypadre.binding.model.sklearn_evaluator import SKLearnEvaluator
from pypadre.core.model.code.code_mixin import CodeMixin
from pypadre.core.model.pipeline import pipeline
from pypadre.core.model.pipeline.components.component_mixins import EstimatorComponentMixin
from pypadre.core.model.pipeline.components.components import PipelineComponent, DefaultSplitComponent, SplitComponent


class SKLearnPipeline(pipeline.DefaultPythonExperimentPipeline):

    def __init__(self, *, splitting: Optional[Union[Type[CodeMixin], Callable]] = None, parameter_provider=None,
                 pipeline_fn: Callable, **kwargs):
        """

        :param splitting:
        :param pipeline_fn: A function that returns a Sklearn pipeline as the return value
        :param kwargs:
        """
        pipeline = pipeline_fn()
        # visitor = SciKitVisitor(pipeline)
        # TODO use visitor to extract parameter schema from pipeline

        # Check if the return type is a Sklearn Pipeline
        assert (isinstance(pipeline, Pipeline))

        # Verify running two instances of the function creates two Pipeline objects
        assert (pipeline is not pipeline_fn())

        # TODO provider for a specific node
        sk_learn_estimator = SKLearnEstimator(pipeline=pipeline, parameter_provider=parameter_provider,
                                              reference=kwargs.get("reference"))
        sk_learn_evaluator = SKLearnEvaluator(reference=kwargs.get("reference"))
        super().__init__(splitting=splitting, estimator=sk_learn_estimator, evaluator=sk_learn_evaluator, **kwargs)


class SKLearnPipelineV2(pipeline.Pipeline):
    def __init__(self, *, preprocessing_fn: Optional[Union[CodeMixin, Callable]],
                 splitting: Optional[Union[Type[CodeMixin], Callable]] = None, pipeline: Pipeline, **kwargs):
        super().__init__(**kwargs)

        self._preprocessor = PipelineComponent(name="preprocessor", provides=["dataset"], code=preprocessing_fn,
                                               **kwargs) if preprocessing_fn else None
        if splitting is None:
            self._splitter = DefaultSplitComponent(predecessors=self._preprocessor, reference=kwargs.get("reference"))
        else:
            self._splitter = SplitComponent(code=splitting, predecessors=self._preprocessor, **kwargs)

        self.add_node(self._splitter)

        self._estimators = []
        predecessor = self._splitter
        for name, component in pipeline.named_steps:
            estimator = EstimatorComponentMixin(name=name, code=component.__class__, predecessors=[predecessor],
                                                **kwargs)
            self._estimators.append(estimator)
            self.add_node(estimator)
            predecessor = estimator

        self._evaluator = SKLearnEvaluator(predecessors=[predecessor], reference=kwargs.get("reference"))
        self.add_node(self._evaluator)

        # Build the graph
        if self._preprocessor:
            self.add_edge(self._preprocessor, self._splitter)
        self.add_edge(self._splitter, self._estimators[0])
        for i in range(len(self._estimators) - 1):
            self.add_edge(self._estimators[i], self._estimators[i + 1])

        self.add_edge(self._estimators[-1], self._evaluator)

    @property
    def estimators(self):
        return self._estimators
