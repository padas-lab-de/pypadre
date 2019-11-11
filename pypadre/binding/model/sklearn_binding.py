from collections import Callable
from typing import Union, Type

from sklearn.pipeline import Pipeline

from pypadre.binding.model.sklearn_estimator import SKLearnEstimator
from pypadre.binding.model.sklearn_evaluator import SKLearnEvaluator
from pypadre.binding.visitors.scikit import SciKitVisitor
from pypadre.core.model.code.codemixin import CodeMixin
from pypadre.core.model.pipeline import pipeline
from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline


class SKLearnPipeline(DefaultPythonExperimentPipeline):

    def __init__(self, *, splitting: Union[Type[CodeMixin], Callable] = None, parameter_provider=None, pipeline_fn: Callable, **kwargs):
        """

        :param splitting:
        :param pipeline_fn: A function that returns a Sklearn pipeline as the return value
        :param kwargs:
        """
        pipeline = pipeline_fn()
        visitor = SciKitVisitor(pipeline)
        # TODO use visitor to extract parameter schema from pipeline

        # Check if the return type is a Sklearn Pipeline
        assert(isinstance(pipeline, Pipeline))

        # Verify running two instances of the function creates two Pipeline objects
        assert(pipeline is not pipeline_fn())

        # TODO provider for a specific node
        sk_learn_estimator = SKLearnEstimator(pipeline=pipeline, parameter_provider=parameter_provider)
        sk_learn_evaluator = SKLearnEvaluator()
        super().__init__(splitting=splitting, estimator=sk_learn_estimator, evaluator=sk_learn_evaluator, **kwargs)


class SKLearnPipelineV2(pipeline.Pipeline):
    def __init__(self, *, splitting: Union[CodeMixin, Callable] = None, pipeline: Pipeline, **kwargs):
        super().__init__(**kwargs)
        # TODO for each pipeline element in sklearn create a pipeline component
