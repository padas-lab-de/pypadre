from collections import Callable
from copy import deepcopy
from typing import cast, Optional, Union

import numpy as np
from padre.PaDREOntology import PaDREOntology
from sklearn.pipeline import Pipeline

from pypadre.binding.visitors.scikit import SciKitVisitor
from pypadre.core.base import exp_events, phases
from pypadre.core.model.code.code import Code
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.computation.evaluation import Evaluation
from pypadre.core.model.computation.training import Training
from pypadre.core.model.pipeline import pipeline
from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline
from pypadre.core.model.pipeline.components import EstimatorComponent, EvaluatorComponent
from pypadre.core.model.split.split import Split
from pypadre.core.visitors.mappings import name_mappings, alternate_name_mappings


def _is_sklearn_pipeline(pipeline):
    """
    checks whether pipeline is a sklearn pipeline
    :param pipeline:
    :return:
    """
    # we do checks via strings, not isinstance in order to avoid a dependency on sklearn
    return type(pipeline).__name__ == 'Pipeline' and type(pipeline).__module__ == 'sklearn.pipeline'


class SKLearnEstimator(EstimatorComponent):
    """
    This class encapsulates an sklearn workflow which allows to run sklearn pipelines or a list of sklearn components,
    report the results according to the outcome via the experiment logger.

    A workflow is a single run of fitting, transformation and inference.
    It does not contain any information on the particular split or state of an experiment.
    Workflows are used for abstracting from the underlying machine learning framework.
    """

    def __init__(self, *, pipeline=None, **kwargs):
        # TODO don't change state of pipeline!!!
        # check for final component to determine final results
        # if step wise is true, log intermediate results. Otherwise, log only final results.
        # distingusish between training and fitting in classification.

        if not pipeline or not _is_sklearn_pipeline(pipeline):
            raise ValueError("SKLearnEstimator needs a delegate defined as sklearn.pipeline")
        self._pipeline = pipeline

        super().__init__(name="SKLearnEstimator", **kwargs)

    def _execute_(self, *, data, **kwargs):
        split = data

        self.send_start(phase='sklearn.' + phases.fitting)
        y = None
        if split.train_targets is not None:
            y = split.train_targets.reshape((len(split.train_targets),))
        else:
            # Create dummy target of zeros if target is not present.
            y = np.zeros(shape=(len(split.train_features, )))
        self._pipeline.fit(split.train_features, y)
        self.send_stop(phase='sklearn.' + phases.fitting)
        if self.is_scorer():
            self.send_start(phase=f"sklearn.scoring.trainset")
            score = self._pipeline.score(split.train_features, y)
            self.send_stop(phase=f"sklearn.scoring.trainset")
            # TODO use other signals?
            self.send_log(keys=['training score'], values=[score], message="Logging the training score")

            if split.has_valset():
                y = split.val_targets.reshape((len(split.val_targets),))
                self.send_start(phase='sklearn.scoring.valset')
                score = self._pipeline.score(split.val_features, y)
                self.send_stop(phase='sklearn.scoring.valset')
                self.send_log(keys=['validation score'], values=[score], message="Logging the validation score")
        return Training(split=split, model=self._pipeline, **kwargs)

    def hash(self):
        # TODO hash should not change with training
        return self.pipeline.__hash__()

    def configuration(self):
        return SciKitVisitor(self._pipeline)

    def is_inferencer(self):
        return getattr(self._pipeline, "predict", None)

    def is_scorer(self):
        return getattr(self._pipeline, "score", None)

    def is_transformer(self):
        return getattr(self._pipeline, "transform", None)

    @property
    def pipeline(self):
        return self._pipeline


class SKLearnEvaluator(EvaluatorComponent):
    """
    This class takes the output of an sklearn workflow which represents the fitted model along with the corresponding split,
    report and save all possible results that allows for common/custom metric computations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def hash(self):
        # TODO
        return self.__hash__()

    def _execute_(self, *, data, **kwargs):
        model = data.model
        split = data.split

        # TODO CLEANUP. METRICS SHOULDN'T BE CALCULATED HERE BUT CALCULATED BY INDEPENDENT METRICS MEASURES
        # TODO still allow for custom metrics which are added by using sklearn here?

        train_idx = split.train_idx
        test_idx = split.test_idx

        self.send_error(message="Test set is missing.", condition=not split.has_testset())

        self.send_start(phase='sklearn.' + phases.inferencing)
        train_idx = train_idx.tolist()
        test_idx = test_idx.tolist()

        y_predicted_probabilities = None
        y = split.test_targets.reshape((len(split.test_targets),))

        y_predicted = np.asarray(model.predict(split.test_features))
        self.send_stop(phase='sklearn.' + phases.inferencing)

        results = {'predicted': y_predicted.tolist(),
                   'truth': y.tolist()}

        modified_results = dict()

        self.send_log(mode='probability', pred=y_predicted, truth=y,
                      message="Checking if the workflow supports probability computation or not.")

        # Check if the final estimator has an attribute called probability and if it has check if it is True
        compute_probabilities = True
        if hasattr(model.steps[-1][1], 'probability') and not model.steps[-1][1].probability:
            compute_probabilities = False

        # log the probabilities of the result too if the method is present

        final_estimator_name = model.steps[-1][0]
        if name_mappings.get(final_estimator_name) is None:
            # If estimator name is not present in name mappings check whether it is present in alternate names
            estimator = alternate_name_mappings.get(str(final_estimator_name).lower())
            final_estimator_type = name_mappings.get(estimator).get('type')
        else:
            final_estimator_type = name_mappings.get(model.steps[-1][0]).get('type')

        self.send_error(condition=final_estimator_type is None,
                        message='Final estimator could not be found in names or alternate names')

        if final_estimator_type == 'Classification' or \
                (final_estimator_type == 'Neural Network' and np.all(np.mod(y_predicted, 1)) == 0):
            results['type'] = PaDREOntology.SubClassesExperiment.Classification.value

            if compute_probabilities:
                y_predicted_probabilities = model.predict_proba(split.test_features)
                self.send_log(mode='probability', pred=y_predicted, truth=y, probabilities=y_predicted_probabilities,
                              message="Computing and saving the prediction probabilities")
                results['probabilities'] = y_predicted_probabilities.tolist()
        else:
            results['type'] = PaDREOntology.SubClassesExperiment.Regression.value

        if self.is_scorer(model):
            self.send_start(phase=f"sklearn.scoring.testset")
            score = model.score(split.test_features, y, )
            self.send_stop(phase=f"sklearn.scoring.testset")
            self.send_log(keys=["test score"], values=[score], message="Logging the testing score")

        results['dataset'] = split.dataset.name
        results['train_idx'] = train_idx
        results['test_idx'] = test_idx
        results['training_sample_count'] = len(train_idx)
        results['testing_sample_count'] = len(test_idx)
        results['split_num'] = split.number

        return Evaluation(training=data, metadata=results, **kwargs)

    @staticmethod
    def is_inferencer(model=None):
        return getattr(model, 'predict', None)

    @staticmethod
    def is_scorer(model=None):
        return getattr(model, 'score', None)

    @staticmethod
    def is_transformer(model=None):
        return getattr(model, 'transform', None)


class SKLearnPipeline(DefaultPythonExperimentPipeline):
    def __init__(self, *, splitting: Union[Code, Callable] = None, pipeline_fn: Callable, **kwargs):
        """

        :param splitting:
        :param pipeline_fn: A function that returns a Sklearn pipeline as the return value
        :param kwargs:
        """
        pipeline = pipeline_fn()
        visitor = SciKitVisitor(pipeline)

        # Check if the return type is a Sklearn Pipeline
        assert(isinstance(pipeline, Pipeline))

        # Verify running two instances of the function creates two Pipeline objects
        assert(pipeline is not pipeline_fn())
        sk_learn_estimator = SKLearnEstimator(pipeline=pipeline)
        sk_learn_evaluator = SKLearnEvaluator()
        super().__init__(splitting=splitting, estimator=sk_learn_estimator, evaluator=sk_learn_evaluator, **kwargs)


class SKLearnPipelineV2(pipeline.Pipeline):
    def __init__(self, *, splitting: Union[Code, Callable] = None, pipeline: Pipeline, **kwargs):
        super().__init__(**kwargs)
        # TODO for each pipeline element in sklearn create a pipeline component
