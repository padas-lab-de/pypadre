from collections import Callable
from copy import deepcopy
from typing import cast

import numpy as np
from sklearn.pipeline import Pipeline

from pypadre.binding.visitors.scikit import SciKitVisitor
from pypadre.core.base import exp_events, phases
from pypadre.core.model.computation.evaluation import Evaluation
from pypadre.core.model.computation.training import Training
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
            self.send_log(keys=['training score'], values=[score], message="TODO")

            if split.has_valset():
                y = split.val_targets.reshape((len(split.val_targets),))
                self.send_start(phase='sklearn.scoring.valset')
                score = self._pipeline.score(split.val_features, y)
                self.send_stop(phase='sklearn.scoring.valset')
                self.send_log(keys=['validation score'], values=[score], message="TODO")
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
        def is_inferencer():
            return getattr(model, "predict", None)

        def is_scorer():
            return getattr(model, "score", None)

        def is_transformer():
            return getattr(model, "transform", None)

        if is_inferencer() and split.has_testset():
            train_idx = split.train_idx.tolist()
            test_idx = split.test_idx.tolist()
        else:
            train_idx = split.train_idx
            test_idx = split.test_idx

        if split.has_testset() and is_inferencer():
            y_predicted_probabilities = None
            y = split.test_targets.reshape((len(split.test_targets),))
            # todo: check if we can estimate probabilities, scores or hard decisions
            # this also changes the result type to be written.
            # if possible, we will always write the "best" result type, i.e. which retains most information (
            # if
            y_predicted = np.asarray(model.predict(split.test_features))
            results = {'predicted': y_predicted.tolist(),
                       'truth': y.tolist()}

            modified_results = dict()

            self.send_log(mode='probability', pred=y_predicted, truth=y, message="TODO")

            # Check if the final estimator has an attribute called probability and if it has check if it is True
            # SVC has such an attribute
            compute_probabilities = True
            if hasattr(model.steps[-1][1], 'probability') and not model.steps[-1][1].probability:
                compute_probabilities = False

            # log the probabilities of the result too if the method is present
            final_estimator_type = None
            final_estimator_name = model.steps[-1][0]
            if name_mappings.get(final_estimator_name) is None:
                # If estimator name is not present in name mappings check whether it is present in alternate names
                estimator = alternate_name_mappings.get(str(final_estimator_name).lower())
                final_estimator_type = name_mappings.get(estimator).get('type')
            else:
                final_estimator_type = name_mappings.get(model.steps[-1][0]).get('type')

            self.send_error(condition=final_estimator_type is None, message='Final estimator could not be found in names or alternate names')

            if final_estimator_type == 'Classification' or \
                    (final_estimator_type == 'Neural Network' and np.all(np.mod(y_predicted, 1)) == 0):
                results['type'] = 'classification'

                if compute_probabilities:
                    y_predicted_probabilities = model.predict_proba(split.test_features)
                    self.send_log(mode='probability', pred=y_predicted, truth=y, probabilities=y_predicted_probabilities, message="TODO")
                    results['probabilities'] = y_predicted_probabilities.tolist()
            else:
                results['type'] = 'regression'

            if is_scorer():
                score = model.score(data.test_features, y, )
                self.send_log(keys=["test score"], values=[score], message="TODO")

            results['dataset'] = split.dataset.name
            results['train_idx'] = train_idx
            results['test_idx'] = test_idx
            results['training_sample_count'] = len(train_idx)
            results['testing_sample_count'] = len(test_idx)
            results['split_num'] = split.number

            if y_predicted_probabilities is None:
                for idx in range(0, len(test_idx)):
                    prop = dict()
                    prop['truth'] = int(y[idx])
                    prop['predicted'] = int(y_predicted[idx])
                    prop['probabilities'] = dict()
                    modified_results[test_idx[idx]] = deepcopy(prop)

            else:
                for idx in range(0, len(test_idx)):
                    prop = dict()
                    prop['truth'] = int(y[idx])
                    prop['probabilities'] = y_predicted_probabilities[idx].tolist()
                    prop['predicted'] = int(y_predicted[idx])
                    modified_results[test_idx[idx]] = deepcopy(prop)

            results['predictions'] = modified_results

            estimator_parameters = split.run.execution.experiment.hyperparameters()

            # Save the hyperparameters to the workflow hyperparameters variable
            for curr_estimator in estimator_parameters:
                parameters = estimator_parameters.get(curr_estimator).get('hyper_parameters').get(
                    'model_parameters')
                param_value_dict = dict()
                for curr_param in parameters:
                    param_value_dict[curr_param] = parameters.get(curr_param).get('value')

                estimator_name = estimator_parameters.get(curr_estimator).get('algorithm').get('value')
                self._hyperparameters[estimator_name] = deepcopy(param_value_dict)
            return Evaluation(training=data, metadata=results, **kwargs)
        else:
            raise NotImplementedError


class SKLearnPipeline(DefaultPythonExperimentPipeline):
    def __init__(self, *, splitting: Callable=None, pipeline: Pipeline, **kwargs):
        # TODO kwargs passing
        sk_learn_estimator = SKLearnEstimator(pipeline=pipeline, **kwargs.get("SKLearnPipeline", {}))
        sk_learn_evaluator = SKLearnEvaluator(pipeline=pipeline, **kwargs.get("SKLearnPipeline", {}))
        super().__init__(splitting=splitting, estimator=sk_learn_estimator, evaluator=sk_learn_evaluator, **kwargs)
