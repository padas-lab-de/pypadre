from collections import Callable
from typing import Union

import numpy as np
from sklearn.pipeline import Pipeline

from pypadre.binding.visitors.scikit import SciKitVisitor
from pypadre.core.base import exp_events, phases
from pypadre.core.model.pipeline.pipeline import Estimator, DefaultPythonExperimentPipeline
from pypadre.core.visitors.mappings import name_mappings, alternate_name_mappings
from pypadre.core.events import assert_condition
from pypadre.core.events import trigger_event


def _is_sklearn_pipeline(pipeline):
    """
    checks whether pipeline is a sklearn pipeline
    :param pipeline:
    :return:
    """
    # we do checks via strings, not isinstance in order to avoid a dependency on sklearn
    return type(pipeline).__name__ == 'Pipeline' and type(pipeline).__module__ == 'sklearn.pipeline'


class SKLearnEstimator(Estimator):
    """
    This class encapsulates an sklearn workflow which allows to run sklearn pipelines or a list of sklearn components,
    report the results according to the outcome via the experiment logger.

    A workflow is a single run of fitting, transformation and inference.
    It does not contain any information on the particular split or state of an experiment.
    Workflows are used for abstracting from the underlying machine learning framework.
    """

    _results = dict()
    _metrics = dict()
    _hyperparameters = None

    def __init__(self, *, pipeline=False, step_wise=False, **kwargs):
        # check for final component to determine final results
        # if step wise is true, log intermediate results. Otherwise, log only final results.
        # distingusish between training and fitting in classification.

        if not _is_sklearn_pipeline(pipeline):
            raise ValueError("SKLearnEstimator needs a delegate defined as sklearn.pipeline")

        self._pipeline = pipeline
        self._step_wise = step_wise
        self._results = dict()
        self._metrics = dict()
        self._hyperparameters = dict()

    def _fit(self, *args, data, **kwargs):
        # todo split as parameter just for logging is not very good design. Maybe builder pattern would be better?
        if self._step_wise:
            raise NotImplemented()
        else:
            # Trigger event
            trigger_event('EVENT_LOG_EVENT', source=data, kind=exp_events.start, phase='sklearn.' + phases.fitting)
            y = None
            if data.train_targets is not None:
                y = data.train_targets.reshape((len(data.train_targets),))
            else:
                # Create dummy target of zeros if target is not present.
                y = np.zeros(shape=(len(data.train_features, )))

            self._pipeline.fit(data.train_features, y)
            trigger_event('EVENT_LOG_EVENT', source=data, kind=exp_events.stop, phase='sklearn.' + phases.fitting)
            if self.is_scorer():
                trigger_event('EVENT_LOG_EVENT', source=data, kind=exp_events.start, phase=f"sklearn.scoring.trainset")
                score = self._pipeline.score(data.train_features, y)
                trigger_event('EVENT_LOG_EVENT', source=data, kind=exp_events.stop, phase=f"sklearn.scoring.trainset")

                # Trigger event
                trigger_event('EVENT_LOG_RESULTS', source=data, keys=['training score'], values=[score])

                if data.has_valset():
                    y = data.val_targets.reshape((len(data.val_targets),))
                    trigger_event('EVENT_LOG_EVENT', source=data, kind=exp_events.start, phase='sklearn.scoring.valset')
                    score = self._pipeline.score(data.val_features, y)
                    trigger_event('EVENT_LOG_EVENT', source=data, kind=exp_events.stop, phase='sklearn.scoring.valset')
                    trigger_event('EVENT_LOG_SCORE', source=data, keys=['validation score'], values=score)

    def _infer(self, *args, data, **kwargs):

        ctx = data
        if self.is_inferencer() and data.has_testset():
            train_idx = data.train_idx.tolist()
            test_idx = data.test_idx.tolist()


        from copy import deepcopy
        if self._step_wise:
            # step wise means going through every component individually and log their results / timing
            raise NotImplemented()
        else:
            # do logging here
            if ctx.has_testset() and self.is_inferencer():
                y_predicted_probabilities = None
                y = ctx.test_targets.reshape((len(ctx.test_targets),))
                # todo: check if we can estimate probabilities, scores or hard decisions
                # this also changes the result type to be written.
                # if possible, we will always write the "best" result type, i.e. which retains most information (
                # if
                y_predicted = np.asarray(self._pipeline.predict(ctx.test_features))
                results = {'predicted': y_predicted.tolist(),
                           'truth': y.tolist()}

                modified_results = dict()

                trigger_event('EVENT_LOG_RESULTS', source=ctx, mode='probability', pred=y_predicted, truth=y,
                              probabilities=None, scores=None, transforms=None, clustering=None)
                metrics = dict()
                metrics['dataset'] = ctx.dataset.name

                # Check if the final estimator has an attribute called probability and if it has check if it is True
                # SVC has such an attribute
                compute_probabilities = True
                if hasattr(self._pipeline.steps[-1][1], 'probability') and not self._pipeline.steps[-1][1].probability:
                    compute_probabilities = False

                # log the probabilities of the result too if the method is present
                final_estimator_type = None
                final_estimator_name = self._pipeline.steps[-1][0]
                if name_mappings.get(final_estimator_name) is None:
                    # If estimator name is not present in name mappings check whether it is present in alternate names
                    estimator = alternate_name_mappings.get(str(final_estimator_name).lower())
                    final_estimator_type = name_mappings.get(estimator).get('type')
                else:
                    final_estimator_type = name_mappings.get(self._pipeline.steps[-1][0]).get('type')

                assert_condition(condition= final_estimator_type is not None, source=self,
                                 message='Final estimator could not be found in names or alternate names')

                if final_estimator_type == 'Classification' or \
                        (final_estimator_type == 'Neural Network' and np.all(np.mod(y_predicted, 1)) == 0):
                    results['type'] = 'classification'

                    if compute_probabilities:
                        y_predicted_probabilities = self._pipeline.predict_proba(ctx.test_features)
                        trigger_event('EVENT_LOG_RESULTS', source=ctx, mode='probability', pred=y_predicted, truth=y,
                                      probabilities=y_predicted_probabilities, scores=None,
                                      transforms=None, clustering=None)
                        results['probabilities'] = y_predicted_probabilities.tolist()
                    # Calculate the confusion matrix
                    confusion_matrix = self.compute_confusion_matrix(Predicted=y_predicted.tolist(),
                                                                     Truth=y.tolist())
                    metrics['confusion_matrix'] = confusion_matrix
                    metrics['type'] = 'classification'

                    classification_metrics = self.compute_classification_metrics(confusion_matrix)
                    metrics.update(classification_metrics)

                else:
                    metrics['type'] = 'regression'
                    metrics.update(self.compute_regression_metrics(predicted=y_predicted, truth=y))
                    results['type'] = 'regression'

                self._metrics = deepcopy(metrics)

                if self.is_scorer():
                    score = self._pipeline.score(ctx.test_features, y, )
                    trigger_event('EVENT_LOG_SCORE', source=ctx, keys=["test score"], values=[score])

                results['dataset'] = ctx.dataset.name
                results['train_idx'] = train_idx
                results['test_idx'] = test_idx
                results['training_sample_count'] = len(train_idx)
                results['testing_sample_count'] = len(test_idx)
                results['split_num'] = ctx.number

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
                self._results = deepcopy(results)
                estimator_parameters = ctx.run.experiment.hyperparameters()

                # Save the hyperparameters to the workflow hyperparameters variable
                for curr_estimator in estimator_parameters:
                    parameters = estimator_parameters.get(curr_estimator).get('hyper_parameters').get(
                        'model_parameters')
                    param_value_dict = dict()
                    for curr_param in parameters:
                        param_value_dict[curr_param] = parameters.get(curr_param).get('value')

                    estimator_name = estimator_parameters.get(curr_estimator).get('algorithm').get('value')
                    self._hyperparameters[estimator_name] = deepcopy(param_value_dict)

    def is_inferencer(self):
        return getattr(self._pipeline, "predict", None)

    def is_scorer(self):
        return getattr(self._pipeline, "score", None)

    def is_transformer(self):
        return getattr(self._pipeline, "transform", None)

    def configuration(self):
        return SciKitVisitor(self._pipeline)

    @property
    def results(self):
        return self._results

    @property
    def metrics(self):
        return self._metrics

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @property
    def pipeline(self):
        return self._pipeline


class SKLearnPipeline(DefaultPythonExperimentPipeline):
    def __init__(self, *, splitting: Callable=None, pipeline: Pipeline, **kwargs):
        # TODO kwargs passing
        sk_learn_estimator = SKLearnEstimator(pipeline=pipeline, **kwargs.get("SKLearnPipeline", {}))
        super().__init__(splitting=splitting, estimator=sk_learn_estimator, **kwargs)
