import numpy as np
from padre.eventhandler import trigger_event
from padre.core.base import exp_events, phases
from padre.visitors.scikit import SciKitVisitor
from padre.eventhandler import assert_condition

class SKLearnWorkflow:
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

    def __init__(self, pipeline, step_wise=False):
        # check for final component to determine final results
        # if step wise is true, log intermediate results. Otherwise, log only final results.
        # distingusish between training and fitting in classification.
        self._pipeline = pipeline
        self._step_wise = step_wise
        self._results = dict()
        self._metrics = dict()
        self._hyperparameters = dict()

    def fit(self, ctx):
        # todo split as parameter just for logging is not very good design. Maybe builder pattern would be better?
        if self._step_wise:
            raise NotImplemented()
        else:
            # Trigger event
            trigger_event('EVENT_LOG_EVENT', source=ctx, kind=exp_events.start, phase='sklearn.'+phases.fitting)
            y = None
            if ctx.train_targets is not None:
                y = ctx.train_targets.reshape((len(ctx.train_targets),))
            else:
                # Create dummy target of zeros if target is not present.
                y = np.zeros(shape=(len(ctx.train_features,)))

            assert_condition(condition=np.all(np.mod(ctx.train_targets, 1) == 0) and
                                       self._pipeline._estimator_type is 'classifier',
                             source=self, message='Classification not possible on continous data')
            self._pipeline.fit(ctx.train_features, y)
            trigger_event('EVENT_LOG_EVENT', source=ctx, kind=exp_events.stop, phase='sklearn.'+phases.fitting)
            if self.is_scorer():
                trigger_event('EVENT_LOG_EVENT', source=ctx, kind=exp_events.start, phase=f"sklearn.scoring.trainset")
                score = self._pipeline.score(ctx.train_features, y)
                trigger_event('EVENT_LOG_EVENT', source=ctx, kind=exp_events.stop, phase=f"sklearn.scoring.trainset")

                # Trigger event
                trigger_event('EVENT_LOG_RESULTS', source=ctx, keys=['training score'], values=[score])

                if ctx.has_valset():
                    y = ctx.val_targets.reshape((len(ctx.val_targets),))
                    trigger_event('EVENT_LOG_EVENT', source=ctx, kind=exp_events.start, phase='sklearn.scoring.valset')
                    score = self._pipeline.score(ctx.val_features, y)
                    trigger_event('EVENT_LOG_EVENT', source=ctx, kind=exp_events.stop, phase='sklearn.scoring.valset')
                    trigger_event('EVENT_LOG_SCORE', source=ctx, keys=['validation score'], values=score)

    def infer(self, ctx, train_idx, test_idx):
        from copy import deepcopy
        if self._step_wise:
            # step wise means going through every component individually and log their results / timing
            raise NotImplemented()
        else:
            # do logging here
            if ctx.has_testset() and self.is_inferencer():

                y = ctx.test_targets.reshape((len(ctx.test_targets),))
                # todo: check if we can estimate probabilities, scores or hard decisions
                # this also changes the result type to be written.
                # if possible, we will always write the "best" result type, i.e. which retains most information (
                # if
                y_predicted = np.asarray(self._pipeline.predict(ctx.test_features))
                results = {'predicted': y_predicted.tolist(),
                           'truth': y.tolist()}

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
                if 'predict_proba' in dir(self._pipeline.steps[-1][1]) and np.all(np.mod(y_predicted, 1) == 0) and \
                        compute_probabilities:
                    y_predicted_probabilities = self._pipeline.predict_proba(ctx.test_features)
                    trigger_event('EVENT_LOG_RESULTS', source=ctx, mode='probability', pred=y_predicted, truth=y,
                                  probabilities=y_predicted_probabilities, scores=None,
                                  transforms=None, clustering=None)
                    results['probabilities'] = y_predicted_probabilities.tolist()
                    results['type'] = 'classification'
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

    def compute_confusion_matrix(self, Predicted=None,
                                 Truth=None):
        """
        This function computes the confusion matrix of a classification result.
        This was done as a general purpose implementation of the confusion_matrix
        :param Predicted: The predicted values of the confusion matrix
        :param Truth: The truth values of the confusion matrix
        :return: The confusion matrix
        """
        import copy
        if Predicted is None or Truth is None or \
                len(Predicted) != len(Truth):
            return None

        # Get the number of labels from the predicted and truth set
        label_count = len(set(Predicted).union(set(Truth)))
        confusion_matrix = np.zeros(shape=(label_count, label_count), dtype=int)
        # If the labels given do not start from 0 and go up to the label_count - 1,
        # a mapping function has to be created to map the label to the corresponding indices
        if (min(Predicted) != 0 and min(Truth) != 0) or \
                (max(Truth) != label_count - 1 and max(Predicted) != label_count - 1):
            labels = list(set(Predicted).union(set(Truth)))
            for idx in range(0, len(Truth)):
                row_idx = int(labels.index(Truth[idx]))
                col_idx = int(labels.index(Predicted[idx]))
                confusion_matrix[row_idx][col_idx] += 1

        else:

            # Iterate through the array and update the confusion matrix
            for idx in range(0, len(Truth)):
                confusion_matrix[int(Truth[idx])][int(Predicted[idx])] += 1

        return copy.deepcopy(confusion_matrix.tolist())

    def compute_classification_metrics(self, confusion_matrix=None, option='macro'):
        """
        This function calculates the classification metrics like precision,
        recall, f-measure, accuracy etc
        TODO: Implement weighted sum of averaging metrics

        :param confusion_matrix: The confusion matrix of the classification
        :param option: Micro averaged or macro averaged

        :return: Classification metrics as a dictionary
        """
        import copy
        if confusion_matrix is None:
            return None

        classification_metrics = dict()
        precision = np.zeros(shape=(len(confusion_matrix)))
        recall = np.zeros(shape=(len(confusion_matrix)))
        f1_measure = np.zeros(shape=(len(confusion_matrix)))
        tp = 0
        column_sum = np.sum(confusion_matrix, axis=0)
        row_sum = np.sum(confusion_matrix, axis=1)
        for idx in range(0, len(confusion_matrix)):
            tp = tp + confusion_matrix[idx][idx]
            # Removes the 0/0 error
            precision[idx] = np.divide(confusion_matrix[idx][idx], column_sum[idx] + int(column_sum[idx] == 0))
            recall[idx] = np.divide(confusion_matrix[idx][idx], row_sum[idx] + int(row_sum[idx] == 0))
            if recall[idx] == 0 or precision[idx] == 0:
                f1_measure[idx] = 0
            else:
                f1_measure[idx] = 2 / (1.0 / recall[idx] + 1.0 / precision[idx])

        accuracy = tp / np.sum(confusion_matrix)
        if option == 'macro':
            classification_metrics['recall'] = float(np.mean(recall))
            classification_metrics['precision'] = float(np.mean(precision))
            classification_metrics['accuracy'] = accuracy
            classification_metrics['f1_score'] = float(np.mean(f1_measure))

        elif option == 'micro':
            classification_metrics['recall'] = accuracy
            classification_metrics['precision'] = accuracy
            classification_metrics['accuracy'] = accuracy
            classification_metrics['f1_score'] = accuracy

        else:
            classification_metrics['recall'] = recall.tolist()
            classification_metrics['precision'] = precision.tolist()
            classification_metrics['accuracy'] = accuracy
            classification_metrics['f1_score'] = f1_measure.tolist()

        return copy.deepcopy(classification_metrics)

    def compute_regression_metrics(self, predicted=None, truth=None):
        """
        The function computes the regression metrics of results

        :param predicted: The predicted values

        :param truth: The truth values

        :return: Dictionary containing the computed metrics
        """
        metrics_dict = dict()
        error = truth - predicted
        metrics_dict['mean_error'] = np.mean(error)
        metrics_dict['mean_absolute_error'] = np.mean(abs(error))
        metrics_dict['standard_deviation'] = np.std(error)
        metrics_dict['max_absolute_error'] = np.max(abs(error))
        metrics_dict['min_absolute_error'] = np.min(abs(error))
        return metrics_dict