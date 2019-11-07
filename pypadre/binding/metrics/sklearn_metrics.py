from copy import deepcopy
from typing import Optional, List

import numpy as np
from padre.PaDREOntology import PaDREOntology

from pypadre import _name, _version
from pypadre.core.metrics.metric_registry import metric_registry
from pypadre.core.metrics.metrics import MetricProviderMixin, Metric
from pypadre.core.model.generic.custom_code import ProvidedCodeMixin
from pypadre.core.model.pipeline.components.component_mixins import EvaluatorComponentMixin
from pypadre.core.util.utils import unpack

TOTAL_ERROR = "total_error"
MEAN_ERROR = "mean_error"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
STANDARD_DEVIATION = "standard_deviation"
MAX_ABSOLUTE_ERROR = "max_absolute_error"
MIN_ABSOLUTE_ERROR = "min_absolute_error"

PRECISION = "precision"
RECALL = "recall"
ACCURACY = "accuracy"
F1_SCORE = "f1_score"

CLASSIFICATION_METRICS = "classification_metrics"
REGRESSION_METRICS = "regression_metrics"
CONFUSION_MATRIX = "ConfusionMatrix"


def matrix(ctx, **kwargs) -> Optional[Metric]:
    # TODO extend
    # :param predicted: The predicted values of the confusion matrix
    # :param truth: The truth values of the confusion matrix
    """
            This function computes the confusion matrix of a classification result.
            This was done as a general purpose implementation of the confusion_matrix
            :param computation: The computation which is the input for confusion matrix calculation. Has to hold truth and predicted values.
            :return: The confusion matrix
            """
    import copy
    # import the constant strings that are the dictionary keys from the evaluator component
    from pypadre.core.model.pipeline.components.component_mixins import EvaluatorComponentMixin
    (computation,) = unpack(ctx, "computation")

    # create the predicted values and the truth values array from the computation results
    predictions = computation.result[EvaluatorComponentMixin.PREDICTIONS]

    predicted = []
    truth = []

    # The predictions dictionary contains as the key the testing row index, and the value is a dictionary. The
    # dictionary contains the truth value, predicted value and probabilities
    for row_idx in predictions:
        prediction_results = predictions.get(row_idx)
        predicted.append(prediction_results.get(EvaluatorComponentMixin.PREDICTED))
        truth.append(prediction_results.get(EvaluatorComponentMixin.TRUTH))

    if predicted is None or truth is None or len(predicted) != len(truth):
        computation.send_error("")
        return None

    # Get the number of labels from the predicted and truth set
    label_count = len(set(predicted).union(set(truth)))
    confusion_matrix = np.zeros(shape=(label_count, label_count), dtype=int)
    # If the labels given do not start from 0 and go up to the label_count - 1,
    # a mapping function has to be created to map the label to the corresponding indices
    if (min(predicted) != 0 and min(truth) != 0) or \
            (max(truth) != label_count - 1 and max(predicted) != label_count - 1):
        labels = list(set(predicted).union(set(truth)))
        for idx in range(0, len(truth)):
            row_idx = int(labels.index(truth[idx]))
            col_idx = int(labels.index(predicted[idx]))
            confusion_matrix[row_idx][col_idx] += 1

    else:

        # Iterate through the array and update the confusion matrix
        for idx in range(0, len(truth)):
            confusion_matrix[int(truth[idx])][int(predicted[idx])] += 1

    return Metric(name=CONFUSION_MATRIX, computation=computation, result=copy.deepcopy(confusion_matrix.tolist()))


class ConfusionMatrix(ProvidedCodeMixin, MetricProviderMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(package=__name__, fn_name="matrix", requirement=_name.__name__,
                         version=_version.__version__, **kwargs)

    @property
    def consumes(self) -> str:
        # return "classification"
        return PaDREOntology.SubClassesExperiment.Classification.value


def regression(ctx, **kwargs) -> Optional[List[Metric]]:

    (computation,) = unpack(ctx, "computation")

    predictions = computation.result[EvaluatorComponentMixin.PREDICTIONS]

    predicted = []
    truth = []

    # The predictions dictionary contains as the key the testing row index, and the value is a dictionary. The
    # dictionary contains the truth value, predicted value and probabilities
    for row_idx in predictions:
        prediction_results = predictions.get(row_idx)
        predicted.append(prediction_results.get(EvaluatorComponentMixin.PREDICTED))
        truth.append(prediction_results.get(EvaluatorComponentMixin.TRUTH))

    if predicted is None or truth is None or len(predicted) != len(truth):
        computation.send_error("")
        return None

    """
    The function computes the regression metrics of results

    :param predicted: The predicted values

    :param truth: The truth values

    :return: Dictionary containing the computed metrics
    """
    error = np.array(truth) - np.array(predicted)

    regression_metrics = dict()
    regression_metrics[TOTAL_ERROR] = float(np.sum(error))
    regression_metrics[MEAN_ERROR] = float(np.mean(error))
    regression_metrics[MEAN_ABSOLUTE_ERROR] = float(np.mean(abs(error)))
    regression_metrics[STANDARD_DEVIATION] = float(np.std(error))
    regression_metrics[MAX_ABSOLUTE_ERROR] = float(np.max(abs(error)))
    regression_metrics[MIN_ABSOLUTE_ERROR] = float(np.min(abs(error)))
    return Metric(name=REGRESSION_METRICS, computation=computation, result=deepcopy(regression_metrics))


class RegressionMetrics(ProvidedCodeMixin, MetricProviderMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(package=__name__, fn_name="regression", requirement=_name.__name__,
                         version=_version.__version__, **kwargs)

    @property
    def consumes(self) -> str:
        # return "regression"
        return PaDREOntology.SubClassesExperiment.Regression.value


# TODO extend
def classification(ctx, option='macro', **kwargs):
    (confusion_matrix_metric, computation) = unpack(ctx, "data", "computation")
    confusion_matrix = confusion_matrix_metric.result

    """
    This function calculates the classification metrics like precision,
    recall, f-measure, accuracy etc
    TODO: Implement weighted sum of averaging metrics

    :param confusion_matrix: The confusion matrix of the classification
    :param option: Micro averaged or macro averaged

    :return: Classification metrics as a dictionary
    """
    if confusion_matrix is None:
        return None

    classification_metrics = dict()
    precision = np.zeros(shape=(len(confusion_matrix)))
    recall = np.zeros(shape=(len(confusion_matrix)))
    f1_measure = np.zeros(shape=(len(confusion_matrix)))
    tp = 0
    column_sum = np.sum(confusion_matrix, axis=0)
    row_sum = np.sum(confusion_matrix, axis=1)
    for idx in range(len(confusion_matrix)):
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
        classification_metrics[RECALL] = float(np.mean(recall))
        classification_metrics[PRECISION] = float(np.mean(precision))
        classification_metrics[ACCURACY] = accuracy
        classification_metrics[F1_SCORE] = float(np.mean(f1_measure))

    elif option == 'micro':
        classification_metrics[RECALL] = accuracy
        classification_metrics[PRECISION] = accuracy
        classification_metrics[ACCURACY] = accuracy
        classification_metrics[F1_SCORE] = accuracy

    else:
        classification_metrics[RECALL] = recall.tolist()
        classification_metrics[PRECISION] = precision.tolist()
        classification_metrics[ACCURACY] = accuracy
        classification_metrics[F1_SCORE] = f1_measure.tolist()

    return Metric(name=CLASSIFICATION_METRICS, computation=computation, result=deepcopy(classification_metrics))


class ClassificationMetrics(ProvidedCodeMixin, MetricProviderMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(package=__name__, fn_name="classification", requirement=_name.__name__,
                         version=_version.__version__, **kwargs)

    @property
    def consumes(self) -> str:
        return str(ConfusionMatrix)


metric_registry.add_providers(ConfusionMatrix(), RegressionMetrics(), ClassificationMetrics())
