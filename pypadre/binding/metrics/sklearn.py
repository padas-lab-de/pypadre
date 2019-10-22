from typing import Optional

import numpy as np

from pypadre.core.metrics.metrics import MeasureMeter, Metric
from pypadre.core.model.computation.computation import Computation


class ConfusionMatrix(MeasureMeter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _execute_helper(self, *, computation: Computation, **kwargs) -> Optional[Metric]:
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

        predicted = computation.result.predicted
        truth = computation.result.truth

        if predicted is None or truth is None or len(predicted) != len(truth):
            self.send_error("")
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

        return Metric(name="ConfusionMatrix", computation=computation, result=copy.deepcopy(confusion_matrix.tolist()))


class RegressionMetrics(MeasureMeter):
    def _execute_helper(self, *, truth, predicted, **kwargs):
        """
        The function computes the regression metrics of results

        :param predicted: The predicted values

        :param truth: The truth values

        :return: Dictionary containing the computed metrics
        """
        metrics = []
        error = truth - predicted
        metrics.append(Metric('mean_error', np.mean(error)))
        metrics.append(Metric('mean_absolute_error', np.mean(abs(error))))
        metrics.append(Metric('standard_deviation', np.std(error)))
        metrics.append(Metric('max_absolute_error', np.max(abs(error))))
        metrics.append(Metric('min_absolute_error', np.min(abs(error))))
        return metrics


class ClassificationMetrics(MeasureMeter):
    # TODO extend
    def _execute_helper(self, *, confusion_matrix=None, option='macro', **kwargs):
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
