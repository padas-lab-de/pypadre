"""
This is a minimal example of a PaDRE experiment.
"""
from copy import deepcopy

import numpy as np
from sklearn.datasets import load_iris

# Import the metrics to register them
# noinspection PyUnresolvedReferences
from pypadre import _name, _version
from pypadre.binding.metrics import sklearn_metrics
from pypadre.binding.metrics.sklearn_metrics import ConfusionMatrix
from pypadre.core.metrics.metric_registry import metric_registry
from pypadre.core.metrics.metrics import MetricProviderMixin, Metric
from pypadre.core.model.code.code_mixin import PythonPackage, PipIdentifier
from pypadre.core.util.utils import unpack
from pypadre.examples.base_example import example_app

app = example_app()

# Defining the string name
ACCURACY = "Accuracy"


# Defining the function how the metric is computed
def accuracy_computation(ctx, option='macro', **kwargs):
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

    metrics = dict()
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
        metrics[ACCURACY] = accuracy

    elif option == 'micro':
        metrics[ACCURACY] = accuracy

    else:
        metrics[ACCURACY] = accuracy

    return Metric(name=ACCURACY, computation=computation, result=deepcopy(metrics))


# Defining the class that contains the metric, its name and what it consumes
class Accuracy(MetricProviderMixin):

    NAME = "Accuracy"

    def __init__(self, *args, **kwargs):
        super().__init__(code=PythonPackage(package=__name__, variable="accuracy_computation",
                                            repository_identifier=PipIdentifier(pip_package=_name.__name__,
                                                                                version=_version.__version__)), **kwargs)

    @property
    def consumes(self) -> str:
        return str(ConfusionMatrix)

    def __str__(self):
        return self.NAME


# Defining the reference to the metric
accuracy_ref = PythonPackage(package=__name__, variable="accuracy",
                             repository_identifier=PipIdentifier(pip_package=_name.__name__,
                                                                 version=_version.__version__))

# Creating the metric object
accuracy = Accuracy(reference=accuracy_ref)

# Adding it to the metric registry
metric_registry.add_providers(accuracy)



@app.dataset(name="iris",
             columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                      'petal width (cm)', 'class'], target_features='class')
def dataset():
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)


@app.experiment(dataset=dataset, reference_git=__file__,
                experiment_name="Iris SVC - Custom Metrics", seed=1, allow_metrics=True, project_name="Examples")
def experiment():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)
