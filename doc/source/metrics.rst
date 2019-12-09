Metrics
=========
Metrics are an important part of an experiment. It allows us to compare different experiments and make informed
decisions. In PaDRe, metrics are a very important part of running experiments. The users can compute different metrics
and basic metrics are provided PaDRe itself, such as confusion matrix, precision, recall, f1 measure and accuracy for
classification and mean error, mean absolute error etc for regression.

Metrics can be computed at the output of any component. For example, the user might want to know the average distance
between points in an embedding space and this should be computed after the preprocessing step. PaDRe supports
computation of intermediate metrics from components. All the metrics are stored in a metrics registry as metric objects.
Each metric object has a field that specifies what type of data it would consume. For example, for precision it would
be a confusion matrix. A matrix graph is constructed by chaining the producer of metrics and the next consumer. PaDRe
supports branching of metrics too.

During execution of each component, PaDRe checks whether there are any available metrics for that particular component
and if there are available metrics, PaDRe prints out the available metrics to the user.

If the user wants to add a custom metric, all the user has to do is wrap the function for computing the metric in a
metric object, specify what the metric object would consume and add the metric object to the metric registry. The user
also has to create a reference to code for logging the version of the code.

.. code-block:: python



And if the user wants only a specific metric that can also be accomplished in PaDRe. For example, if the user needs only
the confusion matrix, the user specifies that only this metric is needed and PaDRe computes all the simple paths in the
graph for that component that leads up to that metric and executes only those paths.

.. code-block:: python

    ACCURACY = "Accuracy"


    # Defining the function how the metric is computed
    def accuracy_computation(ctx, option='macro', **kwargs):
        (confusion_matrix_metric, computation) = unpack(ctx, "data", "computation")
        confusion_matrix = confusion_matrix_metric.result

        """
        This function calculates the accuracy
        :param confusion_matrix: The confusion matrix of the classification
        :param option: Micro averaged or macro averaged

        :return: accuracy
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
                                                                                    version=_version.__version__)),
                                                                                    **kwargs)

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

In addition to all this, there is a flag that is known as allow\_metrics. This flag allows the user to turn off the
metric computation if there is ever such a scenario. The allow\_metric flag is set as true by default but can be turned
off on a component level basis.
