"""
Create and save visualization for confusion matrix for multi class classification.

For this example
    - Iris dataset will be used in the experiment.
    - A new experiment will be executed and saved on the local system.
    - Visualization for confusion matrix will be saved on the local system.

"""

import numpy as np
import uuid
from sklearn.datasets import load_iris

# Import the metrics to register them
# noinspection PyUnresolvedReferences
from pypadre.binding.metrics import sklearn_metrics
from pypadre.core import plot
from pypadre.examples.base_example import example_app

app = example_app()
name = str(uuid.uuid4())[0:8]


@app.dataset(name="iris",
             columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                      'petal width (cm)', 'class'], target_features='class')
def dataset():
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)


@app.experiment(dataset=dataset, reference_git=__file__,
                experiment_name="Iris SVC - confusion matrix " + name, seed=1, allow_metrics=True,
                project_name="Examples")
def experiment():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)


run_id = experiment.executions[0].run[0].id
output_id = ""

for output_pipline in app._pipeline_output_app.list():
    if output_pipline.parent.id == run_id:
        output_id = output_pipline.id

plt = plot.ExperimentPlot(
    experiment.project.name,
    experiment.name,
    str(experiment.executions[0].id),
    run_id,
    output_id,
    base_path=app.backends[0].root_dir)

# Save visualization for the current experiment
vis = plt.plot_confusion_matrix("Iris SVC - confusion matrix")
app.backends[0].split.put_visualization(vis,
                                        file_name="confusion_matrix.json",
                                        base_path=plt.split_path())