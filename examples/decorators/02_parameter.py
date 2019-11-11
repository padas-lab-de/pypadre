
"""
This is a minimal example of a PaDRE experiment.
"""
import numpy as np
from sklearn.datasets import load_iris

from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.pod.app.padre_app import example_app

app = example_app()


@app.dataset(name="iris", columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                                'petal width (cm)', 'class'], target_features='class')
def dataset():
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)


@app.parameter_map()
def parameters():
    # TODO try to find right component to apply parameters to automatically
    return {'SKLearnEstimator': {'parameters': {'SVC': {'C': [0.5]}}}}


@app.workflow(dataset=dataset, parameters=parameters, experiment_name="Iris SVC", project_name="Examples", ptype=SKLearnPipeline)
def experiment():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)
