"""
This is a minimal example of a PaDRE experiment.
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

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
    return {'SKLearnEstimator': {'parameters': {'SVC': {'C': [0.1, 0.5, 1.0]}, 'PCA': {'n_components': [1, 2, 3]}}}}


@app.parameter_provider()
def provider(ctx, **parameters: dict):
    import itertools

    params_list = []
    master_list = []

    for estimator in parameters:
        param_dict = parameters.get(estimator)
        for parameter in param_dict:
            parameter_values = param_dict.get(parameter)

            def add_one(i):
                return i + 1

            if parameter == 'C':
                parameter_values = map(add_one, parameter_values)

            master_list.append(parameter_values)
            params_list.append(''.join([estimator, '.', parameter]))

    # TODO make this provider make sense
    grid = itertools.product(*master_list)
    return grid, params_list


@app.workflow(dataset=dataset, reference_package=__name__, parameters=parameters, parameter_provider=provider,
              experiment_name="Iris SVC", project_name="Examples", ptype=SKLearnPipeline)
def experiment():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('PCA', PCA()), ('SVC', SVC(probability=True))]
    return Pipeline(estimators)
