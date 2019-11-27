import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.examples.base_example import example_app

app = example_app()


@app.dataset(name="iris", columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                   'petal width (cm)', 'class'], target_features='class')
def dataset():
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)


@app.parameter_map()
def parameters():
    return {'SKLearnEstimator': {'parameters': {'SVC': {'C': 1.0}, 'PCA': {'n_components': [3]}}},
            'default_split': {'parameters': {'strategy': ['cv'], 'n_folds': [4]}}}


# default_split is the name of Padre default splitter component. For custom splitting, see the example
# in 13_custom_splitting

@app.experiment(dataset=dataset, reference_git=__file__, parameters=parameters,
                experiment_name="Iris SVC", project_name="Examples", ptype=SKLearnPipeline)
def experiment():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('PCA', PCA()), ('SVC', SVC(probability=True))]
    return Pipeline(estimators)
