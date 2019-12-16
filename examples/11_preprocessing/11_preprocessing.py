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


@app.preprocessing(reference_git=__file__)
def preprocessing(dataset, **kwargs):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(dataset.features())
    _features = scaler.transform(dataset.features())
    targets = dataset.targets()
    new_data = np.hstack((_features, targets))
    return new_data


@app.parameter_map()
def parameters():
    return {'SKLearnEstimator': {'parameters': {'SVC': {'C': [1.0]}, 'PCA': {'n_components': [3]}}}}


@app.experiment(dataset=dataset, reference_git=__file__, parameters=parameters, preprocessing_fn=preprocessing,
                experiment_name="Iris SVC - preprocessing", project_name="Examples", ptype=SKLearnPipeline)
def experiment():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('PCA', PCA()), ('SVC', SVC(probability=True))]
    return Pipeline(estimators)
