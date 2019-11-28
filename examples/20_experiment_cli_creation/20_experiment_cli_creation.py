from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline
from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.pod.app.padre_app import PadreAppFactory

app = PadreAppFactory.get(config)


@app.dataset(name="iris", columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                   'petal width (cm)', 'class'], target_features='class')
def data():
    from sklearn.datasets import load_iris
    import numpy as np
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)

@app.custom_splitter(name="Custom splitter", reference_git=path)
def splitter(dataset, **kwargs):
    import numpy as np
    idx = np.arange(dataset.size[0])
    cutoff = int(len(idx) / 2)
    return idx[:cutoff], idx[cutoff:], None

@app.parameter_map()
def params():
    return {'SKLearnEstimator': {'parameters': {'SVC': {'C': [1.0,2.0,3.0]}, 'PCA': {'n_components': [3,4]}}}}

@app.experiment(dataset=data, reference_git=path, parameters=params, splitting=splitter,
                experiment_name=experiment_name, project_name=project_name, ptype=SKLearnPipeline)
def main():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    estimators = [('PCA', PCA(n_components=4)), ('SVC', SVC(probability=True))]
    return Pipeline(estimators)
