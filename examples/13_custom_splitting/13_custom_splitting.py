from pypadre.core.util.utils import unpack
from pypadre.examples.base_example import example_app
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from pypadre.binding.model.sklearn_binding import SKLearnPipeline

app = example_app()


@app.dataset(name="iris", columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                   'petal width (cm)', 'class'], target_features='class')
def dataset():
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)


@app.custom_splitter(name="Custom splitter", reference_package=__file__)
def custom_splitter(ctx, **kwargs):
    (data,) = unpack(ctx, "data")
    idx = np.arange(data.size[0])
    cutoff = int(len(idx) / 2)
    return idx[:cutoff], idx[cutoff:], None


@app.parameter_map()
def parameters():
    return {'SKLearnEstimator': {'parameters': {'SVC': {'C': [1.0]}, 'PCA': {'n_components': [3]}}}}

@app.experiment(dataset=dataset, reference_package=__name__, parameters=parameters, splitting=custom_splitter,
                experiment_name="Iris SVC", project_name="Examples", ptype=SKLearnPipeline)
def experiment():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('PCA', PCA()), ('SVC', SVC(probability=True))]
    return Pipeline(estimators)