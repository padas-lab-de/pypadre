from pypadre.core.model.dataset.dataset import Transformation
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


@app.preprocessing(reference_git=__file__, store=True)
def preprocessing(ctx, **kwargs):
    (dataset,) = unpack(ctx, "data")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(dataset.features())
    _dataset = Transformation(name="Standarized_%s" % dataset.name, dataset=dataset)
    _features = scaler.transform(dataset.features())
    targets = dataset.targets()
    new_data = np.hstack((_features, targets))
    _dataset.set_data(new_data, attributes=dataset.attributes)
    return _dataset


@app.parameter_map()
def parameters():
    return {'SKLearnEstimator': {'parameters': {'SVC': {'C': [1.0]}, 'PCA': {'n_components': [3]}}}}


@app.experiment(dataset=dataset, reference_git=__file__, parameters=parameters, preprocessing_fn=preprocessing,
                experiment_name="Iris SVC", project_name="Examples", ptype=SKLearnPipeline)
def experiment():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('PCA', PCA()), ('SVC', SVC(probability=True))]
    return Pipeline(estimators)
