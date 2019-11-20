from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from pypadre.core.util.utils import persistent_hash
from pypadre.examples.base_example import example_app
from pypadre.pod.util.git_util import git_hash

app = example_app()


@app.dataset(name="iris", columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                   'petal width (cm)', 'class'], target_features='class')
def dataset():
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)


@app.parameter_map()
def parameters():
    return {'SKLearnEstimator': {'parameters': {'SVC': {'C': [1.0]}, 'PCA': {'n_components': [3]}}}}


@app.experiment(dataset=dataset, reference_git=__file__, parameters=parameters,
                experiment_name="Iris SVC", project_name="Examples")
def experiment():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('PCA', PCA()), ('SVC', SVC(probability=True))]
    return Pipeline(estimators)


print(experiment.reference_hash)
Code_Reference = persistent_hash((str(Path(__file__).parent),git_hash(str(Path(__file__).parent))))
print('Versioned code git reference hash: {}'.format(Code_Reference))
