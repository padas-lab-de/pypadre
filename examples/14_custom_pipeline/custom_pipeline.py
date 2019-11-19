from sklearn.datasets import load_iris
import numpy as np
from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline
from pypadre.core.util.utils import unpack
from pypadre.examples.base_example import example_app


app = example_app()


# Custom pipeline example for graph embedding evaluation

@app.dataset(name="iris", columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                   'petal width (cm)', 'class'], target_features='class')
def dataset():
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)


def estimator(ctx,**kwargs):
    (split, component, run, initial_hyperparameters) = unpack(ctx, "data", "component", "run",
                                                              "initial_hyperparameters")
