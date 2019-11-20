from sklearn.datasets import load_iris
import numpy as np
from pypadre.examples.base_example import example_app

app = example_app()


@app.dataset(name="iris",
             columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                      'petal width (cm)', 'class'], target_features='class')
def dataset():
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)


@app.experiment(dataset=dataset,
                reference_package=__file__,
                experiment_name="Iris SVC_", project_name="Examples")
def experiment():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)

# executions = app.executions.list({'experiment_id':experiment.id})
#TODO get and show git diff between executions (commits IDs)


