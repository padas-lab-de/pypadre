from sklearn.datasets import load_iris
import numpy as np
from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline
from pypadre.core.util.utils import unpack
from pypadre.examples.base_example import example_app

app = example_app()


@app.dataset(name="iris", columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                   'petal width (cm)', 'class'], target_features='class')
def dataset():
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)


@app.parameter_map()
def config():
    return {"n_neighbors": 20}


@app.custom_splitter(name="Custom splitter", reference_git=__file__)
def custom_splitter(dataset, **kwargs):
    idx = np.arange(dataset.size[0])
    cutoff = int(len(idx) / 2)
    return idx[:cutoff], idx[cutoff:], None


@app.estimator(config=config, reference_git=__file__)
def estimator(X_train, y_train, *args, **kwargs):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(*args)
    knn.fit(X_train, y_train)
    return knn


@app.evaluator(task_type="Classification", reference_git=__file__)
def evaluator(model, X_test, *args, **kwargs):
    y_predicted = model.predict(X_test)
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
    return y_predicted, probabilities


@app.experiment(dataset=dataset, reference_git=__file__, splitting=custom_splitter,
                estimator=estimator, evaluator=evaluator,
                experiment_name="Iris KNN", project_name="Examples", ptype=DefaultPythonExperimentPipeline)
def experiment():
    return
