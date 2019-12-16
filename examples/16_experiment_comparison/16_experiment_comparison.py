import numpy as np
from pypadre.pod.app import PadreApp
from sklearn.datasets import load_iris

from pypadre.examples.base_example import example_app

# create example app
padre_app = example_app()


def create_experiment1(app: PadreApp, name="", project="", auto_main=True):
    @app.dataset(name="iris",
                 columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
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

    @app.experiment(dataset=dataset, reference_git=__file__, preprocessing_fn=preprocessing,
                    experiment_name=name, seed=1, project_name=project, auto_main=auto_main)
    def experiment():
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC
        estimators = [('SVC', SVC(probability=True, C=1.0))]
        return Pipeline(estimators)

    return experiment


def create_experiment2(app: PadreApp, name="", project="", auto_main=True):
    @app.dataset(name="iris",
                 columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                          'petal width (cm)', 'class'], target_features='class')
    def dataset():
        data = load_iris().data
        target = load_iris().target.reshape(-1, 1)
        return np.append(data, target, axis=1)

    @app.custom_splitter(reference_git=__file__)
    def custom_splitter(dataset, **kwargs):
        idx = np.arange(dataset.size[0])
        cutoff = int(len(idx) / 2)
        return idx[:cutoff], idx[cutoff:], None

    @app.experiment(dataset=dataset, reference_git=__file__, splitting=custom_splitter,
                    experiment_name=name, seed=1, project_name=project, auto_main=auto_main)
    def experiment():
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC
        from sklearn.decomposition import PCA
        estimators = [('PCA',PCA()),('SVC', SVC(probability=True, C=1.0))]
        return Pipeline(estimators)

    return experiment



experiment1 = create_experiment1(app=padre_app, name="Iris SVC - preprocessing", project="Iris - experiments")

experiment2 = create_experiment2(app=padre_app, name="Iris SVC - custom_splitting", project="Iris - experiments")

metadata, pipelines = experiment1.compare(experiment2)

print("Experiments metadata: ")
print(metadata)
print("Experiments pipelines: ")
print(pipelines)