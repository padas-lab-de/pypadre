import os

from sklearn.datasets import load_iris
import numpy as np
from pypadre.pod.app import PadreConfig
from pypadre.pod.app.padre_app import PadreAppFactory


def gitlab_app():
    config_path = os.path.join(os.path.expanduser("~"), ".padre-gitlab.cfg")
    workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-gitlab")

    config = PadreConfig(config_file=config_path)
    config.set("backends", str([
        {
            "root_dir": workspace_path,
            "gitlab_url": 'http://gitlab.padre.backend:30080/',
            "user": "username",
            "token": "access_token"
        }
    ]))
    return PadreAppFactory.get(config)


app = gitlab_app()


@app.dataset(name="iris",
             columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                      'petal width (cm)', 'class'], target_features='class')
def dataset():
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)


@app.experiment(dataset=dataset,
                reference_package=__file__,
                experiment_name="Iris SVC", project_name="Examples")
def experiment():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)
