"""
This file shows an example on how to upload local experiment to server.
"""
from pypadre.ds_import import load_sklearn_toys
import pprint
from pypadre.core import Experiment


def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)


def create_preprocessing_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    estimators = [('PCA', PCA())]
    return Pipeline(estimators)


def split(idx):
    limit = int(.7 * len(idx))
    return idx[0:limit], idx[limit:], None


if __name__ == '__main__':
    from pypadre.app import p_app
    p_app.set_printer(print)
    ds = [i for i in load_sklearn_toys()][4]
    print(ds)
    experiment_name = "Test Experiment SVM upload local"
    ex = Experiment(name=experiment_name,
                    description="Testing Support Vector Machines via SKLearn Pipeline",
                    dataset=ds, preprocessing=create_preprocessing_pipeline(),
                    workflow=create_test_pipeline(), keep_splits=True, strategy="random",
                    function=split)
    conf = ex.configuration()
    pprint.pprint(ex.hyperparameters())
    ex.execute()

    # Authenticate and upload
    p_app.authenticate("hmafnan", "test")
    p_app.experiments.upload_local_experiment(experiment_name)