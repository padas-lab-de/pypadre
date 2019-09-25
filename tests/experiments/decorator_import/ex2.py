from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from pypadre import *
from pypadre.pod.importing.dataset.ds_import import load_sklearn_toys


@Workflow(exp_name="Test2",
          description="Test2 experiment with decorators")
def create_test_pipeline():
    estimators = [('clf', SVC(probability=True, C=0.2))]
    return Pipeline(estimators)


@Dataset(exp_name="Test2")
def get_dataset():
    ds = [i for i in load_sklearn_toys()][3]
    return ds