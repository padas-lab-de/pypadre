from pypadre.ds_import import load_sklearn_toys
from pypadre import *
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


@Workflow(exp_name="Test",
          description="Test experiment with decorators")
def create_test_pipeline():
    estimators = [('clf', SVC(probability=True))]
    return Pipeline(estimators)


@Dataset(exp_name="Test")
def get_dataset():
    ds = [i for i in load_sklearn_toys()][2]
    return ds
