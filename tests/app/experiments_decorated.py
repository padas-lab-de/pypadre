"""
the module defines two fast experiments under the name test_AppExp and test_AppExp2.

The test can be used by importing thsi module and running pypadre.experiments.run(decorated=ture)
```
from tests.app.experiments_decorated import *
ex = p_app.experiments.run(decorated=True)
```
"""
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from pypadre import *
from pypadre.pod.importing.dataset.ds_import import load_sklearn_toys


@Workflow(exp_name="test_AppExp",
          description="Test experiment with decorators")
def create_test_pipeline():
    estimators = [('clf', SVC(probability=True))]
    return Pipeline(estimators)


@Workflow(exp_name="test_AppExp2",
          description="Test2 experiment with decorators")
def create_test_pipeline():
    estimators = [('clf', SVC(probability=True, C=0.2))]
    return Pipeline(estimators)


@Dataset(exp_name="test_AppExp")
def get_dataset():
    ds = [i for i in load_sklearn_toys()][2]
    return ds


@Dataset(exp_name="test_AppExp2")
def get_dataset():
    ds = [i for i in load_sklearn_toys()][3]
    return ds

