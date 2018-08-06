"""
This file shows an example on how to use PyPaDRE via decorators defining multipe experiments.

Note: it is a proof of concept now rather than a test.
"""
# Note that we want to include all decorator at once using package import
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from padre import *
from padre.app import pypadre
from padre.ds_import import load_sklearn_toys


@Workflow(exp_name="AppExpTest",
          description="Test experiment with decorators")
def create_test_pipeline():
    estimators = [('clf', SVC(probability=True))]
    return Pipeline(estimators)


@Workflow(exp_name="AppExpTest2",
          description="Test2 experiment with decorators")
def create_test_pipeline():
    estimators = [('clf', SVC(probability=True, C=0.2))]
    return Pipeline(estimators)


@Dataset(exp_name="AppExpTest")
def get_dataset():
    ds = [i for i in load_sklearn_toys()][2]
    return ds


@Dataset(exp_name="AppExpTest2")
def get_dataset():
    ds = [i for i in load_sklearn_toys()][3]
    return ds


if __name__ == '__main__':
    # delete experiments
    ex = pypadre.experiments.run(decorated=True)
    # check if experiments exist

