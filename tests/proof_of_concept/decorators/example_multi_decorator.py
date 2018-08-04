"""
This file shows an example on how to use PyPaDRE via decorators defining multipe experiments.

Note: it is a proof of concept now rather than a test.
"""
# Note that we want to include all decorator at once using package import
from tests.proof_of_concept.decorators import *
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from padre.ds_import import load_sklearn_toys


@Workflow(exp_name="Test",
          description="Test experiment with decorators")
def create_test_pipeline():
    estimators = [('clf', SVC(probability=True))]
    return Pipeline(estimators)


@Workflow(exp_name="Test2",
          description="Test2 experiment with decorators")
def create_test_pipeline():
    estimators = [('clf', SVC(probability=True, C=0.2))]
    return Pipeline(estimators)


@Dataset(exp_name="Test")
def get_dataset():
    ds = [i for i in load_sklearn_toys()][2]
    return ds


@Dataset(exp_name="Test2")
def get_dataset():
    ds = [i for i in load_sklearn_toys()][3]
    return ds


if __name__ == '__main__':
    exs = run()  # run the experiment and report
    for ex in exs:
        for r in ex.runs:
            print (ex.name+": "+str(r))
