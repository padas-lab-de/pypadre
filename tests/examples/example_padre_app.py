"""
This example is to demonstrate the capabilities and functionalities of the Padre App.
"""
'''
from padre.app import pypadre
from padre.ds_import import load_sklearn_toys
from padre import *
from padre.app import pypadre
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from padre.app import pypadre

@Workflow(exp_name="Decorated Experiment",
          description="Test experiment with decorators and SVC")
def create_test_pipeline():
    estimators = [('clf', SVC(probability=True))]
    return Pipeline(estimators)


@Dataset(exp_name="Test")
def get_dataset():
    ds = [i for i in load_sklearn_toys()][2]
    return ds

def main():

    ex = pypadre.experiments.run(decorated=True)
    ex = run("Test")  # run the experiment and report


if __name__ == '__main__':
    main()
'''
"""
This file shows an example on how to use PyPaDRE via decorators defining a single experiments.

Note: it is a proof of concept now rather than a test.
"""
from padre.app import pypadre
# Note that we want to include all decorator at once using package import
from padre import *
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from padre.app import pypadre
from padre.ds_import import load_sklearn_toys

@Workflow(exp_name="Test",
          description="Test experiment with decorators")
def create_test_pipeline():
    estimators = [('clf', SVC(probability=True))]
    return Pipeline(estimators)


@Dataset(exp_name="Test")
def get_dataset():
    ds = [i for i in load_sklearn_toys()][2]
    return ds

if __name__ == '__main__':
    # call run without pypadre app backend
    '''
    ex = run("Test")
    print("Runs retained in memory ")
    for r in ex:
        print(r)
    '''
    # call via pipadre backend
    ex = pypadre.experiments.run(decorated=True)
    ex = run("Test")  # run the experiment and report