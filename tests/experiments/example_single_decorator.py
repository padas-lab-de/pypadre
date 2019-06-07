"""
This file shows an example on how to use PyPaDRE via decorators defining a single experiments.

Note: it is a proof of concept now rather than a test.
"""
# Note that we want to include all decorator at once using package import
from pypadre import *
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from pypadre.app import p_app
from pypadre.ds_import import load_sklearn_toys

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
    ex = run("Test")
    print("Runs retained in memory ")
    for r in ex:
        print(r)
    # call via pipadre backend
    ex = p_app.experiments.run(decorated=True)
    ex = run("Test")  # run the experiment and report
    print("Runs stored on disk via padre app")
    for e in ex:
        print ("ex: %s"%str(e.name))
        for idx2, run in enumerate(p_app.experiments.list_runs(e.name)):
            print("\tRun: %s" % str(run))
