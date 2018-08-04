"""
This file shows an example on how to use the pypadre app via decorates.

Note: it is a proof of concept now rather than a test.
"""
from padre.app import pypadre
# Note that we want to include all decorator at once using package import
from tests.proof_of_concept.decorators import *
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from padre.app import pypadre
from padre.ds_import import load_sklearn_toys

@Workflow(exp_name="Test",
          description="Test experiment with decorators")
def create_test_pipeline():
    estimators = [('clf', SVC(probability=True))]
    return Pipeline(estimators)


@Dataset(exp_name="Test",
         backend=pypadre.file_repository.experiments)
# Note that actually putting the backend here is not good style
# It should only demonstrate, that parameters for the Experiment can be provided via the decorators
# in fact, the padre app should have a run method where the backend parameters are set automatically
def get_dataset():
    ds = [i for i in load_sklearn_toys()][2]
    return ds

if __name__ == '__main__':
    ex = run("Test")  # run the experiment and report
    print("Runs retained in memory ")
    for r in ex.runs:
        print(r)

    print("Runs on disk via padre app")
    for idx2, run in enumerate(pypadre.experiments.list_runs(ex.name)):
        print("\tRun: %s" % str(run))
