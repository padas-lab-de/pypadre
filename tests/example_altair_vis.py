"""
Example showing how to use the altair visualisation together with the pypadre framework.
"""
from padre import *
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from padre.app import pypadre
from padre.ds_import import load_sklearn_toys

@Workflow(exp_name="VisTest",
          description="Test experiment with decorators",
          hyperparameters={
              "c_val": [0.1, 0.2],
              "kernel": ["linear"]
          })
def create_test_pipeline(c_val, kernel):
    estimators = [('clf', SVC(probability=True, C=c_val, kernel=kernel))]
    return Pipeline(estimators)


@Dataset(exp_name="VisTest")
def get_dataset():
    ds = [i for i in load_sklearn_toys()][2]
    return ds

def run_experiment():
    ex = pypadre.experiments.run(decorated=True)

if __name__ == '__main__':

    runs = pypadre.experiments.list_experiments("VisTest.*")
    if len(runs) == 0:
        run_experiment()
        runs = pypadre.experiments.list_runs("VisTest")

    print("\n".join(runs))