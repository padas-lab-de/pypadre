"""
This file runs a digits classification based on the dataset
present in scikit-learn datasets.
This tests the PCA data transformation and LogisticRegression functions
The example is taken from <http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html#sphx-glr-auto-examples-plot-digits-pipe-py>
"""
import pprint

from padre.ds_import import load_sklearn_toys
from padre.core import Experiment


def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn import linear_model, decomposition
    #estimators = [('clf', svm.SVC(probability=True, gamma=0.001))]
    estimators=[('pca', decomposition.PCA()), ('logistic', linear_model.LogisticRegression())]
    return Pipeline(estimators)

def main():

    # Get the dataset from load_sklearn_toys
    # The dataset index is 3
    from padre.app import pypadre
    pypadre.set_printer(print)
    # NOTE: Server MUST BE RUNNING!!! See Padre Server!
    # Start PADRE Server and run
    ds = None
    try:
        pypadre.datasets.list()
        ds = pypadre.datasets.get_dataset("http://localhost:8080/api/datasets/5")
    except:
        ds = [i for i in load_sklearn_toys()][2]

    print(ds)
    ex = Experiment(name="Digits Recognition",
                    description="Testing Support Vector Classification via SKLearn Pipeline",
                    dataset=ds,
                    workflow=create_test_pipeline(),
                    backend=pypadre.local_backend.experiments)

    conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
    pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
    ex.run()  # run the experiment and report
    print("========Available experiments=========")
    for idx, ex in enumerate(pypadre.experiments.list_experiments()):
        print("%d: %s" % (idx, str(ex)))
        for idx2, run in enumerate(pypadre.experiments.list_runs(ex)):
            print("\tRun: %s" % str(run))

    # ex.report_results() # last step, but we can also look that up on the server


if __name__ == '__main__':
    main()