"""
This file runs a digits classification based on the dataset
present in scikit-learn datasets.
This tests the PCA data transformation and LogisticRegression functions
The example is taken from <http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html#sphx-glr-auto-examples-plot-digits-pipe-py>
"""
import pprint

from pypadre.pod.importing.dataset.ds_import import load_sklearn_toys
from pypadre.core.model.experiment import Experiment


def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn import linear_model, decomposition
    #estimators = [('clf', svm.SVC(probability=True, gamma=0.001))]
    estimators=[('pca', decomposition.PCA()), ('logistic', linear_model.LogisticRegression())]
    return Pipeline(estimators)

def main():

    # Get the dataset from load_sklearn_toys
    # The dataset index is 3
    from pypadre.app import p_app
    p_app.set_printer(print)
    # NOTE: Server MUST BE RUNNING!!! See Padre Server!
    # Start PADRE Server and run
    ds = None
    try:
        p_app.datasets.list()
        ds = p_app.datasets.get_dataset("http://localhost:8080/api/datasets/5")
    except:
        ds = [i for i in load_sklearn_toys()][2]

    print(ds)
    ex = Experiment(name="Digits Recognition",
                    description="Testing Support Vector Classification via SKLearn Pipeline",
                    dataset=ds,
                    workflow=create_test_pipeline())

    conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
    pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
    ex.run()  # run the experiment and report
    print("========Available experiments=========")
    for idx, ex in enumerate(p_app.experiments.list_experiments()):
        print("%d: %s" % (idx, str(ex)))
        for idx2, run in enumerate(p_app.experiments.list_runs(ex)):
            print("\tRun: %s" % str(run))

    # ex.report_results() # last step, but we can also look that up on the server


if __name__ == '__main__':
    main()