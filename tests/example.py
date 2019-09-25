"""
This file shows an example on how to use the pypadre app.
"""
import pprint

from pypadre.core.model.experiment import Experiment
from pypadre.pod.importing.dataset.ds_import import load_sklearn_toys


def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)


def create_preprocessing_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    estimators = [('PCA', PCA())]
    return Pipeline(estimators)


def split(idx):
    # Do a 70:30 split
    limit = int(.7 * len(idx))
    return idx[0:limit], idx[limit:], None


if __name__ == '__main__':
    from pypadre.app import p_app
    p_app.set_printer(print)

    # NOTE: Server MUST BE RUNNING!!! See Padre Server!
    # Start PADRE Server and run
    ds = None
    try:
        p_app.datasets.list()
        ds = p_app.datasets.get_dataset("http://localhost:8080/api/datasets/5")
    except:
        ds = [i for i in load_sklearn_toys()][4]

    if ds is None:
        ds = [i for i in load_sklearn_toys()][4]
    print(ds)
    ex = Experiment(name="Test Experiment SVM",
                    description="Testing Support Vector Machines via SKLearn Pipeline",
                    dataset=ds,
                    workflow=create_test_pipeline(), keep_splits=True, strategy="random", random_seed=0,
                    function=split, preprocessing=create_preprocessing_pipeline())
    conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
    pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
    ex.execute()  # run the experiment and report

    p_app.metrics_evaluator.add_experiments([ex, ex])
    print(p_app.metrics_evaluator.show_metrics())
    p_app.metrics_evaluator.add_experiments('Test Experiment SVM')
    print(p_app.metrics_evaluator.show_metrics())
    '''
    print("========Available experiments=========")
    for idx, ex in enumerate(p_app.experiments.list_experiments()):
        print("%d: %s" % (idx, str(ex)))
        for idx2, run in enumerate(p_app.experiments.list_runs(ex)):
            print("\tRun: %s"%str(run))
    '''
    # ex.report_results() # last step, but we can also look that up on the server

