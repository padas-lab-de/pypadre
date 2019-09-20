from pypadre.app import p_app
from pypadre.pod.importing.dataset.ds_import import load_sklearn_toys
from pypadre.core.model.experiment import Experiment
from pypadre.pod.base import PadreLogger
from pypadre.core.events import add_logger


def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)


def main():

    logger = PadreLogger()
    logger.backend = p_app.repository
    add_logger(logger=logger)
    ds = [i for i in load_sklearn_toys()][2]
    ex = Experiment(name="Test Experiment SVM",
                        description="Testing Support Vector Machines via SKLearn Pipeline",
                        dataset=ds,
                        workflow=create_test_pipeline(), keep_splits=True, strategy="random")
    ex.execute()


if __name__ == '__main__':
    main()