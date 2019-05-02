from padre.app import pypadre
from padre.ds_import import load_sklearn_toys
from padre.core import Experiment
from padre.base import PadreLogger
from padre.eventhandler import add_logger

def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)

def main():

    logger = PadreLogger()
    logger.backend = pypadre.repository
    add_logger(logger=logger)
    ds = [i for i in load_sklearn_toys()][2]
    ex = Experiment(name="Test Experiment SVM",
                        description="Testing Support Vector Machines via SKLearn Pipeline",
                        dataset=ds,
                        workflow=create_test_pipeline(), keep_splits=True, strategy="random")
    ex.execute()


if __name__ == '__main__':
    main()