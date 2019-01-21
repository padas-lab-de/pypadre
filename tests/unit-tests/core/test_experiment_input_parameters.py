from padre.core import Experiment
from padre.base import default_logger
from padre.eventhandler import assert_condition, add_logger
from padre.app import pypadre
import unittest

default_logger.backend = pypadre.repository
add_logger(default_logger)

def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    estimators = [('clf', SVC(probability=True))]
    return Pipeline(estimators)

def get_toy_dataset():
    from padre.ds_import import load_sklearn_toys
    return [i for i in load_sklearn_toys()][2]


class TestExperiment(unittest.TestCase):

    def test_experiment_constructor_no_workflow(self):
        self.assertRaises(ValueError, Experiment, name='No workflow',
                          description='Test Experiment without any workflow', dataset=get_toy_dataset())

    def test_experiment_constructor_no_dataset(self):
        self.assertRaises(ValueError, Experiment, name='No dataset',
                          description='Test Experiment without a dataset', workflow=create_test_pipeline())

    def test_experiment_constructor_no_name(self):
        self.assertTrue(Experiment(description='Test Experiment without any name',
                                   workflow=create_test_pipeline(), dataset=get_toy_dataset()), None)

    def test_experiment_constructor_no_description(self):
        self.assertRaises(ValueError, Experiment, name='No description',
                          workflow=create_test_pipeline(), dataset=get_toy_dataset())

