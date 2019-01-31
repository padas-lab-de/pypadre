from padre.core import Experiment, Run
from padre.base import default_logger
from padre.eventhandler import assert_condition, add_logger
from padre.app import pypadre
import numpy as np
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

class TestRun(unittest.TestCase):

    def test_run_constructor_no_experiment(self):
        self.assertRaises(ValueError, Run, experiment=None, workflow=create_test_pipeline(), options=dict())

    def test_run_constructor_invalid_experiment_type(self):
        self.assertRaises(ValueError, Run, experiment=3.141592653, workflow=create_test_pipeline(), options=dict())

    def test_run_constructor_no_workflow(self):
        experiment = Experiment(name='Test_Run_No_Workflow',
                                description="Testing run class without workflow",
                                workflow=create_test_pipeline(), dataset=get_toy_dataset())
        self.assertRaises(ValueError, Run, experiment=experiment, workflow=None, options=dict())

    def test_run_constructor_invalid_workflow_type(self):
        experiment = Experiment(name='Test_Run_Invalid_Type_Workflow',
                                description="Testing run class with invalid workflow type",
                                workflow=create_test_pipeline(), dataset=get_toy_dataset())
        self.assertRaises(ValueError, Run, experiment=experiment, workflow=2.718281, options=dict())

    def test_run_constructor_no_options(self):
        experiment = Experiment(name='Test_Run_No_Options',
                                description="Testing run class with no options",
                                workflow=create_test_pipeline(), dataset=get_toy_dataset())
        self.assertRaises(ValueError, Run, experiment=experiment, workflow=create_test_pipeline(), options=None)

    def test_run_constructor_invalid_option_type(self):
        experiment = Experiment(name='Test_Run_Invalid_Type_Options',
                                description="Testing run class with invalid type for options",
                                workflow=create_test_pipeline(), dataset=get_toy_dataset())
        self.assertRaises(ValueError, Run, experiment=experiment, workflow=create_test_pipeline(), options=True)

