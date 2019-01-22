from padre.core import Experiment
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

    def test_experiment_constructor_validate_type_experiment_name(self):
        self.assertRaises(ValueError, Experiment, name=5.55555,
                          description='Experiment with wrong type in Experiment name',
                          workflow=create_test_pipeline(), dataset=get_toy_dataset())

    def test_experiment_constructor_validate_type_dataset(self):
        self.assertRaises(ValueError, Experiment, name='Wrong Dataset Type',
                          description='Experiment with wrong dataset type',
                          workflow=create_test_pipeline(), dataset=np.zeros([50,50]))

    def test_experiment_execute_validate_type_parameters(self):
        ex = Experiment(name='Test_Incorrect_Execute_Parameter_Type',
                        description="Experiment with incorrect datatype for execute parameters",
                        workflow=create_test_pipeline(),
                        dataset=get_toy_dataset())
        self.assertRaises(ValueError, ex.execute, parameters=100)

    def test_experiment_execute_validate_parameters(self):
        ex = Experiment(name='Test_Incorrect_Execute_Parameters',
                        description="Experiment with incorrect execute parameters",
                        workflow=create_test_pipeline(),
                        dataset=get_toy_dataset())
        parameters = dict()
        parameters['greeting'] = 'hello'
        parameters['good'] = {'night':'morning'}
        self.assertRaises(ValueError, ex.execute, parameters=parameters)

    def test_experiment_execute_validate_incorrect_estimator(self):
        ex = Experiment(name='Test_Incorrect_Estimator',
                        description="Experiment with incorrect estimator in parameters",
                        workflow=create_test_pipeline(),
                        dataset=get_toy_dataset())
        parameters = dict()
        parameters['good'] = {'night': 'morning'}
        self.assertRaises(ValueError, ex.execute, parameters=parameters)

