from pypadre.core.model.experiment import Experiment
import numpy as np
import unittest


def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)


def get_toy_dataset_classification():
    from pypadre.pod.importing.dataset.ds_import import load_sklearn_toys
    return [i for i in load_sklearn_toys()][1]


def get_toy_dataset_regression():
    from pypadre.pod.importing.dataset.ds_import import load_sklearn_toys
    return [i for i in load_sklearn_toys()][0]


class TestExperiment(unittest.TestCase):

    def test_experiment_constructor_no_workflow(self):
        self.assertRaises(ValueError, Experiment, name='Test_No_Workflow',
                          description='Test Experiment without any workflow', dataset=get_toy_dataset_classification())

    def test_experiment_constructor_no_dataset(self):
        self.assertRaises(ValueError, Experiment, name='Test_No_dataset',
                          description='Test Experiment without a dataset', workflow=create_test_pipeline())

    def test_experiment_constructor_no_name(self):
        self.assertTrue(Experiment(description='Test Experiment without any name',
                                   workflow=create_test_pipeline(), dataset=get_toy_dataset_classification()), None)

    def test_experiment_constructor_no_description(self):
        self.assertRaises(ValueError, Experiment, name='Test_No_Description',
                          workflow=create_test_pipeline(), dataset=get_toy_dataset_classification())

    def test_experiment_constructor_validate_type_experiment_name(self):
        self.assertRaises(ValueError, Experiment, name=5.55555,
                          description='Experiment with incorrect type in Experiment name',
                          workflow=create_test_pipeline(), dataset=get_toy_dataset_classification())

    def test_experiment_constructor_validate_type_dataset(self):
        self.assertRaises(ValueError, Experiment, name='Test_Incorrect_Dataset_Type',
                          description='Experiment with incorrect dataset type',
                          workflow=create_test_pipeline(), dataset=np.zeros([50,50]))

    def test_experiment_constructor_validate_type_keep_runs(self):
        self.assertRaises(ValueError, Experiment, name='Test_Incorrect_Parameter_Type_keep_runs',
                          description='Experiment with incorrect parameter type for keep_runs',
                          workflow=create_test_pipeline(), dataset=get_toy_dataset_classification(),
                          keep_runs=3.141592653)

    def test_experiment_constructor_validate_type_keep_splits(self):
        self.assertRaises(ValueError, Experiment, name='Test_Incorrect_Parameter_Type_keep_splits',
                          description='Experiment with incorrect parameter type for keep_splits',
                          workflow=create_test_pipeline(), dataset=get_toy_dataset_classification(),
                          keep_splits=2.71828158284)

    def test_experiment_constructor_validate_workflow(self):
        self.assertRaises(ValueError, Experiment, name='Test_Incorrect_Workflow',
                          description='Experiment with incorrect workflow type',
                          workflow=dict(), dataset=get_toy_dataset_classification())

    def test_experiment_execute_validate_type_parameters(self):
        ex = Experiment(name='Test_Incorrect_Execute_Parameter_Type',
                        description="Experiment with incorrect datatype for execute parameters",
                        workflow=create_test_pipeline(),
                        dataset=get_toy_dataset_classification())
        self.assertRaises(ValueError, ex.execute, parameters=100)

    def test_experiment_execute_validate_parameters(self):
        ex = Experiment(name='Test_Incorrect_Execute_Parameters',
                        description="Experiment with incorrect execute parameters",
                        workflow=create_test_pipeline(),
                        dataset=get_toy_dataset_classification())
        parameters = dict()
        parameters['greeting'] = 'hello'
        parameters['good'] = {'night':'morning'}
        self.assertRaises(ValueError, ex.execute, parameters=parameters)

    def test_experiment_execute_validate_incorrect_estimator(self):
        ex = Experiment(name='Test_Incorrect_Estimator',
                        description="Experiment with incorrect estimator in parameters",
                        workflow=create_test_pipeline(),
                        dataset=get_toy_dataset_classification())
        parameters = dict()
        parameters['good'] = {'night': 'morning'}
        self.assertRaises(ValueError, ex.execute, parameters=parameters)

    def test_experiment_continous_data_with_classifier_estimator(self):
        self.assertRaises(ValueError, Experiment, name='Test_Regression_dataset_with_classification_estimator',
                          description='Experiment with a regression dataset passed to a classification estimator',
                          workflow=create_test_pipeline(), dataset=get_toy_dataset_regression())

    def test_experiment_one_row_dataset(self):
        from pypadre.pod.importing.dataset.ds_import import load_pandas_df
        import pandas as pd
        # Creating a random dataset
        data = np.random.random_sample((1, 11))
        df = pd.DataFrame(data)
        ds = load_pandas_df(df)
        self.assertRaises(ValueError, Experiment, name='Test_Single_Row_Dataset',
                          description='Experiment with a single row as dataset',
                          workflow=create_test_pipeline(),
                          dataset=ds)




