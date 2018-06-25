"""
This file will be used to test the functionalities of the ExperimentCreator class and the metrics class
"""
from padre.ds_import import load_sklearn_toys
from padre.experiment import Experiment
from padre.app import pypadre
import copy
from padre.ExperimentCreator import ExperimentCreator

def main():
    experiment_helper = ExperimentCreator()
    print(experiment_helper.get_dataset_names())
    workflow = experiment_helper.create_test_pipeline(['principal component analysis', 'linear regression'])
    params_linear_pca = {'num_components': [4, 5, 6, 7, 10]}
    params_dict_linear = {'principal component analysis': params_linear_pca}
    experiment_helper.set_param_values('Test Experiment PCA Linear', params_dict_linear)
    experiment_helper.create_experiment(name='Test Experiment PCA Linear',
                                        description='Test Experiment with pca and linear regression',
                                        dataset=None,
                                        workflow=workflow,
                                        backend=pypadre.file_repository.experiments)

    workflow = experiment_helper.create_test_pipeline(['logistic regression'])
    params_logistic_pca = {'penalty_norm': ['l1', 'l2']}
    params_dict_logistic = {'logistic regression': params_logistic_pca}
    experiment_helper.set_param_values('Test Experiment PCA Logistic', params_dict_logistic)
    experiment_helper.create_experiment(name='Test Experiment PCA Logistic',
                                        description='Test Experiment with pca and logistic regression',
                                        dataset='Boston_House_Prices',
                                        workflow=workflow,
                                        backend=pypadre.file_repository.experiments)
    experiment_helper.execute_experiments()

    experiment = ['Test Experiment PCA Linear', 'Test Experiment PCA Logistic']
    datasets = ['Diabetes', 'Boston_House_Prices']
    datasets_logistic = ['Iris', 'Digits', 'Breast_Cancer', 'Boston_House_Prices']
    experiment_datasets = {experiment[0]:datasets,
                           experiment[1]:datasets_logistic}
    experiment_helper.do_experiments(experiment_datasets)


if __name__ == '__main__':
    main()