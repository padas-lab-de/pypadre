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
    params_logistic_pca = {'num_components': [4, 5, 6, 7, 10]}
    params_dict_logistic = {'principal component analysis': params_logistic_pca}
    experiment_helper.set_param_values('Test Experiment PCA Linear', params_dict_logistic)
    experiment_helper.create_experiment(name='Test Experiment PCA Linear',
                                        description='Test Experiment with pca and logistic regression',
                                        dataset='Diabetes',
                                        workflow=workflow,
                                        backend=pypadre.file_repository.experiments)
    experiment_helper.execute_experiments()

    experiment = ['Test Experiment PCA Linear']
    datasets = ['Diabetes', 'Boston_House_Prices']
    experiment_datasets = {experiment[0]:datasets}
    experiment_helper.do_experiments(experiment_datasets)



if __name__ == '__main__':
    main()