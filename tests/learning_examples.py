"""
This file tests the different learning methods and its parameters
"""
import copy
import pprint

import numpy as np
from sklearn import linear_model, decomposition, manifold
from sklearn.svm import SVC, SVR

from padre.ds_import import load_sklearn_toys
from padre.experiment import Experiment

# TODO: Method to log errors from outside the experiment class too

# TODO: This should mapped dynamically
# Default values of parameters are set here
# Use method set_parameters to set parameter values after initialization
workflow_dict = {'pca': decomposition.PCA(),
                 'logistic': linear_model.LogisticRegression(),
                 'SVC': SVC(probability=True),
                 'LSA': decomposition.TruncatedSVD(n_components=2),
                 'isomap':manifold.Isomap(n_neighbors=10, n_components=5),
                 'lle':manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=5,
                                                       method='standard'),
                 'SVR': SVR(kernel='rbf', degree=4,
                            gamma='auto',C=1.0)}
# This contains the all the classifiers.
classifier_list=(['SVC'])

# All parameters of estimators are to be listed here
# as a list under the key of the estimator name
# The parameters will be entered as a list to the key
# based on the estimator name
parameters = {'SVC': ['probability'],
              'LSA': ['n_components'],
              'isomap': ['n_neighbors', 'n_components'],
              'lle': ['n_neighbors', 'n_components', 'method'],
              'SVR': ['kernel', 'degree', 'gamma', 'C']}


# This function sets the parameters of the estimators and classifiers created
# in the pipeline. The param_val_dict contains the name of the parameter and
# the value to be set for that parameter
# The parameters are looked up from the parameters dictionary
# Parameters:
# estimator: The estimator whose parameters have to be modified
# estimator_name: The name of the estimator to look up whether,
#                 the parameters are present in that estimator
# param_val_dict: The dictionary containing the parameter names
#                 and their values
# Return Value: If successful, the modified estimator
#               Else, None
def set_parameters(estimator, estimator_name, param_val_dict):
    if estimator is None:
        print(estimator_name + ' does not exist in the workflow')
        return None
    available_params = parameters.get(estimator_name)
    for param in param_val_dict:
        if param not in available_params:
            print(param + ' is not present for estimator ' + estimator_name)
        else:
            estimator.set_params(**{param: param_val_dict.get(param)})

    return estimator


# This function creates the pipeline for the experiment
# The function looks up the name of the estimator passed and then
# deep copies the estimator to the pipeline
# Arguments:
# estimator_list: A list of strings containing all estimators to be used
#                 in the pipeline in exactly the order to be used
# param_value_dict: This contains the parameters for each of the estimators used
#                   The dict is a nested dictionary with the
#                   outer-most key being the estimator name
#                   The inner key is the parameter name
# Return Value: If successful, the Pipeline
#               Else None
def create_test_pipeline(estimator_list, param_value_dict=None):
    from sklearn.pipeline import Pipeline
    estimators = []
    for estimator_name in estimator_list:
        if workflow_dict.get(estimator_name) is None:
            print(estimator_name + ' not present in list')
            return None

        # Deep copy of the estimator because the estimator object is mutable
        estimator = copy.deepcopy(workflow_dict.get(estimator_name))
        estimators.append((estimator_name, estimator))
        if param_value_dict is not None and \
                param_value_dict.get(estimator_name) is not None:
            set_parameters(estimator, estimator_name, param_value_dict.get(estimator_name))

    return Pipeline(estimators)


# Initialization of the experiments dict
experiments_dict = dict()


# This function adds an experiment to the dictionary
# Parameters:
# name: Name of the experiment. It should be unique for this set of experiments
# description: The description of the experiment
# dataset: The dataset to be used for the experiment
# workflow: The scikit pipeline to be used for the experiment.
# backend: The backend of the experiment
def create_experiment(name, description,
                      dataset, workflow,
                      backend):
    if name is None or description is None or \
       dataset is None or workflow is None or backend is None:
        print('Missing values in experiment')
        return None

    # Classifiers cannot work on continous data and rejected as experiments
    if not np.all(np.mod(dataset.targets(), 1) == 0):
        for estimator in workflow.named_steps:
            if estimator in classifier_list:
                print('Estimator ' + estimator + ' cannot work on continous data')
                return None

    # Experiment name should be unique
    if experiments_dict.get(name, None) is None:
        data_dict = dict()
        data_dict['description'] = description
        data_dict['dataset'] = dataset
        data_dict['workflow'] = workflow
        data_dict['backend'] = backend
        experiments_dict[name] = data_dict

    else:
        print('Error creating experiment')
        if experiments_dict.get(name, None) is not None:
            print('Experiment name: ', name, ' already present. Experiment name should be unique')


# This function returns the dataset from pypadre
# This done by using the pre-defined names of the datasets
# Parameters:
# name: The name of the dataset
# Return_value: If successful, the dataset
#               Else, None
def get_local_dataset(name=None):
    local_dataset = ['Boston_House_Prices',
                     'Breast_Cancer',
                     'Diabetes',
                     'Digits',
                     'Iris',
                     'Linnerrud']
    if name is None:
        print('Dataset name is empty')
        return None
    if name in local_dataset:
        return [i for i in load_sklearn_toys()][local_dataset.index(name)]
    else:
        print(name + ' Local Dataset not found')
        return None


def main():
    # Get the dataset from load_sklearn_toys
    # The dataset index is 3
    from padre.app import pypadre
    pypadre.set_printer(print)
    # NOTE: Server MUST BE RUNNING!!! See Padre Server!
    # Start PADRE Server and run
    ds = None
    try:
        pypadre.datasets.list_datasets()
        ds = pypadre.datasets.get_dataset("http://localhost:8080/api/datasets/5")
    except:
        ds = [i for i in load_sklearn_toys()][2]

    # ex.report_results() # last step, but we can also look that up on the server

    # Testing Support Vector Classification using default values
    workflow = create_test_pipeline(['SVC'])
    create_experiment(name='Experiment1', description='This is the first test experiment',
                      dataset=get_local_dataset('Breast_Cancer'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Create another experiment which has a transformation and logistic regression
    workflow = create_test_pipeline(['pca', 'logistic'])
    create_experiment(name='Experiment2', description='This is the second test experiment',
                      dataset=get_local_dataset('Iris'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Experiment using SVD in the pipeline
    # Setting parameters for estimator 'LSA'/Truncated SVD
    params = {'n_components': 5}
    param_value_dict = {'LSA': params}
    workflow = create_test_pipeline(['LSA'], param_value_dict)
    create_experiment(name='Experiment3', description='This is the third test experiment',
                      dataset=get_local_dataset('Diabetes'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Testing isometric mapping with default values
    workflow = create_test_pipeline((['isomap']))
    create_experiment(name='Experiment4', description='This is the fourth test experiment',
                      dataset=get_local_dataset('Diabetes'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Testing setting parameters to isometric mapping
    params = {'n_neighbors': 10, 'n_components': 5}
    param_value_dict = {'isomap': params}
    create_experiment(name='Experiment5', description='This is the fifth test experiment',
                      dataset=get_local_dataset('Boston_house_price'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Testing the combination of PCA and Local Linear Embedding
    workflow = create_test_pipeline(['pca','lle'])
    create_experiment(name='Experiment6', description='This is the sixth test experiment',
                      dataset=get_local_dataset('Diabetes'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Testing Support Vector Regression on the Diabetes dataset
    workflow = create_test_pipeline(['SVR'])
    create_experiment(name='Experiment7', description='This is the seventh test experiment',
                      dataset=get_local_dataset('Diabetes'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Testing missing parameters/error cases when initializing the Experiment class
    # The aim of this is that, while running the experiment it should never crash
    # Missing name
    workflow = create_test_pipeline(['LSA'])
    create_experiment(name=None, description='This is the third test experiment',
                      dataset=get_local_dataset('Diabetes'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Missing description
    workflow = create_test_pipeline(['LSA'])
    create_experiment(name='Error2', description=None,
                      dataset=get_local_dataset('Diabetes'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Missing dataset
    workflow = create_test_pipeline(['LSA'])
    create_experiment(name='Error3', description='This experiment is missing the dataset',
                      dataset=get_local_dataset('Missing_dataset'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Missing workflow
    workflow = create_test_pipeline(['Wrong_classifier'])
    create_experiment(name='Error4 ', description='This experiment is missing a workflow',
                      dataset=get_local_dataset('Diabetes'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Missing backend
    workflow = create_test_pipeline(['LSA'])
    create_experiment(name='Error5 ', description='This experiment is missing a workflow',
                      dataset=get_local_dataset('Diabetes'), workflow=workflow,
                      backend=None)

    # Applying a classifer on a continous dataset
    workflow = create_test_pipeline(['SVC'])
    create_experiment(name='Error6', description='This experiment has a classifier/dataset mismatch',
                      dataset=get_local_dataset('Boston_House_Prices'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Applying a linear regressor and classifier
    workflow = create_test_pipeline(['linear', 'SVC'])
    create_experiment(name='Error7', description='This experiment has a regressor and classifier',
                      dataset=get_local_dataset('iris'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)
    # Run all the experiments in the list
    for experiment in experiments_dict:
        ex = Experiment(name=experiment,
                        description=experiments_dict.get(experiment).get('description'),
                        dataset=experiments_dict.get(experiment).get('dataset', None),
                        workflow=experiments_dict.get(experiment).get('workflow', None),
                        backend=experiments_dict.get(experiment).get('backend', None))
        conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
        pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
        ex.run()  # run the experiment and report

    print("========Available experiments=========")
    for idx, ex in enumerate(pypadre.experiments.list_experiments()):
        print("%d: %s" % (idx, str(ex)))
        for idx2, run in enumerate(pypadre.experiments.list_runs(ex)):
            print("\tRun: %s" % str(run))


if __name__ == '__main__':
    main()
