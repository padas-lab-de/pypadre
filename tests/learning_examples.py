"""
This file tests the different learning methods and its parameters
"""
import pprint

from sklearn import linear_model, decomposition
from sklearn.svm import SVC

from padre.ds_import import load_sklearn_toys
from padre.experiment import Experiment


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


# TODO: This should mapped dynamically
# TODO: Parameters should be accepted to the workflow
workflow_dict = {'pca': decomposition.PCA(),
                 'logistic': linear_model.LogisticRegression(),
                 'SVC': SVC(probability=True),
                 'LSA': decomposition.TruncatedSVD(n_components=2)}

parameters = {'SVC': ['probability'],
              'LSA': ['n_components']}


# This function sets the parameters of the estimators and classifiers created
# in the pipeline. The param_val_dict contains the name of the parameter and
# the value to be set for that parameter
# The parameters are looked up from the parameters dictionary
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


def create_test_pipeline(estimator_list, param_value_dict=None):
    from sklearn.pipeline import Pipeline
    estimators = []
    # estimators=[('pca', decomposition.PCA()), ('logistic', linear_model.LogisticRegression())]
    for estimator_name in estimator_list:
        if workflow_dict.get(estimator_name) is None:
            print(estimator_name + ' not present in list')
            return None
        estimator = workflow_dict.get(estimator_name)
        estimators.append((estimator_name, estimator))
        if param_value_dict is not None and \
                param_value_dict.get(estimator_name) is not None:
            set_parameters(estimator, estimator_name, param_value_dict.get(estimator_name))

    return Pipeline(estimators)


experiments_dict = dict()


def create_experiment(name, description,
                      dataset, workflow,
                      backend):
    # Experiment name should be unique
    if experiments_dict.get(name, None) is None and \
        not(name is None or description is None or dataset is None
            or workflow is None or backend is None):
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

    # Create a sample experiment
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
    # setting parameters for estimator 'LSA'
    params = {'n_components': 5}
    param_value_dict = {'LSA': params}
    workflow = create_test_pipeline(['LSA'], param_value_dict)
    #set_parameters(workflow,['LSA'],{'n_components':5})
    create_experiment(name='Experiment3', description='This is the third test experiment',
                      dataset=get_local_dataset('Diabetes'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Testing missing parameters when initializing the Experiment class

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
                      dataset=get_local_dataset('MISSING'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Missing workflow
    workflow = create_test_pipeline(['ERROR'])
    create_experiment(name='Error4 ', description='This experiment is missing a workflow',
                      dataset=get_local_dataset('Diabetes'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Missing backend
    workflow = create_test_pipeline(['LSA'])
    create_experiment(name='Error1 ', description='This experiment is missing a workflow',
                      dataset=get_local_dataset('Diabetes'), workflow=workflow,
                      backend=None)

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
