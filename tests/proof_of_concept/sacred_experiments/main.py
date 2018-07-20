from sacred import Experiment
from padre.ExperimentCreator import ExperimentCreator
from padre.app import pypadre

ex = Experiment('PyPaDRe')


@ex.config
def my_config():
    name = 'Test Experiment PCA Linear'
    description = 'Test Experiment with pca and linear regression'
    pipeline = ['principal component analysis', 'linear regression']
    dataset = 'Boston_House_Prices'
    params_linear_pca = {'num_components': [4, 5, 6, 7, 10]}
    params_dict_linear = {'principal component analysis': params_linear_pca}

    params = dict()
    params['name'] = name
    params['description'] = description
    params['pipeline'] = pipeline
    params['dataset'] = dataset
    params['estimator_params'] = params_dict_linear
    params['run_for_multiple_datasets'] = False


@ex.automain
def main(params):
    experiment_helper = ExperimentCreator()
    workflow = experiment_helper.create_test_pipeline(params.get('pipeline', None))

    #experiment_helper.set_param_values('Test Experiment PCA Linear',
    #                                   'principal component analysis.num_components:[4, 5, 6, 7, 10]|'
    #                                   'principal component analysis.whiten:[False, True]')

    experiment_helper.set_param_values(params.get('name', None), params.get('estimator_params', None))
    experiment_helper.create_experiment(name=params.get('name', None),
                                        description=params.get('description', None),
                                        dataset=params.get('dataset', None),
                                        workflow=workflow,
                                        backend=pypadre.file_repository.experiments)

    if params.get('run_for_multiple_datasets', False) is False:
        experiment_helper.execute_experiments()

    else:
        if params.get('experiment_datasets', None) is not None:
            experiment_helper.do_experiments(params.get('experiment_datasets'))


    '''
    experiment = ['Test Experiment PCA Linear', 'Test Experiment PCA Logistic']
    datasets = ['Diabetes', 'Boston_House_Prices']
    datasets_logistic = ['Iris', 'Digits', 'Breast_Cancer', 'Boston_House_Prices']
    experiment_datasets = {experiment[0]: datasets,
                           experiment[1]: datasets_logistic}
    experiment_helper.do_experiments(experiment_datasets)
    '''