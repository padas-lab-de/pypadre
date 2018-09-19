import copy
import pprint

from padre.experiment import Experiment

from padre.ExperimentCreator import ExperimentCreator

def main():
    from padre.app import pypadre
    pypadre.set_printer(print)
    experiment_param_dict = dict()
    # Experiment using SVD in the pipeline
    # Setting parameters for estimator 'LSA'/Truncated SVD
    experiment_helper = ExperimentCreator()
    params = {'n_neighbors': [2,8,10], 'n_components': [3,7]}
    param_value_dict = {'isomap embedding': params}
    workflow = experiment_helper.create_test_pipeline(['isomap embedding'])
    experiment_helper.create_experiment(name='Grid_search_experiment_1',
                                        description='This is the first grid search test experiment',
                                        dataset=experiment_helper.get_local_dataset('Boston_House_Prices'),
                                        workflow=workflow,
                                        backend=pypadre.file_repository.experiments)

    experiment_param_dict['Grid_search_experiment_1'] = experiment_helper.convert_alternate_estimator_names(param_value_dict)

    params_isomap = {'n_neighbors': [2, 8, 10], 'n_components': [3, 7]}
    params_pca = {'n_components': [4, 5]}
    param_value_dict['isomap embedding'] = params_isomap
    param_value_dict['pca'] = params_pca
    experiment_param_dict['Grid_search_experiment_2'] = experiment_helper.convert_alternate_estimator_names(param_value_dict)
    workflow = experiment_helper.create_test_pipeline(['pca', 'isomap embedding'])
    experiment_helper.create_experiment(name='Grid_search_experiment_2',
                                        description='This is the second grid search test experiment',
                                        dataset=experiment_helper.get_local_dataset('Boston_House_Prices'),
                                        workflow=workflow,
                                        backend=pypadre.file_repository.experiments)
    params_svc = {'C': [0.5, 1.0, 1.5],
                  'degree': [1,2,3,4],
                  'probability': [True]}
    params_ = {'SVC': params_svc}
    workflow = experiment_helper.create_test_pipeline(['SVC'])
    experiment_param_dict['Grid_search_experiment_3'] = experiment_helper.convert_alternate_estimator_names(params_)
    experiment_helper.create_experiment(name='Grid_search_experiment_3',
                                        description='Grid search experiment with SVC',
                                        dataset=experiment_helper.get_local_dataset('Iris'),
                                        workflow=workflow,
                                        backend=pypadre.file_repository.experiments
                                        )
    params_pca = {'n_components': [1, 2, 3, 4, 5, 6]}
    params_svr = {'C': [0.5, 1.0, 1.5],
                  'degree': [1, 2, 3]}
    params_dict_svr = {'SVR': params_svr, 'pca': params_pca}
    workflow = experiment_helper.create_test_pipeline(['pca', 'SVR'])
    experiment_param_dict['Grid_search_experiment_4'] = experiment_helper.convert_alternate_estimator_names(params_dict_svr)
    experiment_helper.create_experiment(name='Grid_search_experiment_4',
                                        description='Grid search experiment with SVR',
                                        dataset=experiment_helper.get_local_dataset('Diabetes'),
                                        workflow=workflow,
                                        backend=pypadre.file_repository.experiments
                                        )

    workflow = experiment_helper.create_test_pipeline(['pca', 'MaxEnt'])
    params_logistic_pca = {'n_components': [4, 5, 6, 7, 10]}
    params_dict_logistic = {'pca': params_logistic_pca}
    experiment_param_dict['Grid_search_experiment_5'] = experiment_helper.convert_alternate_estimator_names(params_dict_logistic)
    experiment_helper.create_experiment(name='Grid_search_experiment_5',
                                        description='Grid search experiment with logistic regression',
                                        dataset=experiment_helper.get_local_dataset('Diabetes'),
                                        workflow=workflow,
                                        backend=pypadre.file_repository.experiments)

    experiments_dict = experiment_helper.experiments
    # Run all the experiments in the list
    for experiment in experiments_dict:
        ex = Experiment(name=experiment,
                        description=experiments_dict.get(experiment).get('description'),
                        dataset=experiments_dict.get(experiment).get('dataset', None),
                        workflow=experiments_dict.get(experiment).get('workflow', None),
                        backend=experiments_dict.get(experiment).get('backend', None),
                        strategy=experiments_dict.get(experiment).get('strategy', 'random'),
                        keep_splits=True)
        conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
        pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
        ex.grid_search(parameters=experiment_param_dict.get(experiment))  # run the experiment and report
        pprint.pprint(ex.results)


if __name__ == '__main__':
    main()

