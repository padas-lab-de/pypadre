from pypadre.pod.experimentexecutor import ExperimentExecutor
from pypadre.pod.experimentcreator import ExperimentCreator
from pypadre.pod.eventhandler import trigger_event


def split(idx):
    # Do a 70:30 split
    limit = int(.7 * len(idx))
    return idx[0:limit], idx[limit:], None


def main():

    trigger_event('EVENT_LOG_EVENT', source='Experiment_Executor_Test',
                  message='Experiment Executor Starting')
    experiment_creator = ExperimentCreator()

    # FIRST TEST EXPERIMENT WITH MULTIPLE DATASETS
    params = {'num_neighbours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'num_components': [2, 3, 4, 5, 6, 7]}
    param_value_dict = {'isomap embedding': params}
    workflow = experiment_creator.create_test_pipeline(['isomap embedding'])
    preprocessing = experiment_creator.create_test_pipeline(['PCA'])
    experiment_creator.create(name='Executor Test 1',
                              description='This is the first executor search test experiment',
                              dataset_list=['Diabetes', 'Boston_House_Prices', 'Iris'],
                              workflow=workflow,
                              keep_splits=True,
                              params=param_value_dict,
                              preprocessing=preprocessing)

    # SECOND TEST EXPERIMENT WITH SINGLE DATASET
    params = {'num_neighbours': [1, 4, 5], 'num_components': [4, 5, 6]}
    param_value_dict = {'isomap embedding': params}
    workflow = experiment_creator.create_test_pipeline(['isomap embedding'])
    experiment_creator.create(name='Executor Test 2',
                              description='This is the second executor search test experiment',
                              dataset_list=['Iris'],
                              workflow=workflow,
                              strategy='function',
                              function=split,
                              params=param_value_dict)

    # THIRD TEST EXPERIMENT WITH MULTIPLE DATASETS
    params_pca = {'num_components': [2, 3, 4, 5, 6]}
    params_svr = {'C': [0.5, 1.0, 1.5],
                  'poly_degree': [1, 2, 3],
                  'tolerance': [1,3]}
    params_dict = {'SVR': params_svr, 'pca': params_pca}
    workflow = experiment_creator.create_test_pipeline(['pca', 'SVR'])
    params_dict = experiment_creator.convert_alternate_estimator_names(params_dict)
    experiment_creator.create(name='Executor Test 3',
                              description='Grid search experiment with SVR',
                              dataset_list=['Boston_House_Prices', 'Diabetes', 'Digits'],
                              workflow=workflow,
                              params=params_dict
                              )

    experiment_creator.parse_config_file('experiment.json')
    experiment_creator.clear_experiments('Executor Test 3(Diabetes)')

    experiments_list = experiment_creator.createExperimentList()
    experiments_executor = ExperimentExecutor(experiments=experiments_list)
    import time
    c1 = time.time()
    experiments_executor.execute(local_run=True, threads=1)
    c2 = time.time()
    print('Execution time:{time_diff}'.format(time_diff=c2-c1))

    from pypadre.pod.metrics import CompareMetrics
    metrics = CompareMetrics(experiments_list=experiments_executor.experiments)
    print(metrics.show_metrics())


if __name__ == '__main__':
    main()
