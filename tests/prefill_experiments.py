"""
This file aims to create multiple different experiments to prefill the server.
The experiments work on the sklearn toy datasets and are a combination of classification and regression tasks
Some experiments will also contain a grid search(testing multiple hyperparameters)
"""
from padre.experimentexecutor import ExperimentExecutor
from padre.experimentcreator import ExperimentCreator
from padre.eventhandler import trigger_event


def main():
    trigger_event('EVENT_LOG_EVENT', source='Experiment_Executor_Test',
                  message='Experiment Executor Starting')
    experiment_creator = ExperimentCreator()

    # A decision tree classifier with Iris and Digits datasets and multiple parameter assignments
    workflow = experiment_creator.create_test_pipeline(['decision tree classifier'])
    params = {'max_depth_tree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_samples_split': [2, 3, 4, 5, 6, 7]}
    param_value_dict = {'decision tree classifier': params}
    experiment_creator.create(name='Executor Test 1',
                              description='This is the first executor search test experiment',
                              dataset_list=['Iris', 'Digits'],
                              workflow=workflow,
                              params=param_value_dict)

    # SECOND TEST EXPERIMENT WITH SINGLE DATASET
    params = {'num_neighbours': [1, 4, 5], 'num_components': [4, 5, 6]}
    param_value_dict = {'isomap embedding': params}
    workflow = experiment_creator.create_test_pipeline(['SVM'])
    experiment_creator.create(name='Executor Test 2',
                              description='This is the second executor search test experiment',
                              dataset_list=['Iris'],
                              workflow=workflow,
                              params=param_value_dict)

    '''
    # THIRD TEST EXPERIMENT WITH MULTIPLE DATASETS
    params_pca = {'num_components': [2, 3, 4, 5, 6]}
    params_svr = {'C': [0.5, 1.0, 1.5],
                  'poly_degree': [1, 2, 3]}
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
    '''
    experiments_list = experiment_creator.createExperimentList()
    experiments_executor = ExperimentExecutor(experiments=experiments_list)
    import time
    c1 = time.time()
    experiments_executor.execute(local_run=True, threads=1)
    c2 = time.time()
    print('Execution time:{time_diff}'.format(time_diff=c2 - c1))

    from padre.metrics import CompareMetrics
    metrics = CompareMetrics(experiments_list=experiments_executor.experiments)
    print(metrics.show_metrics())


if __name__ == '__main__':
    main()