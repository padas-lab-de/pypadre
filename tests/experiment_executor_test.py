from padre.experimentexecutor import ExperimentExecutor
from padre.experimentcreator import ExperimentCreator
from padre.app import pypadre

def main():
    experiment_creator = ExperimentCreator()

    # FIRST TEST EXPERIMENT
    params = {'num_neighbours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'num_components': [2, 3, 4, 5, 6, 7]}
    param_value_dict = {'isomap embedding': params}
    workflow = experiment_creator.create_test_pipeline(['isomap embedding'])
    experiment_creator.create_experiment(name='Executor Test 1',
                                         description='This is the first executor search test experiment',
                                         dataset_list=['Boston_House_Prices'],
                                         workflow=workflow,
                                         backend=pypadre.file_repository.experiments,
                                         params=param_value_dict)

    # SECOND TEST EXPERIMENT
    params = {'num_neighbours': [1, 4, 5], 'num_components': [4, 5, 6]}
    param_value_dict = {'isomap embedding': params}
    workflow = experiment_creator.create_test_pipeline(['isomap embedding'])
    experiment_creator.create_experiment(name='Executor Test 2',
                                         description='This is the second executor search test experiment',
                                         dataset_list=['Boston_House_Prices'],
                                         workflow=workflow,
                                         backend=pypadre.file_repository.experiments,
                                         params=param_value_dict)

    experiments_list = experiment_creator.createExperimentList()
    experiments_executor = ExperimentExecutor(experiments=experiments_list)
    experiments_executor.execute(local_run=True, threads=2)


if __name__ == '__main__':
    main()
