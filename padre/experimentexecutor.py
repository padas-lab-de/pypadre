"""
The purpose of this class is to provide a flexible execution method for the user for the experiments
The experiments could be run
1. Sequentially on the local machine
2. Parallelly on the local machine
3. Sequentially on the server
4. Parallelly on the server
5. Use a queueing system
"""
from padre.experiment import Experiment
from padre.ds_import import load_sklearn_toys
from padre.base import default_logger


class ExperimentExecutor:
    """
    The class provides the different method implementation for providing a flexible execution method of experiments.
    The user could select between running the experiment on the local machine or on the server, in parallel or
    sequentially
    """

    _experiments = None
    _local_dataset = []

    def __init__(self, **options):
        self._experiments = options.get('experiments', None)
        self._local_dataset = self.initialize_dataset_names()

    def initialize_dataset_names(self):
        """
        The function returns all the datasets currently available to the user. It can be from the server and also the
        local datasets availabe

        TODO: Dynamically populate the list of datasets

        :return: List of names of available datasets
        """
        dataset_names = ['Boston_House_Prices',
                         'Breast_Cancer',
                         'Diabetes',
                         'Digits',
                         'Iris',
                         'Linnerrud']

        return dataset_names

    def get_local_dataset(self, name=None):
        """
        This function returns the dataset from pypadre.
        This done by using the pre-defined names of the datasets defined in _local_dataset
        There is no error checking done here as it is assumed that the experiments passed here from ExperimentCreator
        class has already done all the error checking

        TODO: Datasets need to be loaded either from server or locally

        :param name: The name of the dataset

        :return: If successful, the dataset
                 Else, None
        """
        return [i for i in load_sklearn_toys()][self._local_dataset.index(name)]

    def execute(self, local_run: bool=True, threads: int=0):
        """
        This function decides the type of execution based on the user input.

        :param local_run: Whether the experiment is to be executed on the server or locally
        :param threads: Number of threads for executing the experiments. More than one implies a parallel execution

        :return: None
        """

        if local_run and threads < 2:
            print('Running experiments sequentially on the local machine')

        elif local_run and threads > 1:
            print('Running experiments parallelly on the local machine')

        elif not local_run and threads < 2:
            print('Running expeirments sequentially on the server')

        elif not local_run and threads > 1:
            print('Running experiments parallelly on the server')

    def standardExecution(self):
        import pprint
        for experiment_dict in self._experiments:
            name = experiment_dict.get('name')
            desc = experiment_dict.get('desc')
            dataset = self.get_local_dataset(experiment_dict.get('dataset'))
            workflow = experiment_dict.get('workflow')
            backend = experiment_dict.get('backend')
            strategy = experiment_dict.get('strategy', 'random')
            params = experiment_dict.get('params')

            ex = Experiment(name=name,
                            description=desc,
                            dataset=dataset,
                            workflow=workflow,
                            backend=backend,
                            strategy=strategy)
            conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline

            pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
            ex.grid_search(parameters=params)


    def runLocal(self):
        print('Parallel execution locally')

    def runOnServerSequential(self):
        print('Running on server sequentially')

    def runOnServerParallel(self):
        print('Running on server parallelly')

    def _put_to_queue(self):
        print('Add an experiment to the queue until the queue reaches the max size')