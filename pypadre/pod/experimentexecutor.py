"""
The purpose of this class is to provide a flexible execution method for the user for the experiments
The experiments could be run
1. Sequentially on the local machine
2. Parallelly on the local machine
3. Sequentially on the server
4. Parallelly on the server
5. Use a queueing system
"""
import threading
import time
from copy import deepcopy

from sklearn.externals.joblib import Parallel, delayed

from pypadre.core.model.experiment import Experiment
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.pod.importing.dataset.ds_import import load_sklearn_toys
from pypadre.core.events import trigger_event, assert_condition

EXPERIMENT_EXECUTION_QUEUE = []


class ExecutionThread (threading.Thread):

    def __init__(self, threadID, q, threadCount, queueLock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.q = q
        self.queueLock = queueLock
        self.threadCount = threadCount
        self._experiments = []

    def run(self):
        process_queue(self.q, self.queueLock, self.threadID)


def process_queue(q,  queueLock, threadID):
    import pprint
    continue_process = True
    while continue_process:
        queueLock.acquire()
        if q.qsize() > 0:
            idx = q.get()
            queueLock.release()

            dict_object = EXPERIMENT_EXECUTION_QUEUE[idx]

            ex = deepcopy(dict_object.get('experiment', None))
            params = deepcopy(dict_object.get('params', None))
            name = deepcopy(dict_object.get('name'))
            print('Executing experiment: {name} with thread: {threadID}'.format(name=name, threadID=threadID))
            c1 = time.time()
            conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
            pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
            ex.execute(parameters=params)
            c2 = time.time()
            print('Completed experiment: {name} with thread: {threadID}. '
                  'Execution time: {time_diff}'.format(name=name, threadID=threadID, time_diff=c2-c1))

            #queueLock.release()

        else:
            continue_process = False
            queueLock.release()


def run_experiment(experiment, params):
    conf = experiment.configuration()
    experiment.execute(parameters=params)


def run_workflow(workflow, dataset):
    x = dataset.features()
    y = dataset.targets()
    y = y.reshape(y.shape[0])
    return workflow.fit(x, y)


def run_idx(idx):
    import pprint

    dict_object = EXPERIMENT_EXECUTION_QUEUE[idx]

    ex = dict_object.get('experiment', None)
    params = dict_object.get('params', None)
    name = dict_object.get('name')
    print('Executing experiment: {name}'.format(name=name))
    c1 = time.time()
    conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
    pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
    ex.execute(parameters=params)
    c2 = time.time()
    print('Completed experiment: {name}. '
          'Execution time: {time_diff}'.format(name=name, time_diff=c2 - c1))


def run(experiment_object):
    c1 = time.time()
    ex = experiment_object.get('experiment')
    name = experiment_object.get('name')
    params = experiment_object.get('params')
    ex.execute(parameters=params)
    c2 = time.time()
    print('Completed experiment: {name}. Execution time: {time_diff}'.format(name=name,
                                                                                                     time_diff=c2 - c1))


class ExperimentExecutor:
    """
    The class provides the different method implementation for providing a flexible execution method of experiments.
    The user could select between running the experiment on the local machine or on the server, in parallel or
    sequentially
    """

    _experiments = None
    _local_dataset = []
    _experiment_objects = None

    def __init__(self, **options):
        experiments = options.get('experiments', None)
        assert_condition(condition=
                         isinstance(experiments, list) or isinstance(experiments, dict) or experiments is None,
                         source=self, message='Wrong parameter type provided')
        if self._experiments is None:
            self._experiments = []
        for experiment in experiments:
            assert_condition(condition=isinstance(experiment, dict), source=self,
                             message='Experiment should be an instance of class experiment')
            self._experiments.append(experiment)

        self._experiment_objects = []
        self._local_dataset = self.initialize_dataset_names()

    def add_experiments(self, experiments=None):
        """
        Add experiments to the experiment executor class
        :param experiments: Experiments to be added for execution. Could be a list or an experiment class object
        :return: None
        """
        assert_condition(condition=experiments is not None, source=self, message='Experiments parameter cannot be None')
        assert_condition(condition=isinstance(experiments, list) or isinstance(experiments, Experiment),
                         source=self, message='Wrong parameter type provided')

        if isinstance(experiments, Experiment):
            experiments = [experiments]

        for experiment in experiments:
            assert_condition(condition=isinstance(experiment, dict), source=self,
                             message='Experiment should be an instance of class experiment')
            self._experiments.append(experiment)

    def remove_experiments(self):
        """
        Remove all the experiments present in the executor class
        :return:
        """
        # Remove experiment configurations
        self._experiments = None

        # Remove experiment objects
        self._experiment_objects = None

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
            print('Sequential Execution')
            self.standardExecution()

        elif local_run and threads > 1:
            self.runLocal(threads)

        elif not local_run and threads < 2:
            print('Running experiments sequentially on the server')

        elif not local_run and threads > 1:
            print('Running experiments parallelly on the server')

    def standardExecution(self):
        """
        This function executes the experiments sequentially and on a single thread

        :return: None
        """
        import pprint
        limit = len(self._experiments)
        curr_experiment = 0
        for experiment_dict in self._experiments:
            curr_experiment += 1
            trigger_event('EVENT_LOG_EXPERIMENT_PROGRESS', curr_value=curr_experiment, limit=limit, phase='start')
            name = experiment_dict.get('name')
            dataset = experiment_dict.get('dataset')
            assert_condition(condition=isinstance(dataset, str) or isinstance(dataset, Dataset), source=self,
                             message='Dataset is of incorrect parameter type')

            if isinstance(dataset, str):
                dataset = self.get_local_dataset(experiment_dict.get('dataset'))


            exp_params = deepcopy(experiment_dict)
            exp_params['dataset'] = dataset
            params = exp_params.pop('params', None)
            preprocessing_params = exp_params.pop('preprocessing_params',None)
            trigger_event('EVENT_LOG', message='Executing experiment: {name}'.format(name=name), source=self)
            c1 = time.time()
            ex = Experiment(**exp_params)
            conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline

            pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
            ex.execute(parameters=params,pre_parameters=preprocessing_params)
            self._experiment_objects.append(ex)
            c2 = time.time()
            trigger_event('EVENT_LOG',
                          message = 'Completed experiment: {name} with execution time: '
                                    '{time_diff}'.format(name=name, time_diff=c2-c1), source=self)
            trigger_event('EVENT_LOG_EXPERIMENT_PROGRESS', curr_value=curr_experiment, limit=limit, phase='stop')

    def runLocal(self, threadCount:int = 1):
        """
        This method executes the different experiments in parallel based on the thread count parameter

        :param threadCount: Number of jobs to be run in parallel

        :return: None
        """
        from multiprocessing import Queue

        # Create the parallel execution queue of experiments
        experiment_queue = Queue(len(self._experiments))

        # Create a threading lock object
        queueLock = threading.RLock()

        # Fill the Queue with the experiments
        queueLock.acquire()
        for idx in range(len(self._experiments)):
            experiment_queue.put(idx)
        queueLock.release()

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
            queue_dict = dict()
            queue_dict['name'] = name
            queue_dict['experiment'] = ex
            queue_dict['params'] = params
            EXPERIMENT_EXECUTION_QUEUE.append(deepcopy(queue_dict))

        array_idx = range(0, len(self._experiments))
        jobs = (delayed(run_idx)(idx) for idx in array_idx)
        parallel = Parallel(n_jobs=threadCount)
        results = parallel(jobs)

        '''
        # Create a list to store the threads and an ID starting from 1
        threads = []
        threadID = 0

        for tName in range(threadCount):
            thread = ExecutionThread(threadID, experiment_queue,threadCount=threadCount, queueLock=queueLock)
            thread.start()
            threads.append(thread)
            threadID += 1

        # Wait for queue to empty
        # The empty function is not used since it gave inconsistent results on a test experiment
        while experiment_queue.qsize() > 0:
            time.sleep(0.1)
            pass

        # Wait for all threads to complete
        for t in threads:
            t.join()
        '''

    def runOnServerSequential(self):
        """
        This method will run the experiment sequentially  on the server

        :return:
        """
        print('Running on server sequentially')

    def runOnServerParallel(self):
        """
        This method will distribute the experiment across multiple threads on a server

        :return:
        """
        print('Running on server parallelly')

    @property
    def experiments(self):
        return self._experiment_objects



