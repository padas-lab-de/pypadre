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
import threading
import time
from copy import deepcopy


class ExecutionThread (threading.Thread):

    def __init__(self, threadID, q, threadCount, queueLock, global_queue):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.q = q
        self.queueLock = queueLock
        self.threadCount = threadCount
        self.global_queue = global_queue

    def run(self):
        process_queue(self.q, self.queueLock, self.threadID)

def process_queue(q,  queueLock, threadID):
    import pprint
    continue_process = True
    while continue_process:
        queueLock.acquire()
        if q.qsize() > 0:
            dict_object = q.get()
            ex = dict_object.get('experiment', None)
            params = dict_object.get('params', None)
            name = dict_object.get('name')
            queueLock.release()

            print('Executing experiment: {name} with thread: {threadID}'.format(name=name, threadID=threadID))
            c1 = time.time()
            conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
            pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
            ex.grid_search(parameters=params)
            c2 = time.time()
            print('Completed experiment: {name} with thread: {threadID}. Execution time: {time_diff}'.format(name=name,
                                                                                                             threadID=
                                                                                                             threadID,
                                                                                                             time_diff=
                                                                                                             c2-c1))

        else:
            continue_process = False
            queueLock.release()

def run_experiment(experiment, params):
    conf = experiment.configuration()
    experiment.grid_search(parameters=params)

def run_workflow(workflow, dataset):
    x = dataset.features()
    y = dataset.targets()
    y = y.reshape(y.shape[0])
    workflow.fit(x, y)

def run(experiment_object):
    c1 = time.time()
    ex = experiment_object.get('experiment')
    name = experiment_object.get('name')
    params = experiment_object.get('params')
    ex.grid_search(parameters=params)
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
            print('Sequential Execution')
            self.standardExecution()

        elif local_run and threads > 1:
            self.runLocal(threads)

        elif not local_run and threads < 2:
            print('Running experiments sequentially on the server')

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

            print('Executing experiment: {name}'.format(name=name))
            c1 = time.time()
            ex = Experiment(name=name,
                            description=desc,
                            dataset=dataset,
                            workflow=workflow,
                            backend=backend,
                            strategy=strategy)
            conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline

            pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
            ex.grid_search(parameters=params)
            c2 = time.time()
            print('Completed experiment: {name} with execution time: {time_diff}'.format(name=name, time_diff=c2-c1))

    def runLocal(self, threadCount:int = 1):
        from multiprocessing import Queue


        # Create the parallel execution queue of experiments
        experiment_queue = Queue(len(self._experiments))

        # Create a threading lock object
        queueLock = threading.Lock()

        global_queue = []

        # Fill the Queue with the experiments
        queueLock.acquire()
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
            queue_dict['workflow'] = workflow
            queue_dict['dataset'] = dataset
            experiment_queue.put(deepcopy(queue_dict))
            global_queue.append(deepcopy(queue_dict))
        queueLock.release()

        # Create a list to store the threads and an ID starting from 1
        threads = []
        threadID = 0

        for tName in range(threadCount):
            thread = ExecutionThread(threadID, experiment_queue,threadCount=threadCount, queueLock=queueLock, global_queue = global_queue)
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

    def runOnServerSequential(self):
        print('Running on server sequentially')

    def runOnServerParallel(self):
        print('Running on server parallelly')

    def _put_to_queue(self):
        print('Add an experiment to the queue until the queue reaches the max size')


