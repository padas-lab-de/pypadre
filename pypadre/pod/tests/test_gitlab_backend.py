import configparser
import os

import gitlab

from pypadre.pod.app import PadreConfig
from pypadre.pod.app.padre_app import PadreAppFactory
from pypadre.pod.tests.base_test import PadreAppTest
from pypadre.pod.tests.util.util import connect_log_to_stdout, connect_event_to_stdout, create_sklearn_test_pipeline


# config_path = os.path.join(os.path.expanduser("~"), ".padre_git_test.cfg")
# workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-test")

class PadreGitTest(PadreAppTest):

    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(os.path.expanduser("~"), ".padre-git-test.cfg")
        cls.workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-git-test")

        """Create config file for testing purpose"""
        config = configparser.ConfigParser()
        # test_data = {'test_key': 'value 1', 'key2': 'value 2'}
        # cls.config['TEST'] = test_data
        with open(cls.config_path, 'w+') as configfile:
            config.write(configfile)

        config = PadreConfig(config_file=cls.config_path)
        config.set("backends", str([
            {
                "root_dir": cls.workspace_path,
                "gitlab_url": 'http://gitlab.padre.backend:30080/',
                "user": "root",
                "token": "LvVzAaNyFyS6iiJNzTFf"
            }
        ]))
        cls.app = PadreAppFactory.get(config)
        connect_log_to_stdout()
        connect_event_to_stdout()


class GitlabBackend(PadreGitTest):

    def setUp(self):
        super().setUp()
        self.setup_reference(__file__)

    def tearDown(self):

        super().tearDown()

        server = gitlab.Gitlab(url='http://gitlab.padre.backend:30080/', private_token="LvVzAaNyFyS6iiJNzTFf")

        projects = server.projects.list()

        for project in projects:
            server.projects.delete(project.get_id())

        server.__exit__()

    def test_dataset(self):
        # TODO test putting, fetching, searching, folder/git structure, deletion, git functionality?

        # Puts all the datasets
        self.app.datasets.load_defaults()

        # Gets a dataset by name
        id = '_boston_dataset'
        datasets = self.app.datasets.list({'name': id})
        for dataset in datasets:
            assert id in dataset.name

    def test_project(self):
        from pypadre.core.model.project import Project

        project = Project(name='Test Project', description='Testing the functionalities of project backend',
                          reference=self.test_reference)
        self.app.projects.put(project)

        name = 'Test Project'
        projects = self.app.projects.list({'name': name})
        for project in projects:
            assert name in project.name

    def test_experiment(self):
        project = self.create_project(name='Test Project 2',
                                      description='Testing the functionalities of project backend')
        self.app.projects.put(project)

        self.app.datasets.load_defaults()
        dataset = self.app.datasets.list({'name': '_boston_dataset'})

        from sklearn.svm import SVC
        experiment = self.create_experiment(name='Test Experiment SVM', description='Testing experiment using SVM',
                                            dataset=dataset.pop(), project=project,
                                            pipeline=create_sklearn_test_pipeline(
                                                estimators=[('SVC', SVC(probability=True))],
                                                reference=self.test_reference))

        self.app.experiments.put(experiment)
        name = 'Test Experiment SVM'
        experiments = self.app.experiments.list()
        assert (isinstance(experiments, list))
        assert name in [ex.name for ex in experiments]

    def test_execution(self):
        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.execution import Execution

        self.app.datasets.load_defaults()
        id = '_boston_dataset'
        dataset = self.app.datasets.list({'name': id})

        project = Project(name='Test Project 2', description='Testing the functionalities of project backend',
                          reference=self.test_reference)
        self.app.projects.put(project)

        from sklearn.svm import SVC
        experiment = Experiment(name="Test Experiment SVM",
                                description="Testing Support Vector Machines via SKLearn Pipeline",
                                dataset=dataset[0],
                                pipeline=create_sklearn_test_pipeline(estimators=[('SVC', SVC(probability=True))],
                                                                      reference=self.test_reference),
                                keep_splits=True, strategy="random", project=project,
                                reference=self.test_reference)
        self.app.experiments.put(experiment)

        codehash = 'abdauoasg45qyh34t'
        execution = Execution(experiment, codehash=codehash, append_runs=True, parameters=None,
                              preparameters=None, single_run=True,
                              single_transformation=True)
        self.app.executions.patch(execution)

        executions = self.app.executions.list()
        for execution_ in executions:
            assert codehash in execution_.hash
        if len(executions) > 0:
            execution = self.app.executions.get(executions.__iter__().__next__().id)
            assert execution[0].name == executions[0].name

    def test_run(self):

        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.execution import Execution
        from pypadre.core.model.computation.run import Run

        # TODO clean up testing (tests should be independent and only test the respective functionality)
        self.app.datasets.load_defaults()
        id = '_boston_dataset'
        dataset = self.app.datasets.list({'name': id})

        project = Project(name='Test Project 2', description='Testing the functionalities of project backend',
                          reference=self.test_reference)
        self.app.projects.put(project)

        from sklearn.svm import SVC
        experiment = Experiment(name="Test Experiment SVM",
                                description="Testing Support Vector Machines via SKLearn Pipeline",
                                dataset=dataset[0],
                                pipeline=create_sklearn_test_pipeline(
                                    estimators=[('SVC', SVC(probability=True))],
                                    reference=self.test_reference), keep_splits=True, strategy="random",
                                project=project, reference=self.test_reference)
        self.app.experiments.put(experiment)

        codehash = 'abdauoasg45qyh34t'
        execution = Execution(experiment, codehash=codehash, append_runs=True, parameters=None,
                              preparameters=None, single_run=True,
                              single_transformation=True)
        self.app.executions.put(execution)

        executions = self.app.executions.list({'hash': codehash})
        if len(executions) > 0:
            run = Run(execution=executions[0], pipeline=execution.experiment.pipeline, keep_splits=True)
            self.app.runs.put(run)

        else:
            raise ValueError('Execution not listed for the same code has')

        runs = self.app.runs.list(None)

        assert (isinstance(runs, list))
        for run in runs:
            assert (isinstance(run, Run))

    def test_split(self):
        from pypadre.core.model.split.split import Split

        self.app.datasets.load_defaults()
        project = self.app.projects.service.create(name='Test Project 2',
                                                   description='Testing the functionalities of project backend')

        id = '_iris_dataset'
        dataset = self.app.datasets.list({'name': id})

        from sklearn.svm import SVC
        experiment = self.create_experiment(name="Test Experiment SVM",
                                            description="Testing Support Vector Machines via SKLearn Pipeline",
                                            dataset=dataset[0],
                                            pipeline=create_sklearn_test_pipeline(
                                                estimators=[('SVC', SVC(probability=True))],
                                                reference=self.test_reference), keep_splits=True,
                                            strategy="random", project=project)

        codehash = 'abdauoasg45qyh34t'
        execution = self.create_execution(experiment, codehash=codehash, append_runs=True,
                                          parameters=None,
                                          preparameters=None, single_run=True,
                                          single_transformation=True)

        run = self.create_run(execution=execution, pipeline=execution.experiment.pipeline, keep_splits=True)
        train_range = list(range(1, 1000 + 1))
        test_range = list(range(1000, 1100 + 1))
        split = self.create_split(run=run, num=0, train_idx=train_range, val_idx=None,
                                  test_idx=test_range, keep_splits=True)
        assert (isinstance(split, Split))
        assert (split.test_idx == test_range)
        assert (split.train_idx == train_range)
        self.app.splits.put(split)
        splits = self.app.splits.list({"id": split.id})

        assert split.id in [item.id for item in splits]

    def test_full_stack(self):
        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.binding.metrics import sklearn_metrics
        print(sklearn_metrics)

        self.app.datasets.load_defaults()
        project = Project(name='Test Project 2',
                          description='Testing the functionalities of project backend',
                          reference=self.test_reference)

        id = '_iris_dataset'
        dataset = self.app.datasets.list({'name': id})

        from sklearn.svm import SVC
        experiment = Experiment(name="Test Experiment SVM",
                                description="Testing Support Vector Machines via SKLearn Pipeline",
                                dataset=dataset[0],
                                pipeline=create_sklearn_test_pipeline(
                                    estimators=[('SVC', SVC(probability=True))],
                                    reference=self.test_reference),
                                project=project,
                                reference=self.test_reference)

        experiment.execute()
        assert (experiment.executions is not None)
        computations = self.app.computations.list()
        assert (isinstance(computations, list))
        assert (len(computations) > 0)
        experiments = self.app.experiments.list()
        assert (isinstance(experiments, list))
        assert (len(experiments) > 0)

    def test_all_functionalities_classification(self):

        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.binding.metrics import sklearn_metrics
        print(sklearn_metrics)
        # TODO plugin system

        self.app.datasets.load_defaults()
        project = Project(name='Test Project 2',
                          description='Testing the functionalities of project backend',
                          reference=self.test_reference)

        def find(name, path):
            for root, dirs, files in os.walk(path):
                if name in files:
                    return os.path.join(root, name)

        _id = '_iris_dataset'
        dataset = self.app.datasets.list({'name': _id})

        from sklearn.svm import SVC
        experiment = Experiment(name='Test Experiment', description='Test Experiment',
                                dataset=dataset.pop(), project=project,
                                pipeline=create_sklearn_test_pipeline(estimators=[('SVC', SVC(probability=True))],
                                                                      reference=self.test_reference),
                                reference=self.test_reference)

        parameter_dict = {'SVC': {'C': [0.1, 0.2]}}
        experiment.execute(parameters={'SKLearnEvaluator': {'write_results': True},
                                       'SKLearnEstimator': {'parameters': parameter_dict}
                                       })



        files_found = find('results.bin', os.path.expanduser('~/.pypadre-git-test/projects/Test Project 2/'
                                                             'experiments/Test Experiment/executions'))
        assert (files_found is not None)

        files_found = find('initial_hyperparameters.json',
                           os.path.expanduser('~/.pypadre-git-test/projects/Test Project 2/'
                                              'experiments/Test Experiment/executions'))
        assert (files_found is not None)

        files_found = find('parameters.json', os.path.expanduser('~/.pypadre-git-test/projects/Test Project 2/'
                                                                 'experiments/Test Experiment/executions'))
        assert (files_found is not None)
