import configparser
import os
import shutil
import unittest
import gitlab

from pypadre.core.model.code.icode import Function
from pypadre.pod.app import PadreConfig
from pypadre.pod.app.padre_app import PadreAppFactory
from pypadre.pod.importing.dataset.dataset_import import SKLearnLoader
from pypadre.pod.tests.util.util import connect_log_to_stdout, connect_event_to_stdout, create_sklearn_test_pipeline

# config_path = os.path.join(os.path.expanduser("~"), ".padre_git_test.cfg")
# workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-test")

class PadreGitTest(unittest.TestCase):

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
                        "user" : "root",
                        "token" : "LvVzAaNyFyS6iiJNzTFf"
                    }
                ]))
        cls.app = PadreAppFactory.get(config)
        connect_log_to_stdout()
        connect_event_to_stdout()

    def create_experiment(self, *args, **kwargs):
        return self.app.experiments.service.create(*args, **kwargs)

    def create_project(self, *args, **kwargs):
        return self.app.projects.service.create(*args, **kwargs)

    def create_execution(self, *args, **kwargs):
        return self.app.executions.service.create(*args, **kwargs)

    def create_run(self, *args, **kwargs):
        return self.app.runs.service.create(*args, **kwargs)

    def create_split(self, *args, **kwargs):
        return self.app.splits.service.create(*args, **kwargs)

    def setUp(self):
        # clean up if last teardown wasn't called correctly
        self.tearDown()

    def tearDown(self):
        # delete local data content
        try:
            if os.path.exists(os.path.join(self.workspace_path, "datasets")):
                shutil.rmtree(self.workspace_path+"/datasets")
            if os.path.exists(os.path.join(self.workspace_path, "projects")):
                shutil.rmtree(self.workspace_path+"/projects")
            if os.path.exists(os.path.join(self.workspace_path, "code")):
                shutil.rmtree(self.workspace_path+"/code")
        except FileNotFoundError:
            pass

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls):
        """Remove config file after test"""
        if os.path.isdir(cls.workspace_path):
            shutil.rmtree(cls.workspace_path)
        os.remove(cls.config_path)


class GitlabBackend(PadreGitTest):

    def tearDown(self):

        super().tearDown()

        server = gitlab.Gitlab(url='http://gitlab.padre.backend:30080/',private_token="LvVzAaNyFyS6iiJNzTFf")

        projects = server.projects.list()

        for project in projects:
            server.projects.delete(project.get_id())

        server.__exit__()

    def test_code(self):
        def foo(ctx):
            return "foo"

        foo_code = self.app.code.create(clz=Function, fn=foo)
        self.app.code.put(foo_code, store_code=True)
        code_list = self.app.code.list()
        loaded_code = code_list.pop()

        out = loaded_code.call()
        assert out is "foo"

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

        project = Project(name='Test Project', description='Testing the functionalities of project backend')
        self.app.projects.put(project)
        name = 'Test Project'
        projects = self.app.projects.list({'name': name})
        for project in projects:
            assert name in project.name

    def test_experiment(self):
        from pypadre.core.model.experiment import Experiment
        project = self.create_project(name='Test Project 2',
                                      description='Testing the functionalities of project backend')
        self.app.projects.put(project)

        # self.app.datasets.load_defaults()
        # dataset = self.app.datasets.list({'name': '_boston_dataset'})
        dataset = SKLearnLoader().load("sklearn", utility="load_boston")

        from sklearn.svm import SVC
        experiment = self.create_experiment(name='Test Experiment SVM', description='Testing experiment using SVM',
                                            dataset=dataset, project=project,
                                            pipeline=create_sklearn_test_pipeline(
                                                estimators=[('SVC', SVC(probability=True))]))

        self.app.experiments.put(experiment)

        name = 'Test Experiment SVM'
        experiments = self.app.experiments.list({'name': name})

        assert (isinstance(experiments, list))

        assert name in [ex.name for ex in experiments]
