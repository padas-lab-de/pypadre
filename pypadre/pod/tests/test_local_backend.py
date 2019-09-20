import os
import shutil
import unittest

from pypadre.app import PadreConfig
from pypadre.app.padre_app import PadreFactory

config_path = os.path.join(os.path.expanduser("~"), ".padre-test.cfg")
workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-test")


class LocalBackends(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(LocalBackends, self).__init__(*args, **kwargs)
        config = PadreConfig(config_file=config_path)
        config.set("backends", str([
                    {
                        "root_dir": workspace_path
                    }
                ]))
        self.app = PadreFactory.get(config)

    def tearDown(self):
        # delete data content
        try:
            shutil.rmtree(workspace_path+"/datasets")
            shutil.rmtree(workspace_path+"/projects")
        except FileNotFoundError:
            pass

    @classmethod
    def tearDownClass(cls):
        # delete configuration
        if os.path.isdir(workspace_path):
            shutil.rmtree(workspace_path)
        os.remove(config_path)

    def test_dataset(self):
        # TODO test putting, fetching, searching, folder/git structure, deletion, git functionality?

        # Puts all the datasets
        self.app.datasets.load_defaults()

        # Gets a dataset by name
        id = '_boston_dataset'
        datasets = self.app.datasets.list({'name':id})
        for dataset in datasets:
            assert id in dataset.name

    def test_project(self):
        from pypadre.core.model.project import Project

        project = Project(name='Test Project', description='Testing the functionalities of project backend')
        self.app.projects.patch(project)
        name = 'Test Project'
        projects = self.app.projects.list({'name': name})
        for project in projects:
            assert name in project.name

    def test_experiment(self):
        # TODO test putting, fetching, searching, folder/git structure, deletion, git functionality?

        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.project import Project

        project = Project(name='Test Project 2', description='Testing the functionalities of project backend')
        self.app.projects.put(project)

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        self.app.datasets.load_defaults()
        id = '_boston_dataset'
        dataset = self.app.datasets.list({'name': id})

        experiment = Experiment(name="Test Experiment SVM",
                    description="Testing Support Vector Machines via SKLearn Pipeline",
                    dataset=dataset[0],
                    workflow=create_test_pipeline(), keep_splits=True, strategy="random", project=project,
                                path=os.path.abspath('.'))

        self.app.experiments.patch(experiment)
        name = 'Test Experiment SVM'
        experiments = self.app.experiments.list({'name': name})
        for experiment in experiments:
            assert name in experiment.name

    def test_execution(self):

        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.execution import Execution

        self.app.datasets.load_defaults()
        id = '_boston_dataset'
        dataset = self.app.datasets.list({'name': id})

        project = Project(name='Test Project 2', description='Testing the functionalities of project backend')
        self.app.projects.put(project)

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        experiment = Experiment(name="Test Experiment SVM",
                                description="Testing Support Vector Machines via SKLearn Pipeline",
                                dataset=dataset[0],
                                workflow=create_test_pipeline(), keep_splits=True, strategy="random", project=project)
        self.app.experiments.put(experiment)

        codehash = 'abdauoasg45qyh34t'
        execution = Execution(experiment, codehash=codehash, command=None, append_runs=True, parameters=None,
                              preparameters=None, single_run=True,
                              single_transformation=True)
        self.app.executions.patch(execution)

        executions = self.app.executions.list({'name': codehash})
        for execution_ in executions:
            assert codehash in execution_.name

        execution = self.app.executions.get(executions.__iter__().__next__().id)
        assert execution[0].name == executions[0].name

    def test_run(self):
        """
        project_backend: ProjectFileRepository = self.backend.project
        experiment_backend: ExperimentFileRepository = project_backend.experiment
        execution_backend: ExecutionFileRepository = experiment_backend.execution
        run_backend: RunFileRepository = execution_backend.run
        """

        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.execution import Execution
        from pypadre.core.model.run import Run

        # TODO clean up testing (tests should be independent and only test the respective functionality)
        self.app.datasets.load_defaults()
        id = '_boston_dataset'
        dataset = self.app.datasets.list({'name': id})

        project = Project(name='Test Project 2', description='Testing the functionalities of project backend')
        self.app.projects.put(project)

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        experiment = Experiment(name="Test Experiment SVM",
                                description="Testing Support Vector Machines via SKLearn Pipeline",
                                dataset=dataset[0],
                                workflow=create_test_pipeline(), keep_splits=True, strategy="random", project=project)
        self.app.experiments.put(experiment)

        codehash = 'abdauoasg45qyh34t'
        execution = Execution(experiment, codehash=codehash, command=None, append_runs=True, parameters=None,
                              preparameters=None, single_run=True,
                              single_transformation=True)
        self.app.executions.put(execution)

        executions = self.app.executions.list({'name': codehash})

        run = Run(execution=executions[0], workflow=execution.experiment.workflow, keep_splits=True)
        self.app.runs.put(run)

        runs = self.app.runs.list(None)

    def test_split(self):

        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.execution import Execution
        from pypadre.core.model.run import Run
        from pypadre.core.model.split.split import Split

        self.app.datasets.load_defaults()
        project = Project(name='Test Project 2', description='Testing the functionalities of project backend')

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        id = '_iris_dataset'
        dataset = self.app.datasets.list({'name': id})

        experiment = Experiment(name="Test Experiment SVM",
                                description="Testing Support Vector Machines via SKLearn Pipeline",
                                dataset=dataset[0],
                                workflow=create_test_pipeline(), keep_splits=True, strategy="random", project=project)

        codehash = 'abdauoasg45qyh34t'
        execution = Execution(experiment, codehash=codehash, command=None, append_runs=True, parameters=None,
                              preparameters=None, single_run=True,
                              single_transformation=True)

        run = Run(execution=execution, workflow=execution.experiment.workflow, keep_splits=True)
        split = Split(run=run, num=0, train_idx=list(range(1, 1000+1)), val_idx=None, test_idx=list(range(1000, 1100+1)), keep_splits=True)
        self.app.splits.put(split)

    def test_full_stack(self):
        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.pod.base import PadreLogger
        from pypadre.core.events import add_logger

        self.app.datasets.load_defaults()
        logger = PadreLogger(self.app)
        add_logger(logger=logger)

        project = Project(name='Test Project 2', description='Testing the functionalities of project backend')

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        id = '_iris_dataset'
        dataset = self.app.datasets.list({'name': id})

        experiment = Experiment(name="Test Experiment SVM",
                                description="Testing Support Vector Machines via SKLearn Pipeline",
                                dataset=dataset[0],
                                workflow=create_test_pipeline(), keep_splits=True, strategy="random", project=project)

        experiment.execute()


if __name__ == '__main__':
    unittest.main()
