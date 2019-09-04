import os
import unittest

from pypadre.app import PadreConfig
from pypadre.app.padre_app import PadreFactory
from pypadre.backend.local.file.project.experiment.execution.execution_file_backend import PadreExecutionFileBackend
from pypadre.backend.local.file.project.experiment.execution.run.run_file_backend import PadreRunFileBackend
from pypadre.backend.local.file.project.experiment.execution.run.split.split_file_backend import PadreSplitFileBackend
from pypadre.backend.local.file.project.experiment.experiment_file_backend import PadreExperimentFileBackend
from pypadre.backend.local.file.project.project_file_backend import PadreProjectFileBackend


class LocalBackends(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(LocalBackends, self).__init__(*args, **kwargs)
        config = PadreConfig(config_file=os.path.join(os.path.expanduser("~"), ".padre-test.cfg"))
        config.set("backends", str([
                    {
                        "root_dir": os.path.join(os.path.expanduser("~"), ".pypadre-test")
                    }
                ]))
        self.app = PadreFactory.get(config)

    def tearDown(self):
        pass
        # delete data content

    def __del__(self):
        pass
        # delete configuration

    def test_dataset(self):
        # TODO test putting, fetching, searching, folder/git structure, deletion, git functionality?

        # Puts all the datasets
        self.app.datasets.load_defaults()

        # Gets a dataset by name
        id = 'Boston House Prices dataset'
        datasets = self.app.datasets.list({'name':id})
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
        # TODO test putting, fetching, searching, folder/git structure, deletion, git functionality?

        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.project import Project

        project = Project(name='Test Project 2', description='Testing the functionalities of project backend')

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        id = 'Boston House Prices dataset'
        dataset = self.app.datasets.list({'name': id})

        experiment = Experiment(name="Test Experiment SVM",
                    description="Testing Support Vector Machines via SKLearn Pipeline",
                    dataset=dataset[0],
                    workflow=create_test_pipeline(), keep_splits=True, strategy="random", project=project)

        self.app.experiments.put(experiment)
        name = 'Test Experiment SVM'
        experiments = self.app.experiments.list({'name': name})
        for experiment in experiments:
            assert name in experiment.name

    def test_execution(self):

        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.execution import Execution

        project = Project(name='Test Project 2', description='Testing the functionalities of project backend')

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        id = 'Boston House Prices dataset'
        dataset = self.app.datasets.list({'name': id})

        experiment = Experiment(name="Test Experiment SVM",
                                description="Testing Support Vector Machines via SKLearn Pipeline",
                                dataset=dataset[0],
                                workflow=create_test_pipeline(), keep_splits=True, strategy="random", project=project)

        codehash = 'abdauoasg45qyh34t'
        execution = Execution(experiment, codehash=codehash, command=None, append_runs=True, parameters=None,
                              preparameters=None, single_run=True,
                              single_transformation=True)
        self.app.executions.put(execution)

        executions = self.app.executions.list({'name': codehash})
        for execution_ in executions:
            assert codehash in execution_.name



    def test_run(self):
        """
        project_backend: PadreProjectFileBackend = self.backend.project
        experiment_backend: PadreExperimentFileBackend = project_backend.experiment
        execution_backend: PadreExecutionFileBackend = experiment_backend.execution
        run_backend: PadreRunFileBackend = execution_backend.run
        """

        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.execution import Execution
        from pypadre.core.model.run import Run

        project = Project(name='Test Project 2', description='Testing the functionalities of project backend')

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        id = 'Boston House Prices dataset'
        dataset = self.app.datasets.list({'name': id})

        experiment = Experiment(name="Test Experiment SVM",
                                description="Testing Support Vector Machines via SKLearn Pipeline",
                                dataset=dataset[0],
                                workflow=create_test_pipeline(), keep_splits=True, strategy="random", project=project)

        codehash = 'abdauoasg45qyh34t'
        execution = Execution(experiment, codehash=codehash, command=None, append_runs=True, parameters=None,
                              preparameters=None, single_run=True,
                              single_transformation=True)

        executions = self.app.executions.list({'name': codehash})

        run = Run(execution=execution, workflow=execution.experiment.workflow, keep_splits=True)
        self.app.runs.put(run)

    def test_split(self):

        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.execution import Execution
        from pypadre.core.model.run import Run
        from pypadre.core.model.split.split import Split

        project = Project(name='Test Project 2', description='Testing the functionalities of project backend')

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        id = 'Iris Plants Database'
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


if __name__ == '__main__':
    unittest.main()
