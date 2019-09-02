import unittest

from pypadre.app import PadreConfig
from pypadre.backend.local.file.dataset.dataset_file_backend import PadreDatasetFileBackend
from pypadre.backend.local.file.file import PadreFileBackend
from pypadre.backend.local.file.project.experiment.execution.execution_file_backend import PadreExecutionFileBackend
from pypadre.backend.local.file.project.experiment.execution.run.run_file_backend import PadreRunFileBackend
from pypadre.backend.local.file.project.experiment.execution.run.split.split_file_backend import PadreSplitFileBackend
from pypadre.backend.local.file.project.experiment.experiment_file_backend import PadreExperimentFileBackend
from pypadre.backend.local.file.project.project_file_backend import PadreProjectFileBackend


class LocalBackends(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(LocalBackends, self).__init__(*args, **kwargs)
        self.backend = PadreFileBackend(PadreConfig().get("backends")[1])

    def test_dataset(self):
        dataset_backend: PadreDatasetFileBackend = self.backend.dataset
        # TODO test putting, fetching, searching, folder/git structure, deletion, git functionality?

        from pypadre.app.dataset.dataset_app import DatasetApp
        dataset_app = DatasetApp(self, dataset_backend)
        # Puts all the datasets
        dataset_app.load_defaults()

        # Gets a dataset by name
        id = 'Boston House Prices dataset'
        dataset = dataset_app.list({'name':id})
        print(dataset)

    def test_project(self):
        project_backend: PadreProjectFileBackend = self.backend.project
        from pypadre.core.model.project import Project
        from pypadre.app.project.project_app import ProjectApp
        project_app = ProjectApp(self, project_backend)

        project = Project(name='Test Project', description='Testing the functionalities of project backend')

        project_app.put(project)

        p = project_app.list({'name': 'Test Project'})
        print(p)


    def test_experiment(self):
        # TODO test putting, fetching, searching, folder/git structure, deletion, git functionality?

        from pypadre.core.model.experiment import Experiment
        from pypadre.app.dataset.dataset_app import DatasetApp
        from pypadre.core.model.project import Project
        from pypadre.app.project.project_app import ProjectApp

        dataset_backend: PadreDatasetFileBackend = self.backend.dataset
        dataset_app = DatasetApp(self, dataset_backend)

        project_backend: PadreProjectFileBackend = self.backend.project
        project_app = ProjectApp(self, project_backend)

        project = Project(name='Test Project', description='Testing the functionalities of project backend')

        project_app.put(project)

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            from sklearn.decomposition import PCA
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        project_backend: PadreProjectFileBackend = self.backend.project
        experiment_backend: PadreExperimentFileBackend = project_backend.experiment

        id = 'Boston House Prices dataset'
        dataset = dataset_app.list({'name': id})
        experiment = Experiment(name="Test Experiment SVM",
                    description="Testing Support Vector Machines via SKLearn Pipeline",
                    dataset=dataset[0],
                    workflow=create_test_pipeline(), keep_splits=True, strategy="random", project=project)

        experiment_backend.put(experiment=experiment)
        # TODO

    def test_execution(self):
        project_backend: PadreProjectFileBackend = self.backend.project
        experiment_backend: PadreExperimentFileBackend = project_backend.experiment
        execution_backend: PadreExecutionFileBackend = experiment_backend.execution
        # TODO

    def test_run(self):
        project_backend: PadreProjectFileBackend = self.backend.project
        experiment_backend: PadreExperimentFileBackend = project_backend.experiment
        execution_backend: PadreExecutionFileBackend = experiment_backend.execution
        run_backend: PadreRunFileBackend = execution_backend.run
        # TODO

    def test_split(self):
        project_backend: PadreProjectFileBackend = self.backend.project
        experiment_backend: PadreExperimentFileBackend = project_backend.experiment
        execution_backend: PadreExecutionFileBackend = experiment_backend.execution
        run_backend: PadreRunFileBackend = execution_backend.run
        split_backend: PadreSplitFileBackend = run_backend.split
        # TODO


if __name__ == '__main__':
    unittest.main()
