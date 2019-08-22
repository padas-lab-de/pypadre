import unittest

from pypadre.backend.local.file.dataset.dataset_file_backend import PadreDatasetFileBackend
from pypadre.backend.local.file.file import PadreFileBackend
from pypadre.backend.local.file.project.experiment.execution.execution_file_backend import PadreExecutionFileBackend
from pypadre.backend.local.file.project.experiment.execution.run.run_file_backend import PadreRunFileBackend
from pypadre.backend.local.file.project.experiment.execution.run.split.split_file_backend import PadreSplitFileBackend
from pypadre.backend.local.file.project.experiment.experiment_file_backend import PadreExperimentFileBackend
from pypadre.backend.local.file.project.project_file_backend import PadreProjectFileBackend


class LocalBackends(unittest.TestCase):

    def test_dataset(self):
        dataset_backend: PadreDatasetFileBackend = PadreFileBackend.dataset
        # TODO test putting, fetching, searching, folder/git structure, deletion, git functionality?

    def test_project(self):
        project_backend: PadreProjectFileBackend = PadreFileBackend.project
        # TODO

    def test_experiment(self):
        project_backend: PadreProjectFileBackend = PadreFileBackend.project
        experiment_backend: PadreExperimentFileBackend = project_backend.experiment
        # TODO

    def test_execution(self):
        project_backend: PadreProjectFileBackend = PadreFileBackend.project
        experiment_backend: PadreExperimentFileBackend = project_backend.experiment
        execution_backend: PadreExecutionFileBackend = experiment_backend.execution
        # TODO

    def test_run(self):
        project_backend: PadreProjectFileBackend = PadreFileBackend.project
        experiment_backend: PadreExperimentFileBackend = project_backend.experiment
        execution_backend: PadreExecutionFileBackend = experiment_backend.execution
        run_backend: PadreRunFileBackend = execution_backend.run
        # TODO

    def test_split(self):
        project_backend: PadreProjectFileBackend = PadreFileBackend.project
        experiment_backend: PadreExperimentFileBackend = project_backend.experiment
        execution_backend: PadreExecutionFileBackend = experiment_backend.execution
        run_backend: PadreRunFileBackend = execution_backend.run
        split_backend: PadreSplitFileBackend = run_backend.split
        # TODO


if __name__ == '__main__':
    unittest.main()
