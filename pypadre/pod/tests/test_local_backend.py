import os
import shutil
import unittest

from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.core.events.events import connect_base_signal, LOG_EVENT
from pypadre.pod.app import PadreConfig
from pypadre.pod.app.padre_app import PadreAppFactory

config_path = os.path.join(os.path.expanduser("~"), ".padre-test.cfg")
workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-test")


class LocalBackends(unittest.TestCase):

    def test_dataset(self):
        # TODO test putting, fetching, searching, folder/git structure, deletion, git functionality?
        pass

    def test_project(self):
        pass

    def test_experiment(self):
        # TODO test putting, fetching, searching, folder/git structure, deletion, git functionality?
        pass

    def test_execution(self):
        pass

    def test_run(self):
        """
        project_backend: ProjectFileRepository = self.backend.project
        experiment_backend: ExperimentFileRepository = project_backend.experiment
        execution_backend: ExecutionFileRepository = experiment_backend.execution
        run_backend: RunFileRepository = execution_backend.run
        """
        # TODO clean up testing (tests should be independent and only test the respective functionality)
        pass

    def test_split(self):
        pass

    def test_full_stack(self):
        pass


if __name__ == '__main__':
    unittest.main()
