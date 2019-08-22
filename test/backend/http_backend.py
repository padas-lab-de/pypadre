
import unittest

from pypadre.app import PadreConfig
from pypadre.backend.remote.http.dataset.dataset_http_backend import PadreDatasetHttpBackend
from pypadre.backend.remote.http.http import PadreHttpBackend
from pypadre.backend.remote.http.project.experiment.execution.execution_http_backend import PadreExecutionHttpBackend
from pypadre.backend.remote.http.project.experiment.execution.run.run_http_backend import PadreRunHttpBackend
from pypadre.backend.remote.http.project.experiment.execution.run.split.split_http_backend import PadreSplitHttpBackend
from pypadre.backend.remote.http.project.experiment.experiment_http_backend import PadreExperimentHttpBackend
from pypadre.backend.remote.http.project.project_http_backend import PadreProjectHttpBackend


class HttpBackends(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(HttpBackends, self).__init__(*args, **kwargs)
        self.backend = PadreHttpBackend(PadreConfig().get("backends")[0])

    def test_dataset(self):
        dataset_backend: PadreDatasetHttpBackend = self.backend.dataset
        # TODO test putting, fetching, searching, folder/git structure, deletion, git functionality?

    def test_project(self):
        project_backend: PadreProjectHttpBackend = self.backend.project
        # TODO

    def test_experiment(self):
        project_backend: PadreProjectHttpBackend = self.backend.project
        experiment_backend: PadreExperimentHttpBackend = project_backend.experiment
        # TODO

    def test_execution(self):
        project_backend: PadreProjectHttpBackend = self.backend.project
        experiment_backend: PadreExperimentHttpBackend = project_backend.experiment
        execution_backend: PadreExecutionHttpBackend = experiment_backend.execution
        # TODO

    def test_run(self):
        project_backend: PadreProjectHttpBackend = self.backend.project
        experiment_backend: PadreExperimentHttpBackend = project_backend.experiment
        execution_backend: PadreExecutionHttpBackend = experiment_backend.execution
        run_backend: PadreRunHttpBackend = execution_backend.run
        # TODO

    def test_split(self):
        project_backend: PadreProjectHttpBackend = self.backend.project
        experiment_backend: PadreExperimentHttpBackend = project_backend.experiment
        execution_backend: PadreExecutionHttpBackend = experiment_backend.execution
        run_backend: PadreRunHttpBackend = execution_backend.run
        split_backend: PadreSplitHttpBackend = run_backend.split
        # TODO


if __name__ == '__main__':
    unittest.main()
