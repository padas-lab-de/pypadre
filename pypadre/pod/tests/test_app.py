import os
import unittest

from pypadre.pod.app import PadreApp, PadreConfig
from pypadre.pod.app.dataset.dataset_app import DatasetApp
from pypadre.pod.app.padre_app import PadreAppFactory
from pypadre.pod.app.project.execution_app import ExecutionApp
from pypadre.pod.app.project.experiment_app import ExperimentApp
from pypadre.pod.app.project.project_app import ProjectApp
from pypadre.pod.app.project.run_app import RunApp
from pypadre.pod.app.project.split_app import SplitApp


class LocalBackends(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(LocalBackends, self).__init__(*args, **kwargs)
        config = PadreConfig(config_file=os.path.join(os.path.expanduser("~"), ".padre-test.cfg"))
        config.set("backends", str([
            {
                "root_dir": os.path.join(os.path.expanduser("~"), ".pypadre-test")
            }
        ]))
        self.app = PadreAppFactory.get(config)

    def test_app(self):
        PadreApp
        # TODO test app

    def test_datasets(self):
        dataset_app: DatasetApp = PadreApp.datasets
        # TODO test app

    def test_projects(self):
        project_app: ProjectApp = PadreApp.projects
        # TODO test app

    def test_executions(self):
        execution_app: ExecutionApp = PadreApp.executions
        # TODO test app

    def test_experiments(self):
        experiment_app: ExperimentApp = PadreApp.experiments
        # TODO test app

    def test_runs(self):
        run_app: RunApp = PadreApp.runs
        # TODO test app

    def test_splits(self):
        split_app: SplitApp = PadreApp.splits
        # TODO test app

if __name__ == '__main__':
    unittest.main()
