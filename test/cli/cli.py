import os
import re
import unittest

from click.testing import CliRunner
from jsonschema import ValidationError, validate

from pypadre.app import PadreApp, PadreConfig
from pypadre.app.dataset.dataset_app import DatasetApp
from pypadre.app.padre_app import PadreFactory
from pypadre.app.project.experiment.execution.execution_app import ExecutionApp
from pypadre.app.project.experiment.execution.run.run_app import RunApp
from pypadre.app.project.experiment.execution.run.split.split_app import SplitApp
from pypadre.app.project.experiment.experiment_app import ExperimentApp
from pypadre.app.project.project_app import ProjectApp
from pypadre.cli.pypadre import pypadre


class PadreCli(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PadreCli, self).__init__(*args, **kwargs)
        config = PadreConfig(config_file=os.path.join(os.path.expanduser("~"), ".padre-test.cfg"))
        config.set("backends", str([
                    {
                        "root_dir": os.path.join(os.path.expanduser("~"), ".pypadre-test")
                    }
                ]))

    def tearDown(self):
        pass
        # delete data content

    def __del__(self):
        pass
        # delete configuration

    def test_dataset(self):
        runner = CliRunner()

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test.cfg")])
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test.cfg"),
                                         'dataset', 'load', '-d'])
        assert result.exit_code == 0
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test.cfg"),
                                         'dataset', 'list'])
        assert 'Boston' in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test.cfg"),
                                         'dataset', 'get',
                                         re.search('([a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-zA-Z0-9]+)',
                                                   result.output).group(0)])
        assert 'Diabetes' in result.output

    def test_project(self):
        runner = CliRunner()

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test.cfg")])
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test.cfg"),
                                         'project', 'create', '-d'])

if __name__ == '__main__':
    unittest.main()
