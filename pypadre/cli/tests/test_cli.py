import os
import re
import unittest

from click.testing import CliRunner

from pypadre.cli.pypadre import pypadre
from pypadre.pod.app import PadreConfig


# noinspection PyMethodMayBeStatic
class PadreCli(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PadreCli, self).__init__(*args, **kwargs)
        config = PadreConfig(config_file=os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"))
        config.set("backends", str([
            {
                "root_dir": os.path.join(os.path.expanduser("~"), ".pypadre-test-cli")
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

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg")])
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'dataset', 'load', '-d'])
        assert result.exit_code == 0
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'dataset', 'list'])
        assert '_boston' in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'dataset', 'get',
                                         '_diabetes_dataset'])
        assert '_diabetes' in result.output

    def test_project(self):
        runner = CliRunner()

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg")])

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'project', 'create', '-n', 'Examples'])
        assert result.exit_code == 0

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'project', 'list'])

        assert "Examples" in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'project', 'select', 'Examples'])

        assert result.exit_code == 0

    def test_experiment(self):
        runner = CliRunner()

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg")])
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'project', 'create', '-n', 'Examples'])

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'project', 'select', 'Examples'],
                               input="experiment execute --name Experiment1 --path /home/mehdi/example_exp/example_exp1.py\ny\n")

        assert result.exit_code==0
        assert 'Execution of the experiment is finished!' in result.output
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'experiment', 'list'])

        assert 'Experiment1' in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'experiment', 'select', 'Experiment1'])

        assert result.exit_code==0

    def test_execution(self):
        runner = CliRunner()


    def test_computation(self):
        runner = CliRunner()
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'run', 'select', '5c6d7d64-1378-471c-baab-0fc48e18880a-4462774619032339222'],
                               input='computation list\n')


if __name__ == '__main__':
    unittest.main()
