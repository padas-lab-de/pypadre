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
        config = PadreConfig(config_file=os.path.join(os.path.expanduser("~"), ".padre-example.cfg"))
        config.set("backends", str([
            {
                "root_dir": os.path.join(os.path.expanduser("~"), ".pypadre-example")
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

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-example.cfg")])
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-example.cfg"),
                                         'dataset', 'load', '-d'])
        assert result.exit_code == 0
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-example.cfg"),
                                         'dataset', 'list'])
        assert '_boston' in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-example.cfg"),
                                         'dataset', 'get',
                                         re.search('([a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-zA-Z0-9]+)',
                                                   result.output).group(0)])
        assert '_diabetes' in result.output

    def test_project(self):
        runner = CliRunner()

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test.cfg")])

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-example.cfg"),
                                         'project', 'create', '-n', 'Examples'])
        assert result.exit_code == 0

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-example.cfg"),
                                         'project', 'list'])

        assert "Examples" in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-example.cfg"),
                                         'project', 'select', 'Examples'])

        assert result.exit_code == 0

    def test_experiment(self):
        runner = CliRunner()

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre.cfg")])
        # result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre.cfg"),
        #                                  'experiment', 'list'])

        # assert 'Iris SVC' in result.output

        # runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre.cfg"),
        #                         'project', 'create', '-n', 'Examples'])

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre.cfg"),
                                         'project', 'select', 'Examples'],
                               input="experiment execute --name example_exp --path /home/mehdi/example_exp/example_exp.py")

        # result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-example.cfg"),
        #                                  'experiment', 'create', '-n', 'cli_experiment'])

        assert result.exit_code == 0

    def test_computation(self):
        runner = CliRunner()
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test.cfg"),
                                         'run', 'select', '5c6d7d64-1378-471c-baab-0fc48e18880a-4462774619032339222'],
                               input='computation list\n')


if __name__ == '__main__':
    unittest.main()
