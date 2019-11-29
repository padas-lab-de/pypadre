import os
import re
import shutil
import unittest

from click.testing import CliRunner

from pypadre.cli.pypadre import pypadre
from pypadre.pod.app import PadreConfig


# noinspection PyMethodMayBeStatic
class PadreCli(unittest.TestCase):
    workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-test-cli")

    def __init__(self, *args, **kwargs):
        super(PadreCli, self).__init__(*args, **kwargs)
        config = PadreConfig(config_file=os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"))
        config.set("backends", str([
            {
                "root_dir": os.path.join(os.path.expanduser("~"), ".pypadre-test-cli")
            }
        ]))

    def tearDown(self):
        # delete data content
        try:
            if os.path.exists(os.path.join(self.workspace_path, "datasets")):
                shutil.rmtree(self.workspace_path + "/datasets")
            if os.path.exists(os.path.join(self.workspace_path, "projects")):
                shutil.rmtree(self.workspace_path + "/projects")
            if os.path.exists(os.path.join(self.workspace_path, "code")):
                shutil.rmtree(self.workspace_path + "/code")
        except FileNotFoundError:
            pass

    def __del__(self):
        pass
        # delete configuration

    def test_dataset(self):
        runner = CliRunner()

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

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                'project', 'create', '-n', 'Examples'])

        path = os.path.abspath('../../../examples/20_experiment_cli_creation/20_experiment_cli_creation.py')
        print(path)
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'project', 'select', 'Examples'],
                               input="experiment execute --name Experiment1 --path {}\nn\nk".format(path))

        assert result.exit_code == 0
        assert 'Execution of the experiment is finished!' in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'experiment', 'list'])

        assert 'Experiment1' in result.output

        id = [s for s in result.output.split() if s.startswith('Experiment1-')].pop()

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'experiment', 'get', id])

        assert id in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'experiment', 'select', id])

        assert result.exit_code == 0

    def test_execution(self):
        runner = CliRunner()
        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                'project', 'create', '-n', 'Examples'])

        path = os.path.abspath('../../../examples/20_experiment_cli_creation/20_experiment_cli_creation.py')
        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                'project', 'select', 'Examples'],
                      input="experiment execute --name Experiment1 --path {}\nn\nk".format(path))

        with open(path, "a") as f:
            f.write("# Change added for testing")

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                'project', 'select', 'Examples'],
                      input="experiment execute --name Experiment1 --path {}\nn\nk".format(path))

        with open(path, "r") as f:
            lines = f.readlines()
        with open(path, "w") as f:
            f.writelines(lines[:-1])

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'execution', 'list', '-c', 'id'])

        assert result.exit_code == 0

        import re
        ids = re.findall(r'\b\d+[-]\d+\b', result.output)

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'execution', 'get', ids[0]])

        assert ids[0] in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'execution', 'compare', ids[0], ids[1]])

        assert result.exit_code == 0

        assert 'diff --git' in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'execution', 'select', ids[0]])
        assert result.exit_code == 0


if __name__ == '__main__':
    unittest.main()
