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
        assert '_boston' in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test.cfg"),
                                         'dataset', 'get',
                                         re.search('([a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-zA-Z0-9]+)',
                                                   result.output).group(0)])
        assert '_diabetes' in result.output

    def test_project(self):
        # def handle_missing(obj, e, options):
        #     return "a"
        #
        # p = ValidateableFactory.make(Project, handlers=[
        #     JsonSchemaRequiredHandler(validator="required", get_value=handle_missing)])

        runner = CliRunner()

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test.cfg")])
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test.cfg"),
                                         'project', 'list'])

    def test_experiment(self):
        # def handle_missing(obj, e, options):
        #     return "a"
        #
        # p = ValidateableFactory.make(Project, handlers=[
        #     JsonSchemaRequiredHandler(validator="required", get_value=handle_missing)])

        runner = CliRunner()

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test.cfg")])
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test.cfg"),
                                         'experiment', 'list'])

        assert 'created_at' in result.output


if __name__ == '__main__':
    unittest.main()
