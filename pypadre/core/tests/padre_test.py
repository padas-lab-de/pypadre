import configparser
import os
import shutil
import unittest

from pypadre.pod.app import PadreConfig


class PadreTest(unittest.TestCase):

    def setUp(self):
        self.config_path = os.path.join(os.path.expanduser("~"), ".padre-test.cfg")
        self.workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-test")

        """Create config file for testing purpose"""
        self.config = configparser.ConfigParser()
        test_data = {'test_key': 'value 1', 'key2': 'value 2'}
        self.config['TEST'] = test_data
        with open(self.config_path, 'w+') as configfile:
            self.config.write(configfile)

        config = PadreConfig(config_file=self.config_path)
        config.set("backends", str([
            {
                "root_dir": self.workspace_path
            }
        ]))

    def tearDown(self):
        """Remove config file after test"""
        os.remove(self.config_path)
        try:

            if os.path.exists(os.path.join(self.workspace_path,"datasets")):
                shutil.rmtree(os.path.join(self.workspace_path,"datasets"))

            if os.path.exists(os.path.join(self.workspace_path,"projects")):
                shutil.rmtree(os.path.join(self.workspace_path,"projects"))

        except FileNotFoundError:
            pass
