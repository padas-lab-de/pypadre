import configparser
import os
import shutil
import unittest

from pypadre.pod.app import PadreConfig


class PadreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(os.path.expanduser("~"), ".padre-test.cfg")
        cls.workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-test")

        """Create config file for testing purpose"""
        cls.config = configparser.ConfigParser()
        test_data = {'test_key': 'value 1', 'key2': 'value 2'}
        cls.config['TEST'] = test_data
        with open(cls.config_path, 'w+') as configfile:
            cls.config.write(configfile)

        cls.config = PadreConfig(config_file=cls.config_path)
        cls.config.set("backends", str([
            {
                "root_dir": cls.workspace_path
            }
        ]))

    @classmethod
    def tearDownClass(cls):
        """Remove config file after test"""
        os.remove(cls.config_path)

    def setUp(self):
        pass

    def tearDown(self):
        try:

            if os.path.exists(os.path.join(self.workspace_path,"datasets")):
                shutil.rmtree(os.path.join(self.workspace_path,"datasets"))

            if os.path.exists(os.path.join(self.workspace_path,"projects")):
                shutil.rmtree(os.path.join(self.workspace_path,"projects"))

        except FileNotFoundError:
            pass

        print('Teardown of test')


