"""
This file contains tests covering app.padre_app.PadreConfig class
"""

import os
import unittest
import uuid

import configparser

from padre.app.padre_app import PadreConfig


class TestSetAndGet(unittest.TestCase):
    """Test PadreConfig get and set functions"""
    def setUp(self):
        """Create config file for testing purpose"""
        self.path = os.path.expanduser('~/.tests.cfg')
        self.config = configparser.ConfigParser()
        test_data = {'test_key': 'value 1', 'key2': 'value 2'}
        self.config['TEST'] = test_data
        with open(self.path, 'w+') as configfile:
            self.config.write(configfile)

    def test_set_and_get(self):
        """Test set and get functions"""
        padre_config = PadreConfig(self.path)
        test_key = 'test_key'
        test_value = str(uuid.uuid4())
        padre_config.set(test_key, test_value, 'TEST')
        updated_value = padre_config.get(test_key, 'TEST')
        self.assertEqual(test_value, updated_value, 'Config get or set not working')

    def tearDown(self):
        """Remove config file after test"""
        os.remove(self.path)

