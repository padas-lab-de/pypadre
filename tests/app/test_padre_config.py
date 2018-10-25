"""
This file contains tests covering app.padre_app.PadreConfig class
"""

import os
import unittest
import uuid

import configparser
from mock import Mock

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
        padre_config = PadreConfig(Mock(), self.path)
        test_key = 'test_key'
        test_value = str(uuid.uuid4())
        padre_config.set(test_key, test_value, 'TEST')
        updated_value = padre_config.get(test_key)
        self.assertEqual(test_value, updated_value, 'Config get or set not working')

    def tearDown(self):
        """Remove config file after test"""
        os.remove(self.path)


class TestList(unittest.TestCase):
    """Test PadreConfig.list() function"""
    def setUp(self):
        """Create config file for testing purpose"""
        self.path = os.path.expanduser('~/.tests.cfg')
        self.config = configparser.ConfigParser()
        self.test_data = {'test_key': 'value 1', 'key2': 'value 2'}
        self.config['TEST'] = self.test_data
        self.config['TEST2'] = {'key3': 'value3'}
        with open(self.path, 'w+') as configfile:
            self.config.write(configfile)

    def test_list(self):
        """Test expected data returned from config list"""
        padre_config = PadreConfig(Mock(), self.path)
        result_list = padre_config.list()
        self.assertTrue(
            any(d['TEST'] == self.test_data for d in result_list if 'TEST' in d),
            'Expected data not found in config list')

    def tearDown(self):
        """Remove config file after test"""
        os.remove(self.path)
