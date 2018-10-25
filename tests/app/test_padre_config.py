"""
This file contains tests covering app.padre_app.PadreConfig class
"""

import os
import unittest
import uuid

import configparser
from mock import patch, MagicMock

from padre.app.padre_app import PadreConfig
from padre.backend.http import PadreHTTPClient


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
        padre_config = PadreConfig(MagicMock(), self.path)
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
        padre_config = PadreConfig(MagicMock(), self.path)
        result_list = padre_config.list()
        self.assertTrue(
            any(d['TEST'] == self.test_data for d in result_list if 'TEST' in d),
            'Expected data not found in config list')

    def tearDown(self):
        """Remove config file after test"""
        os.remove(self.path)


class TestAuthenticate(unittest.TestCase):
    """Test PadreConfig.authenticate() function"""

    def setUp(self):
        """Create config file for testing purpose"""
        self.test_token = "Bearer test " + str(uuid.uuid4())

        self.path = os.path.expanduser('~/.tests.cfg')
        self.config = configparser.ConfigParser()
        self.test_data = {'test_key': 'value 1', 'key2': 'value 2'}
        self.config['HTTP'] = self.test_data
        with open(self.path, 'w+') as configfile:
            self.config.write(configfile)

    @patch('padre.backend.http.PadreHTTPClient.get_access_token')
    def test_authenticate_01(self, mock_token):
        """Test PadreConfig.authenticate function for expected call args.

        Scenario: authenticate function calls get_access_token with expected args
        """
        mock_token.return_value = self.test_token
        self.http_client = PadreHTTPClient(user='test user', passwd='test', token='Test')
        padre_config = PadreConfig(self.http_client, self.path)
        args_list = ('test_url', 'test_user', 'test_pass')
        padre_config.authenticate(*args_list)
        self.assertTupleEqual(args_list, mock_token.call_args[0],
                              'Expected args not matches in get_access_token call')

    @patch('padre.backend.http.PadreHTTPClient.get_access_token')
    def test_authenticate_01(self, mock_token):
        """Test PadreConfig.authenticate function for expected token value.

        Scenario: New token is correctly updated in config file
        """
        mock_token.return_value = self.test_token
        self.http_client = PadreHTTPClient(user='test user', passwd='test', token='Test')
        padre_config = PadreConfig(self.http_client, self.path)
        padre_config.authenticate('test_url', 'test_user', 'test_pass')
        config = configparser.ConfigParser()
        config.read(self.path)
        data = config.items('HTTP')
        self.assertTrue(
            any(v == self.test_token for k, v in data if k == 'token'),
            'Token not set in config file')

    def tearDown(self):
        """Remove config file after test"""
        os.remove(self.path)
