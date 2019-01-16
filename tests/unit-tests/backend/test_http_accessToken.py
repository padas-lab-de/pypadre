"""
This file contains tests covering
- backend.http.PadreHTTPClient.get_access_token
- backend.http.PadreHTTPClient.has_token
 functions

All unnecessary function and http calls are mocked
"""
import json
import unittest
import uuid
from requests.exceptions import ConnectionError

from mock import MagicMock, patch

from padre.backend.http import PadreHTTPClient


class TestGetAccessToken01(unittest.TestCase):
    """Test http.PadreHTTPClient.get_access_token will all scenarios
    """

    @patch('padre.backend.http.PadreHTTPClient.get_access_token')
    def setUp(self, mock_token):
        mock_token.return_value = None
        self.test_user = str(uuid.uuid4())[:15]
        self.test_password = str(uuid.uuid4())
        self.test_csrf = 'test-csrf-token'
        args_list = ('test url', self.test_user, self.test_password)
        self.test_object = PadreHTTPClient(*args_list)
        get_mock = MagicMock()
        get_mock.cookies = MagicMock()
        get_mock.cookies.get = MagicMock(return_value=self.test_csrf)
        self.test_object.do_get = MagicMock(return_value = get_mock)

    def test_access_token_01(self):
        """
        Test PadreHTTPClient.get_access_token

        Scenario: status code 200 and correct token returned
        """
        response_mock = MagicMock()
        response_mock.status_code = 200
        test_token = str(uuid.uuid4())
        response_mock.content = json.dumps({'access_token': test_token})
        self.test_object.do_post = MagicMock(return_value=response_mock)
        result = self.test_object.get_access_token()
        self.assertEqual("Bearer " + test_token,
                         result,
                         "Expected token result not matches")

    def test_access_token_02(self):
        """
        Test PadreHTTPClient.get_access_token with failed status code

        Scenario: status code not 200 and None returned
        """
        response_mock = MagicMock()
        response_mock.status_code = 401
        test_token = str(uuid.uuid4())
        response_mock.content = json.dumps({"access_token": test_token})
        self.test_object.do_post = MagicMock(return_value=response_mock)
        result = self.test_object.get_access_token()
        self.assertIsNone(result, "Expected token result not matches")

    def test_access_token_03(self):
        """
        Test PadreHTTPClient.get_access_token with default user and password

        Scenario: Test if do_post called with default user, password and grant_type
        """
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.content = json.dumps({"access_token": "test_token"})
        self.test_object.do_post = MagicMock(return_value=response_mock)
        self.test_object.get_access_token(passwd=self.test_password)
        self.assertEqual(self.test_user,
                         self.test_object.do_post.call_args_list[0][1]['data']['username'],
                         "do_post not called with default username")
        self.assertEqual("password",
                         self.test_object.do_post.call_args_list[0][1]['data']['grant_type'],
                         "do_post not called with grant_type password")

    def test_access_token_04(self):
        """
        Test PadreHTTPClient.get_access_token with given url, user, password and csrf

        Scenario: Test if do_post called with given url, user, password and csrf in url
        """
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.content = json.dumps({"access_token": "test_token"})
        self.test_object.do_post = MagicMock(return_value=response_mock)
        url = "test given url"
        user = "test_user"
        password = "test_password"
        self.test_object.get_access_token(password)
        self.assertEqual(password,
                         self.test_object.do_post.call_args[1]['data']['password'],
                         "do_post not called with given password")
        self.assertEqual("token?=" + self.test_csrf,
                         self.test_object.do_post.call_args[0][0].split('/')[-1],
                         "do_post not called with expected csrf token with url")

    def tearDown(self):
        pass


class TestGetAccessToken02(unittest.TestCase):
    """Test http.PadreHTTPClient.get_access_token with exception
    """

    @patch('padre.backend.http.PadreHTTPClient.get_access_token')
    def setUp(self, mock_token):
        mock_token.return_value = None
        self.test_object = PadreHTTPClient(*('test url', 'user', 'pass'))
        self.test_object.do_get = MagicMock()
        e = ConnectionError('Error')
        e.response = MagicMock()
        e.response.text = ""
        self.test_object.do_get.side_effect = MagicMock(side_effect=e)

    def test_access_token_05(self):
        """
        Test PadreHTTPClient.get_access_token with ConnectionError exception

        Scenario: Token should be None when ConnectionError raised
        """
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.content = json.dumps({'access_token': 'test_token'})
        self.test_object.do_post = MagicMock(return_value=response_mock)
        result = self.test_object.get_access_token()
        self.assertIsNone(result, "In ConnectionError token should be None")

    def tearDown(self):
        pass


class TestHasToken(unittest.TestCase):
    """Test http.PadreHTTPClient.has_token with all scenarios
    """

    def setUp(self):
        pass

    @patch('padre.backend.http.PadreHTTPClient.get_access_token')
    def test_has_token_01(self, mock_token):
        """
        Test PadreHTTPClient.has_token

        Scenario: When token is not None
        """
        mock_token.return_value = "test token"
        self.test_object = PadreHTTPClient(*('test url', 'user', 'pass'))
        result = self.test_object.has_token()
        self.assertTrue(result, "Not returning True")

    @patch('padre.backend.http.PadreHTTPClient.get_access_token')
    def test_has_token_02(self, mock_token):
        """
        Test PadreHTTPClient.has_token

        Scenario: When token is None
        """
        mock_token.return_value = None
        self.test_object = PadreHTTPClient(*('test_url.com/api', 'user'))
        result = self.test_object.has_token()
        self.assertFalse(result, "Not returning False")

    def tearDown(self):
        pass
