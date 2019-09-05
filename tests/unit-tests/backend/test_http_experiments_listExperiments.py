"""
This file contains tests covering
backend.http_experiments.HttpBackendExperiments.list_experiments function
All unnecessary function and http calls are mocked
"""
import json
import unittest

from mock import MagicMock

from pypadre.pod.backend.http.http_experiments import HttpBackendExperiments


class TestListExperiments(unittest.TestCase):
    """Test HttpBackendExperiments.list_experiments

    All unnecessary function call and http calls are mocked
    """
    def setUp(self):
        """Initializing for create_dataset test.

        All non related function calls and http calls will be mocked for this purpose.
        """
        self.base_url = 'test.com'
        self.experiments_path = "/experiments"
        self.count = 10
        self.http_client = MagicMock()
        self.http_client.has_token = MagicMock(return_value=True)
        self.http_client.paths = {"experiments": "/experiments",
                                  "search": lambda entity: "/" + entity + "/search?search="}
        mocked_post_response = MagicMock()
        test_experiments = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        mocked_post_response.content = json.dumps({'_embedded': {"experiments": test_experiments}})
        self.http_client.do_get = MagicMock(return_value=mocked_post_response)

    def test_list_experiments_01(self):
        """Test list_experiments

        Scenario: If correct list experiment names is returned
                  Expected url in call args for do_get matches

        """
        obj = HttpBackendExperiments(self.http_client)
        obj.get_base_url = MagicMock(return_value="test.com")
        size = 10
        response = obj.list_experiments(count=size)
        self.assertListEqual(response, ["a", "b", "c"],
                             "List of experiment names not matches")
        self.assertEqual(self.http_client.do_get.call_args[0][0],
                         self.base_url + self.experiments_path + "?size=" + str(size),
                         "Url for do_get not matches"
                         )

    def test_list_experiments_02(self):
        """Test list_experiments with start and count

        Scenario: If correct list experiment names is returned for start and count

        """
        obj = HttpBackendExperiments(self.http_client)
        obj.get_base_url = MagicMock(return_value="test.com")
        size = 10
        response = obj.list_experiments(start=1, count=size)
        self.assertListEqual(response, ["b", "c"],
                             "List of experiment names not matches")

    def test_list_experiments_03(self):
        """Test list_experiments with search parameter

        Scenario: Expected url in call args for do_get matches

        """
        obj = HttpBackendExperiments(self.http_client)
        obj.get_base_url = MagicMock(return_value="test.com")
        size = 10
        search="test experiment"
        response = obj.list_experiments(search=search, count=size)
        self.assertEqual(self.http_client.do_get.call_args[0][0],
                         self.base_url + "/experiments/search?search=name?:" + search + "&size=" + str(size),
                         "Url for do_get not matches with search attribute is given"
                         )

    def tearDown(self):
        pass
