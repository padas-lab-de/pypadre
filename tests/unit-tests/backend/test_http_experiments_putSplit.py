"""
This file contains tests covering backend.http_experiments.HttpBackendExperiments.put_split
All unnecessary function and http calls are mocked
"""
import unittest
import uuid

import numpy as np
from mock import MagicMock, patch

from pypadre.pod.backend.http.http_experiments import HttpBackendExperiments


class TestPutSplit(unittest.TestCase):
    """Test http_experiments.HttpBackendExperiments.put_split

    All unnecessary function calls and http calls are mocked
    """

    def setUp(self):
        """Initializing for put_split test.

        All non related function calls and http calls will be mocked for this purpose.
        """
        self.test_split_url = 'api/splits/' + '3'
        self.http_client = MagicMock()
        self.http_client.user = 'test user'
        self.http_client.base = 'padretest.com/api'
        self.http_client.has_token = MagicMock(return_value=True)

        mocked_post_split = MagicMock()
        mocked_post_split.headers = {'location': self.test_split_url}

        self.http_client.do_post = MagicMock(return_value=mocked_post_split)

    @patch('pypadre.pod.backend.http_experiments.HttpBackendExperiments.get_or_create_project')
    def test_put_split(self, mock_project):
        """Test HttpBackendExperiments.put_split function.

        Scenario:
            1- Put split should return url of newly created experiment.
            2- Put split should have expected encoded value of split.
        """
        mock_project.return_value = 1
        obj = HttpBackendExperiments(self.http_client)
        run = MagicMock()
        split = MagicMock()
        run.metadata = MagicMock()
        run.metadata = {"server_url": "padretest.com/api/runs/1"}
        split.metadata = MagicMock()
        split.metadata = {"server_url": "padretest.com/api/runs/1"}
        split.train_idx = MagicMock()
        split.test_idx = MagicMock()
        split.val_idx = None
        split.train_idx = np.array([2, 1, 5])  # train:f1t2f2t1
        split.test_idx = np.array([6, 4, 7])   # test: f4t1f1t2
        split.id = str(uuid.uuid4())

        result = obj.put_split(MagicMock(), run, split)
        self.assertEqual(self.test_split_url,
                         result,
                         'Put split does not return url of newly created split')
        self.assertIn('"split": "train:f1t2f2t1,test:f4t1f1t2,val:',
                      self.http_client.do_post.call_args[1]["data"],
                      "Split not posted with expected encoded value")

    def tearDown(self):
        pass
