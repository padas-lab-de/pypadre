"""
This file contains tests covering backend.http_experiments.HttpBackendExperiments.encode_split
All unnecessary function and http calls are mocked
"""
import unittest
import uuid

from mock import MagicMock

from padre.backend.http_experiments import HttpBackendExperiments


class TestEncodeSplit(unittest.TestCase):
    """Test http_experiments.HttpBackendExperiments.encode_split
    """

    def setUp(self):
        """Initializing for encode_split test.
        """
        self.http_client = MagicMock()
        self.http_client.user = 'test user'
        self.http_client.base = 'padretest.com/api'
        self.http_client.has_token = MagicMock(return_value=False)

    def test_encode_split_01(self):
        """Test HttpBackendExperiments.encode_split function.

        Scenario:
            1- Expected encoded value should be returned.
        """
        obj = HttpBackendExperiments(self.http_client)
        split = MagicMock()
        split.train_idx = MagicMock()
        split.test_idx = MagicMock()
        split.train_idx = [2, 1, 5]  # train:f1t2f2t1
        split.test_idx = [6, 4, 7]   # test: f4t1f1t2
        split.id = str(uuid.uuid4())

        result = obj.encode_split(split)
        self.assertIn("train:f1t2f2t1,test:f4t1f1t2",
                      result,
                      "Expected encoded value not returned")

    def test_encode_split_02(self):
        """Test HttpBackendExperiments.encode_split function.

        Scenario:
            1- Test expected value if test set is empty.
        """
        obj = HttpBackendExperiments(self.http_client)
        split = MagicMock()
        split.train_idx = MagicMock()
        split.test_idx = MagicMock()
        split.train_idx = [2, 1, 5]  # train:f1t2f2t1
        split.test_idx = []   # test:
        split.id = str(uuid.uuid4())

        result = obj.encode_split(split)
        self.assertIn("train:f1t2f2t1,test:",
                      result,
                      "Expected encoded value not returned")

    def test_encode_split_03(self):
        """Test HttpBackendExperiments.encode_split function.

        Scenario:
            1- Test expected value if train and test set are empty.
        """
        obj = HttpBackendExperiments(self.http_client)
        split = MagicMock()
        split.train_idx = MagicMock()
        split.test_idx = MagicMock()
        split.train_idx = []  # train:
        split.test_idx = []   # test:
        split.id = str(uuid.uuid4())

        result = obj.encode_split(split)
        self.assertIn("train:,test:",
                      result,
                      "Expected encoded value not returned")

    def test_encode_split_04(self):
        """Test HttpBackendExperiments.encode_split function.

        Scenario:
            1- Test expected value if train and test sets has only one value.
        """
        obj = HttpBackendExperiments(self.http_client)
        split = MagicMock()
        split.train_idx = MagicMock()
        split.test_idx = MagicMock()
        split.train_idx = [1]  # train:f1t1
        split.test_idx = [2]   # test:f2t1
        split.id = str(uuid.uuid4())

        result = obj.encode_split(split)
        self.assertIn("train:f1t1,test:f2t1",
                      result,
                      "Expected encoded value not returned")

    def tearDown(self):
        pass
