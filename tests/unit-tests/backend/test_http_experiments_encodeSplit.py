"""
This file contains tests covering backend.http_experiments.HttpBackendExperiments.encode_split
All unnecessary function and http calls are mocked
"""
import unittest
import uuid

import numpy as np
from mock import MagicMock


from pypadre.pod.backend.http.http_experiments import HttpBackendExperiments


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
        split.val_idx = MagicMock()
        split.train_idx = np.array([2, 1, 5])  # train:f1t2f2t1
        split.test_idx = np.array([6, 4, 7])   # test: f4t1f1t2
        split.val_idx = np.array([8, 10, 9])   # val: f8t3
        split.uid = str(uuid.uuid4())

        result = obj.encode_split(split)
        self.assertIn("train:f1t2f2t1,test:f4t1f1t2,val:f8t3",
                      result,
                      "Expected encoded value not returned")

    def test_encode_split_02(self):
        """Test HttpBackendExperiments.encode_split function.

        Scenario:
            1- Test expected value if test and validation set is empty.
        """
        obj = HttpBackendExperiments(self.http_client)
        split = MagicMock()
        split.train_idx = MagicMock()
        split.test_idx = MagicMock()
        split.val_idx = MagicMock()
        split.train_idx = np.array([2, 1, 5])  # train:f1t2f2t1
        split.test_idx = np.array([])   # test:
        split.val_idx = np.array([])    # val:
        split.uid = str(uuid.uuid4())

        result = obj.encode_split(split)
        self.assertIn("train:f1t2f2t1,test:,val:",
                      result,
                      "Expected encoded value not returned")

    def test_encode_split_03(self):
        """Test HttpBackendExperiments.encode_split function.

        Scenario:
            1- Test expected value if train, test and validation sets are empty.
        """
        obj = HttpBackendExperiments(self.http_client)
        split = MagicMock()
        split.train_idx = MagicMock()
        split.test_idx = MagicMock()
        split.val_idx = MagicMock()
        split.train_idx = np.array([])
        split.test_idx = np.array([])
        split.val_idx = np.array([])
        split.uid = str(uuid.uuid4())
        result = obj.encode_split(split)
        self.assertIn("train:,test:,val:",
                      result,
                      "Expected encoded value not returned")

    def test_encode_split_04(self):
        """Test HttpBackendExperiments.encode_split function.

        Scenario:
            1- Test expected value if train, test and validation sets has only one value.
        """
        obj = HttpBackendExperiments(self.http_client)
        split = MagicMock()
        split.train_idx = MagicMock()
        split.test_idx = MagicMock()
        split.val_idx = MagicMock()
        split.train_idx = np.array([1])  # train:f1t1
        split.test_idx = np.array([2])   # test:f2t1
        split.val_idx = np.array([3])   # val:f3t1
        split.uid = str(uuid.uuid4())

        result = obj.encode_split(split)
        self.assertIn("train:f1t1,test:f2t1,val:f3t1",
                      result,
                      "Expected encoded value not returned")

    def test_encode_split_05(self):
        """Test HttpBackendExperiments.encode_split function.

        Scenario:
            1- Test expected value if train, test and validation idx are None.
        """
        obj = HttpBackendExperiments(self.http_client)
        split = MagicMock()
        split.train_idx = MagicMock()
        split.test_idx = MagicMock()
        split.validation_idx = MagicMock()
        split.train_idx = None
        split.test_idx = None
        split.val_idx = None
        split.uid = str(uuid.uuid4())

        result = obj.encode_split(split)
        self.assertIn("train:,test:,val:",
                      result,
                      "Expected encoded value not returned")

    def tearDown(self):
        pass
