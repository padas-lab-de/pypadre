"""
This file contains tests covering backend.http_experiments.HttpBackendExperiments.decode_split
All unnecessary function and http calls are mocked
"""
import unittest

import numpy as np
from mock import MagicMock

from pypadre.pod.backend.http.http_experiments import HttpBackendExperiments


class TestDecodeSplit(unittest.TestCase):
    """Test http_experiments.HttpBackendExperiments.decode_split
    """

    def setUp(self):
        """Initializing for decode_split test.
        """
        self.http_client = MagicMock()
        self.http_client.user = 'test user'
        self.http_client.base = 'padretest.com/api'
        self.http_client.has_token = MagicMock(return_value=False)

    def test_decode_split_01(self):
        """Test HttpBackendExperiments.decode_split function.

        Scenario:
            1- Expected decoded value should be returned for train, test and val.
            2- Expected type of each list should be nd.array
        """
        obj = HttpBackendExperiments(self.http_client)
        train_idx = [1, 2, 5]  # train:f1t2f2t1
        test_idx = [4, 6, 7]   # test: f4t1f1t2
        val_idx = [8, 9, 10]   # val: f8t3
        split = "train:f1t2f2t1,test:f4t1f1t2,val:f8t3"
        result = obj.decode_split(split)
        self.assertListEqual(train_idx,
                      list(result["train"]),
                      "Expected decoded value for train idx is not returned")
        self.assertListEqual(test_idx,
                             list(result["test"]),
                             "Expected decoded value for test idx is not returned")
        self.assertListEqual(val_idx,
                             list(result["val"]),
                             "Expected decoded value for val idx is not returned")
        self.assertTrue(type(result["train"]) == np.ndarray,
                        "Expected type of train idx is not np.ndarray")
        self.assertTrue(type(result["test"]) == np.ndarray,
                        "Expected type of test idx is not np.ndarray")
        self.assertTrue(type(result["val"]) == np.ndarray,
                        "Expected type of val idx is not np.ndarray")

    def test_decode_split_02(self):
        """Test HttpBackendExperiments.decode_split function.

        Scenario:
            1- Expected decoded value should be returned for train, test and val for single element lists.
        """
        obj = HttpBackendExperiments(self.http_client)
        train_idx = [1]  # train:f1t1
        test_idx = [2]  # test:f2t1
        val_idx = [3]  # val:f3t1
        split = "train:f1t1,test:f2t1,val:f3t1"
        result = obj.decode_split(split)
        self.assertListEqual(train_idx,
                      list(result["train"]),
                      "Expected decoded value for train idx is not returned")
        self.assertListEqual(test_idx,
                             list(result["test"]),
                             "Expected decoded value for test idx is not returned")
        self.assertListEqual(val_idx,
                             list(result["val"]),
                             "Expected decoded value for val idx is not returned")

    def test_decode_split_03(self):
        """Test HttpBackendExperiments.decode_split function for empty encodings.

        Scenario:
            1- Expected None value should be returned for train, test and val if encoding is empty.
        """
        obj = HttpBackendExperiments(self.http_client)
        split = "train:,test:,val:"
        result = obj.decode_split(split)
        self.assertIsNone(result["train"],
                          "Expected None value for train idx is not returned")
        self.assertIsNone(result["test"],
                          "Expected None value for test idx is not returned")
        self.assertIsNone(result["val"],
                          "Expected None value for val idx is not returned")

    def tearDown(self):
        pass
