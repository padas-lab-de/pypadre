"""
This file contains tests covering padre.ds_import.load_csv
All unnecessary function calls are mocked
"""
import unittest

import numpy as np
import pandas as pd
from mock import MagicMock, patch

from padre import ds_import


class TestLoadCSV01(unittest.TestCase):
    """Test ds_import.load_csv

    Test expected target attributes are set in the dataset
    """

    def setUp(self):
        """Initializing for load_csv test.
        """
        self.file_name = "test.csv"

    @patch('padre.ds_import.assert_condition')
    @patch('padre.ds_import.pd.read_csv')
    def test_load_csv_01(self, mocked_csv, mocked_assert):
        """Test target attributes for ds_import.load_csv function.

        Scenario:
            1- Single target attribute is set.
        """
        test_df = pd.DataFrame(
            np.array([[1, 2, 1], [2, 5, 1], [2, 8, 0]]),
            columns=['c1', 'c2', 'class'])
        targets = ["class"]
        mocked_assert.return_value = True
        mocked_csv.return_value = test_df
        result = ds_import.load_csv_new(self.file_name, targets)
        for target in result._binary._targets_idx:
            self.assertTrue(result.attributes[target].name in targets,
                            "Expected target attribute not matches")

    @patch('padre.ds_import.assert_condition')
    @patch('padre.ds_import.pd.read_csv')
    def test_load_csv_02(self, mocked_csv, mocked_assert):
        """Test target attributes for ds_import.load_csv function.

        Scenario:
            1- Multiple target attributes are set.
        """
        test_df = pd.DataFrame(
            np.array([[1, 2, 1], [2, 5, 1], [2, 8, 0]]),
            columns=['c1', 'c2', 'class'])
        targets = ["class", "c1"]
        mocked_assert.return_value = True
        mocked_csv.return_value = test_df
        result = ds_import.load_csv_new(self.file_name, targets)
        for target in result._binary._targets_idx:
            self.assertTrue(result.attributes[target].name in targets,
                            "Expected target attribute not matches")

    def tearDown(self):
        pass


class TestLoadCSV02(unittest.TestCase):
    """Test ds_import.load_csv for attributes and data

    Test expected target attributes are set in the dataset
    and data arrays are equal between expected and actual data
    """

    def setUp(self):
        """Initializing for load_csv test.
        """
        self.name = "Test database"
        self.file_name = "test.csv"
        self.description = "test description"


    @patch('padre.ds_import.assert_condition')
    @patch('padre.ds_import.pd.read_csv')
    def test_load_csv_01(self, mocked_csv, mocked_assert):
        """Test metadata attributes for ds_import.load_csv function.

        Scenario:
            1- Expected name of dataset.
            2- Expected description of dataset.
            3- Expected type of dataset.
            4- Expected original source of dataset.
            5- Expected version of dataset.
        """
        test_df = pd.DataFrame(
            np.array([[1, 2, 1], [2, 5, 1], [2, 8, 0]]),
            columns=['c1', 'c2', 'class'])
        targets = ["class"]
        mocked_assert.return_value = True
        mocked_csv.return_value = test_df
        result = ds_import.load_csv_new(self.file_name,
                                        targets,
                                        name=self.name,
                                        description=self.description)
        self.assertEqual(self.name,
                         result.metadata["name"],
                         "Expected dataset name not matches")
        self.assertEqual(self.description,
                         result.metadata["description"],
                         "Expected dataset description not matches")
        self.assertEqual("http://csvloaded",
                         result.metadata["originalSource"],
                         "Expected dataset original source not matches")
        self.assertEqual("Multivariat",
                         result.metadata["type"],
                         "Expected dataset type not matches")
        self.assertEqual(1,
                         result.metadata["version"],
                         "Expected dataset version not matches")

    @patch('padre.ds_import.assert_condition')
    @patch('padre.ds_import.pd.read_csv')
    def test_load_csv_02(self, mocked_csv, mocked_assert):
        """Test metadata attributes and data for ds_import.load_csv function.

        Scenario:
            1- Expected attributes are set in metadata.
            2- Numpy data arrays are equal
        """
        columns = ['c1', 'c2', 'class']
        test_df = pd.DataFrame(
            np.array([[1, 2, 1], [2, 5, 1], [2, 8, 0]]),
            columns=columns)
        mocked_assert.return_value = True
        mocked_csv.return_value = test_df
        result = ds_import.load_csv_new(self.file_name,
                                        ["class"],
                                        name=self.name,
                                        description=self.description)
        for column in columns:
            self.assertTrue(any(column==attr.name for attr in result.attributes),
                            "Expected column not found in dataset attributes")

        np.testing.assert_array_equal(test_df.values, result.data.values)

    def tearDown(self):
        pass