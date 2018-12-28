""""
Unit test testing the functionality of the padre.datasets module
"""
import unittest
import numpy as np

from padre.datasets import Dataset
from padre.datasets import Attribute

test_numpy_array = np.array([[1.0, "A", 2],
                        [2.0, "B", 2],
                        [3.0, "A", 3],
                        [3.0, "C", 4]])

test_attributes = [{"name": "ratio_att", "measurementLevel": "RATIO",
               "unit": None, "description": "A ratio scaled attribute",
               "defaultTargetAttribute": False, "context": None, "index": 0},
              {"name": "nominal_att", "measurementLevel": "NOMINAL",
               "unit": "count", "description": "A nominal attribute",
               "defaultTargetAttribute": True, "context": None, "index": 1},
              {"name": "ordered attribute", "measurementLevel": "ORDINAL",
               "unit": None, "description": "A ratio scaled attribute",
               "defaultTargetAttribute": False, "context": None, "index": 2},
              ]

test_metadata = {
    "version": "0.12",
    "url": "xyz",
    "something": 1,
}


class TestDataset(unittest.TestCase):




    def setUp(self):
        pass

    def test_create_attribute(self):
        for att in test_attributes:
            a = Attribute(**att)
            assert a.defaultTargetAttribute == att["defaultTargetAttribute"]
            assert a.unit == att["unit"]
            assert a.name == att["name"]
            assert a.description == att["description"]
           # todo check the context whether it contains the same items. should be a dict.
            assert a.index == att["index"]

    def test_create_dataset(self):
        ds = Dataset("test", **test_metadata)
        ds.set_data(test_numpy_array)
        assert ds.id == "test"
        assert ds.metadata is not None and len(ds.metadata) == len(test_metadata)+5  # some metadata get added.
        # todo check in more detail, that the correct metadata has been added
        for k, v in test_metadata.items():
            assert ds.metadata[k] == v
        assert ds.has_data()
        assert len(ds.attributes) == test_numpy_array.shape[1]
        assert np.array_equal(ds.data, test_numpy_array)
        # Todo: check correct attributes
        # Test lazy loading here
        ds.set_data(lambda: (test_numpy_array, None))
        assert np.array_equal(ds.data, test_numpy_array)
        # Test lazy loading on new object.
        ds = Dataset("test", **test_metadata)
        ds.set_data(lambda: (test_numpy_array, None))
        assert np.array_equal(ds.data, test_numpy_array)



