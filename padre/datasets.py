
"""
Module containing python classes for managing data sets
TODO we should integrate statistics of datasets somehow here
"""
import abc
from .metadata import Metadata


class BaseDataset(object):
    """
    Unmutable in Memory Base Data Set
    provides a iterator over
    TODO: implement metadata management properly
    """
    def __init__(self, name, **metadata):
        self._name = name
        self._type = None
        self._data = None
        self._target = None
        self._metadata = Metadata(metadata)
        md = self._metadata
        self._attributes = md.get("attributes")
        self._attributes_types = md.get("attributes_types")
        self._type = md.get("type")

    @property
    def name(self):
        """
        returns the unique name of the data set
        :return: string
        """
        return self._name

    @property
    def type(self):
        """
        returns the type of the dataset.
        :return: multivariate, matrix, graph, media
        """
        return self._type

    @property
    def metadata(self):
        """
        returns the metadata object associated with this dataset
        :return:
        """
        return self._metadata

    @property
    def attributes(self):
        """
        returns the metadata object associated with this dataset
        :return:
        """
        return self._attributes

    @property
    def attributes_types(self):
        """
        returns the metadata object associated with this dataset
        :return:
        """
        return self._attributes_types

    @property
    def target(self):
        return self._target

    @property
    def data(self):
        return self._data

    @abc.abstractmethod
    def format(self):
        """
        returns the format of the data (e.g. numpy)
        :return:
        """
        pass

    @abc.abstractmethod
    def size(self):
        """
        :return: (n_examples, n_attributes)
        """
        pass

    def __str__(self):
        return self._name + ": " + self.type + ", " + str(self.size()) + ", " + self.format()


class NumpyBaseDataset(BaseDataset):
    """
    Unmutable in Memory Base Data Set using Numpy
    """
    def __init__(self,  name, data, target, **metadata):
        """
        :param data: numpy array with n_examples times n_features
        :param target: numpy array with n_examples x 1 target values
        :param attributes: attributes. must be of length n_features
        """
        super().__init__(name, **metadata)
        self._data = data
        self._target = target
        self.metadata["format"] = "numpy"
        # check attribute defaults
        if self.attributes is None:
            self._attributes = [i for i in range(self._data.shape[1])]
        if self.attributes_types is None:
            #TODO do a more intelligent data type mapping from the numpy arry
            self._attributes_types = ["ratio" for _ in range(self.data.shape[1])]

        assert len(self._attributes) == self._data.shape[1]
        assert len(self._attributes_types) == self._data.shape[1]

    def format(self):
        """
        returns the format of the dataset
        :return: numpy, pandas, edge_list, pyarrow
        """
        return self.metadata["format"]

    def size(self):
        """
        :return: (n_examples, n_attributes)
        """
        return self._data.shape


def new_dataset(name, metadata, data, target):

    if metadata["format"] == "numpy":
        return NumpyBaseDataset(name, data, target, **metadata)

    raise TypeError("Unkown dataset type "+metadata["type"])

    







