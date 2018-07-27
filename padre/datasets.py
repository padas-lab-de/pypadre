"""
Module containing python classes for managing data sets
- TODO we should integrate statistics of datasets somehow here. see
- TODO allow group based management of binary files similar to hdF5

"""
import numpy as np
import pandas as pd
from scipy import stats

from padre.base import MetadataEntity
from padre.utils import _const

class _Formats(_const):

    numpy = "numpy"
    pandas = "pandas"

formats = _Formats()


class NumpyContainer:

    def __init__(self, data, attributes=None):
        # todo rework binary data into delegate pattern.
        self._shape = data.shape
        if attributes is None:
            self._attributes = [Attribute(i, "RATIO") for i in range(data.shape[1])]
            self._features = data
            self._targets_idx = None
            self._features_idx = np.arange(self._shape[1])
        else:
            if len(attributes) != data.shape[1]:
                raise ValueError("Incorrect number of attributes."
                                 " Data has %d columns, provided attributes %d."
                                 % (data.shape[1], len(attributes)))
            self._data = data
            self._attributes = attributes
            self._targets_idx = np.array([idx for idx, a in enumerate(attributes) if a.is_target])
            self._features_idx = np.array([idx for idx, a in enumerate(attributes) if not a.is_target])
            assert set(self._features_idx).isdisjoint(set(self._targets_idx)) and \
                   set(self._features_idx).union(set(self._targets_idx)) == set([idx for idx in range(len(attributes))])

    @property
    def attributes(self):
        return self._attributes

    @property
    def features(self):
        if self._features_idx is None:
            return self._data
        else:
            return self._data[:, self._features_idx]

    @property
    def targets(self):
        if self._targets_idx is None:
            return None
        else:
            return self._data[:, self._targets_idx]

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape


    @property
    def num_attributes(self):
        if self._attributes is None:
            return 0
        else:
            return len(self._attributes)

    def describe(self):
        ret = {"n_att" : len(self._attributes),
               "n_target" : len([a for a in self._attributes if a.is_target])}
        if self._data is not None:
            ret["stats"] = stats.describe(self._data, axis=0)
        return ret


class PandasContainer:

    def __init__(self, data, attributes=None):
        # todo rework binary data into delegate pattern.
        self._shape = data.shape
        pd.DataFrame
        if attributes is None:
            self._attributes = [Attribute(i, "RATIO") for i in range(data.shape[1])]
            self._features = data
            self._data = data
            self._targets_idx = None
            self._features_idx = np.arange(self._shape[1])
        else:
            if len(attributes) != data.shape[1]:
                raise ValueError("Incorrect number of attributes."
                                 " Data has %d columns, provided attributes %d."
                                 % (data.shape[1], len(attributes)))
            self._data = data
            self._attributes = attributes
            self._targets_idx = np.array([idx for idx, a in enumerate(attributes) if a.is_target])
            self._features_idx = np.array([idx for idx, a in enumerate(attributes) if not a.is_target])
            assert set(self._features_idx).isdisjoint(set(self._targets_idx)) and \
                   set(self._features_idx).union(set(self._targets_idx)) == set([idx for idx in range(len(attributes))])
            #TODO assert rework

    @property
    def attributes(self):
        return self._attributes

    @property
    def features(self):
        if self._attributes is None:
            return self._data
        else:
            removekeys = []
            for att in self._attributes:
                if(att.is_target):
                    removekeys.append(att.name)
            return self._data.drop(removekeys,axis=1)

    @property
    def targets(self):
        if self._targets_idx is None:
            return None
        else:
            removekeys = []
            for att in self._attributes:
                if (not att.is_target):
                    removekeys.append(att.name)
            return self._data.drop(removekeys, axis=1)

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape


    @property
    def num_attributes(self):
        return self._data.shape[1]

  #      ret = {"n_att" : len(self._attributes),
  #             "n_target" : len([a for a in self._attributes if a.is_target])}
  #      if self._data is not None:
  #          ret["stats"] = stats.describe(self._data, axis=0)
  #      return ret


    def describe(self):
        ret = {"n_att" : len(self._attributes),
               "n_target" : len([a for a in self._attributes if a.is_target])}
        if self._data is not None:
            describe_data={}
            for col in self._data.columns:
                if isinstance(self._data[col][0],np.float64):
                    describe_data[str(col)] = stats.describe(self._data[col].values, axis=0)
            ret["status"]=describe_data

        return ret


class AttributeOnlyContainer:

    def __init__(self, attributes):
        self._attributes = attributes

    @property
    def attributes(self):
        return self._attributes

    @property
    def features(self):
        return None

    @property
    def targets(self):
        return None

    @property
    def data(self):
        return None

    @property
    def shape(self):
        return 0, 0

    @property
    def num_attributes(self):
        if self._attributes is None:
            return 0
        else:
            return len(self._attributes)

    def describe(self):
        return {"n_att" : len(self._attributes),
               "n_target" : len([a for a in self._attributes if a.is_target]),
                "stats": "no records available"}


class Attribute(dict):

    def __init__(self, name, measurement_level, unit=None,
                 description=None, is_target=False,data_class=None,nominal_values=None,number_missing_values=None):
        dict.__init__(self, name=name, measurement_level = measurement_level, unit = unit, description = description, is_target = is_target,
                      data_class=data_class,nominal_values=nominal_values,number_missing_values=number_missing_values)



        self.name = name
        self.measurement_level = measurement_level
        self.unit = unit
        self.description = description
        self.is_target = is_target


        #self.data_type=data_type
        self.data_class=data_class
        self.nominal_values=nominal_values
        self.number_missing_values=number_missing_values



    def __str__(self):
        return self.name + "(" + self.measurement_level + ")"

    def __repr__(self):
        return self.name + "(" + self.measurement_level + "/" + self.unit + ")"


class Dataset(MetadataEntity):
    """
    Unmutable in Memory Base Data Set consisting of
    1. an id (maybe null if the dataset has been newly created and is not synced)
    2. binary data (plus attributes splitting)
    3. Metadata describing the dataset
    """

    def __init__(self, id_=None, **metadata):
        super().__init__(id_, **metadata)
        self._binary = None
        self._binary_format = None


    @property
    def type(self):
        """
        returns the type of the dataset.
        :return: multivariate, matrix, graph, media
        """
        return self._metadata.get("type")

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
        if self.has_data():
            return self._binary.attributes
        else:
            return None

    @property
    def data(self):
        if self.has_data():
            return self._binary.data
        else:
            return None

    def has_data(self):
        return self._binary is not None

    @property
    def size(self):
        """
        :return: (n_examples, n_attributes)
        """
        if not self.has_data():
            return None
        else:
            return self._binary.shape

    def features(self):
        if self.has_data():
            return self._binary.features
        else:
            return None

    def targets(self):
        if self.has_data():
            return self._binary.targets
        else:
            return None

    def binary_format(self):
        """
        returns the format of the data (e.g. numpy)
        :return:
        """
        return self._binary_format


    @property
    def num_attributes(self):
        if self.has_data():
            return self._binary.num_attributes
        else:
            return 0

    def describe(self):
        if self.has_data():
            return self._binary.describe()
        else:
            return "No records available"



    """
    sets the binary data and descriptive attributes
    :param data: binary data in a supported format (numpy, pandas): size must be num_datasets x num_attributes
    :param attributes: Description for attributes or None. If none, the attributes will be estimated as good as possible
    """
    def set_data(self, data, attributes=None):
        if data is None:
            self._binary_format = None
            self._binary = AttributeOnlyContainer(attributes)
        elif isinstance(data, pd.DataFrame):
            self._binary = PandasContainer(data, attributes)
            self._binary_format = formats.pandas
        elif isinstance(data, np.ndarray):
            self._binary = NumpyContainer(data, attributes)
            self._binary_format = formats.numpy
        else:
            raise ValueError("Unknown data format. Type %s not known." % (type(data)))

    def __str__(self):
        return str(self.id) +"_"+ str(self.name) + ": " + str(self.type) + ", " + str(self.size) + ", " + str(self.binary_format())


