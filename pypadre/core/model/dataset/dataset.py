"""
Module containing python classes for managing data sets
- TODO we should integrate statistics of datasets somehow here. see
- TODO allow group based management of binary files similar to hdF5

"""
import networkx as nx
import numpy as np
import pandas as pd
from jsonschema import ValidationError
from padre.PaDREOntology import PaDREOntology
from scipy.stats.stats import DescribeResult

from pypadre.base import MetadataEntity
from pypadre.core.model.dataset.container.graph_container import GraphContainer
from pypadre.core.model.dataset.container.base_container import AttributesOnlyContainer, IBaseContainer
from pypadre.core.model.dataset.container.numpy_container import NumpyContainer
from pypadre.core.model.dataset.container.pandas_container import PandasContainer
from pypadre.importing.base_validator import IValidator
from pypadre.printing.tablefyable import Tablefyable
from pypadre.printing.util.print_util import StringBuilder, get_default_table
from pypadre.util.dict_util import get_dict_attr
from pypadre.util.utils import _const


class _Formats(_const):
    numpy = "numpy"
    numpyMulti = "numpyMultiDim"
    pandas = "pandas"
    graph = "graph"


formats = _Formats()


class DataSetValidator(IValidator):

    @staticmethod
    def validate(obj):
        # TODO validate if metadata are fine. This should prompt the user to input something if the validation fails.

        if not obj.metadata.get("name"):
            raise ValidationError("name has to be set for a dataset")

        if not obj.metadata.get("version"):
            raise ValidationError("version has to be set for a dataset")

        if not obj.metadata.get("originalSource"):
            raise ValidationError("originalSource has to be set for a dataset")

        if not obj.metadata.get("type"):
            raise ValidationError("type has to be set for a dataset")


class Dataset(MetadataEntity, Tablefyable):

    def __init__(self, id_=None, **metadata):
        super().__init__(id_, **{**{"name": "", "version": "1.0", "description": "", "originalSource": "", "type": "",
                                    "published": False}, **self._metadata})
        self._binaries = dict()

        # Add entries for tablefyable
        self._registry.update({'id': get_dict_attr(self, 'id').fget, 'name': get_dict_attr(self, 'name').fget,
                               'type': get_dict_attr(self, 'type').fget})

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
    def attributes(self, bin_format=None):
        container: IBaseContainer = self.container(bin_format)
        return container.attributes()

    def container(self, bin_format=None):
        """
        Gets the container holding the data with given format. If no format is given we just get the only container
        we have or throw an error. This implicitly tries to convert between containers. :param bin_format: :return:
        """
        # Try to get data of unknown binary format
        if bin_format is None:
            if len(self._binaries) == 0:
                raise ValueError("No binary exists.")
            if len(self._binaries) > 1:
                raise ValueError("More than one binary exists. Pass a format.")
            else:
                return next(iter(self._binaries))

        if bin_format in self._binaries:
            return self._binaries.get(bin_format)
        elif len(self._binaries) > 1:
            for key, value in self._binaries.items():
                container: AttributesOnlyContainer = value
                out = container.convert(bin_format)
                if out is not None:
                    return out
            raise ValueError("Couldn't convert to %s." % bin_format)

    def has_container(self, bin_format):
        """
        Check if container exists
        :param bin_format:
        :return:
        """
        return bin_format in self._binaries

    def add_container(self, container):
        """
        Add a container to our dataset
        :param container:
        :return:
        """
        self._binaries[container.format] = container

    def data(self, bin_format=None):
        """
        Get data for given format
        :param bin_format: format
        :return:
        """
        return self.container(bin_format).data

    def set_data(self, data, attributes=None):
        """
        Set new data. Container type can be derived automatically. TODO What about attributes?
        :param data: Data to set
        :param attributes: Attributes of the data
        :return:
        """

        if isinstance(data, pd.DataFrame):
            container = PandasContainer(data, attributes)
        elif isinstance(data, np.ndarray):
            container = NumpyContainer(data, attributes)
        elif isinstance(data, nx.Graph):
            container = GraphContainer(data, attributes)
        else:
            raise ValueError("Unknown data format. Type %s not known." % (type(data)))

        # Add the binary
        self.add_container(container)

    def __str__(self):
        return str(self.id) + "_" + str(self.name) + ": " + str(self.type)

    @property
    def size(self, bin_format=None):
        """
        :return: (n_examples, n_attributes)
        """
        container: IBaseContainer = self.container(bin_format)
        return container.shape()

    def describe(self, bin_format=None):
        container: IBaseContainer = self.container(bin_format)
        return container.describe()

    def profile(self, bin_format=None, bins=50, check_correlation=True, correlation_threshold=0.8,
                correlation_overrides=None, check_recoded=False):
        # TODO check is wrong if the parameters for profiling aren't the same
        if "profile" in self.metadata:
            return self.metadata["profile"]
        container: IBaseContainer = self.container(bin_format)
        return container.profile(bins=bins, check_correlation=check_correlation,
                                 correlation_threshold=correlation_threshold,
                                 correlation_overrides=correlation_overrides,
                                 check_recoded=check_recoded)

    # def profile(self, bins=50, check_correlation=True, correlation_threshold=0.8,
    #             correlation_overrides=None, check_recoded=False):
    #     if "profile" in self.metadata:
    #         return self.metadata["profile"]
    #     elif self.data is not None:
    #         profile = self.data.profile(bins, check_correlation, correlation_threshold,
    #                                    correlation_overrides, check_recoded).get_description()
    #         profile["variables"] = profile["variables"].to_dict(orient="index")
    #
    #         # TODO this can't be done without saving the variables for the profile call
    #         self.metadata["profile"] = profile
    #
    #         _check_profiling_datatype(profile["variables"])
    #         _check_profiling_datatype(profile["table"])
    #
    #         for key in profile["freq"].keys():
    #             profile["freq"][key]=profile["freq"][key].to_dict()
    #
    #         for key in profile["correlations"].keys():
    #             profile["correlations"][key]=profile["correlations"][key].to_dict()
    #
    #         return self.metadata["profile"]
    #     else:
    #         return "No records available"
    #
    # def _check_profiling_datatype(content):
    #     if isinstance(content, dict):
    #         for key in content.keys():
    #             if key == "histogram" or key == "mini_histogram" or content[key] is np.nan:
    #                 content[key] = None
    #             elif isinstance(content[key], np.int32) or isinstance(content[key], np.int64):
    #                 content[key] = int(content[key])
    #             elif isinstance(content[key], np.bool_):
    #                 content[key] = bool(content[key])
    #             elif isinstance(content[key], list) or isinstance(content[key], dict):
    #                 _check_profiling_datatype(content[key])

    def to_detail_string(self):
        sb = StringBuilder()
        sb.append_line(f"Metadata for dataset {self.id}")
        for k, v in self.metadata.items():
            sb.append_line("\t%s=%s" % (k, str(v)))
        sb.append_line("Binary description:")
        for k, v in self.describe().items():
            # todo printing the statistics is not ideal. needs to be improved
            if k == "stats" and isinstance(v, DescribeResult):
                table = get_default_table()
                h = ["statistic"]
                for a in self.attributes:
                    h.append(a.name)
                table.column_headers = h
                for m in [("min", v.minmax[0]), ("max", v.minmax[1]), ("mean", v.mean),
                          ("kurtosis", v.kurtosis), ("skewness", v.skewness)]:
                    r = [m[0]]
                    for val in m[1]:
                        r.append(val)
                    table.append_row(r)
                sb.append(table)
            else:
                sb.append_line("\t%s=%s" % (k, str(v)))
        return sb


class Transformation(Dataset):

    def __init__(self, dataset, id_=None, **metadata):
        super().__init__(id_, **metadata)
        self._dataset = dataset
        # todo rework preprocessing

# class DatasetOld(MetadataEntity, Tablefyable):
#     """
#     Unmutable in Memory Base Data Set consisting of
#     1. an id (maybe null if the dataset has been newly created and is not synced)
#     2. binary data (plus attributes splitting)
#     3. Metadata describing the dataset
#     """
#
#     def __init__(self, id_=None, **metadata):
#         super().__init__(id_, **metadata)
#         self._binary = None
#         self._binary_loader_fn = None
#         self._binary_format = None
#         self._fill_metadata()
#
#         # TODO: we should just pass attribute kwargs without care for structure here
#         if "attributes" in metadata and metadata.get("attributes"):
#             self.set_data(None, [Attribute(a["name"], a["measurementLevel"], a["unit"], a["description"],
#                                 a["defaultTargetAttribute"], a["context"], a["index"])
#                                  for a in metadata.get("attributes")])
#
#         # Add entries for tablefyable
#         self._registry.update({'id': get_dict_attr(self, 'id').fget, 'name': get_dict_attr(self, 'name').fget,
#                                'type': get_dict_attr(self, 'type').fget, 'size': get_dict_attr(self, 'size').fget,
#                                'format': get_dict_attr(self, 'binary_format')})
#
#     def _get_binary(self):
#         """
#         returns the binary. In case that a binary_loader_fn is given, the binary is lazy loaded.
#         :return:
#         """
#         if self._binary is not None:
#             return self._binary.data
#         elif self._binary_loader_fn is not None:
#             self.set_data(*self._binary_loader_fn())  # derefered or lazy loading
#         return self._binary
#
#     @property
#     def type(self):
#         """
#         returns the type of the dataset.
#         :return: multivariate, matrix, graph, media
#         """
#         return self._metadata.get("type")
#
#     @property
#     def metadata(self):
#         """
#         returns the metadata object associated with this dataset
#         :return:
#         """
#         return self._metadata
#
#     @property
#     def attributes(self):
#         """
#         returns the metadata object associated with this dataset
#         :return:
#         """
#         if self.has_data():
#             return self._binary.attributes
#         else:
#             return None
#
#     @property
#     def data(self):
#         """
#         should not be used to access data. @look ap .pandas_repr
#         :return: the stored _binary. Can be numpy-array, pandas-DataFrame or networkx.Graph object
#         """
#         if self.has_data():
#             return self._binary.data
#         else:
#             return None
#
#     def has_data(self):
#         return self._get_binary() is not None
#
#     @property
#     def size(self):
#         """
#         :return: (n_examples, n_attributes)
#         """
#         if not self.has_data():
#             return None
#         else:
#             return self.data.shape
#
#     @property
#     def isgraph(self):
#         if "type" in self.metadata:
#             return self.metadata["type"] == "graph" or self.metadata["type"] == "graphDirected"
#         else:
#             return False
#
#     def _fill_metadata(self):
#         keys = ["name", "version", "description", "originalSource", "type"]
#         for key in keys:
#             if key not in self.metadata:
#                 self.metadata[key] = ""
#         if "published" not in self.metadata:
#             self.metadata["published"] = False
#
#     def pandas_repr(self):
#         """
#         :return: The pandas representation of the dataset. converts Numpy-array, pandas-df, and nx.Graph objects to pandas DF
#         """
#         if self.has_data():
#             return None
#         else:
#             return self.data.pandas_repr()
#
#     def features(self):
#         if self.has_data():
#             return self._binary.features
#         else:
#             return None
#
#     def targets(self):
#         if self.has_data():
#             return self._binary.targets
#         else:
#             return None
#
#     def binary_format(self):
#         """
#         returns the format of the data (e.g. numpy)
#         :return:
#         """
#         return self._binary_format
#
#     def get_target_attribute(self):
#         """
#         Return default target attribute
#         :return:
#         """
#         for attr in self.attributes:
#             if attr["defaultTargetAttribute"]:
#                 return attr["name"]
#         return None
#
#     @property
#     def num_attributes(self):
#         if self.has_data():
#             return self._binary.num_attributes
#         else:
#             return 0
#
#     def profile(self, bins=50, check_correlation=True, correlation_threshold=0.8,
#                 correlation_overrides=None, check_recoded=False):
#         if "profile" in self.metadata:
#             return self.metadata["profile"]
#         elif self.data is not None:
#             profile = self.data.profile(bins, check_correlation, correlation_threshold,
#                                        correlation_overrides, check_recoded).get_description()
#             profile["variables"] = profile["variables"].to_dict(orient="index")
#
#             # TODO this can't be done without saving the variables for the profile call
#             self.metadata["profile"] = profile
#
#             _check_profiling_datatype(profile["variables"])
#             _check_profiling_datatype(profile["table"])
#
#             for key in profile["freq"].keys():
#                 profile["freq"][key]=profile["freq"][key].to_dict()
#
#             for key in profile["correlations"].keys():
#                 profile["correlations"][key]=profile["correlations"][key].to_dict()
#
#             return self.metadata["profile"]
#         else:
#             return "No records available"
#
#     def describe(self):
#         if self.data is not None:
#             return self._binary.describe()
#         else:
#             return "No records available"
#
#     def set_data(self, data, attributes=None):
#         """
#         sets the binary data and descriptive attributes. If data is a function, it is expected that the function
#         returns a tuple (data, attributes) which can be used to call set_data at a later point in time and thus support
#         lazy loading.
#         :param data: binary data in a supported format (numpy, pandas, networkx):
#                     size must be num_datasets x num_attributes. If data is a function,
#                     it is expected that the function is called later.
#         :param attributes: Description for attributes or None.
#                            If none, the attributes will be estimated as good as possible
#
#         """
#         self._binary = None
#         self._binary_format = None
#         if data is None:
#             self._binary_format = None
#             self._binary = AttributeOnlyContainer(attributes)
#         elif hasattr(data, '__call__'):
#             self._binary_loader_fn = data
#             return
#         elif isinstance(data, pd.DataFrame):
#             # Remove non numerical attributes
#             # TODO: Bring in support for nominal and ordinal attributes too
#             self._binary = PandasContainer(data, attributes)
#             self._binary_format = formats.pandas
#         elif isinstance(data, np.ndarray):
#             self._binary = NumpyContainer(data, attributes)
#             self._binary_format = formats.numpy
#         elif isinstance(data, nx.Graph):
#             self._binary = GraphContainer(data, attributes)
#             self._binary_format = formats.graph
#         else:
#             raise ValueError("Unknown data format. Type %s not known." % (type(data)))
#         self._fill_metadata()
#
#     def set_data_multidimensional(self, features, targets, attributes=None):
#         """
#         Sets the data for multidimensional feature vectors like images
#         :param features: Input features
#         :param targets: Targets
#         :param attributes: Attribute list
#         :return:
#         """
#         self._binary = NumpyContainerMultiDimensional(features, targets, attributes)
#
#     def replace_data(self, data, keep_atts=True):
#         """
#         Function is used to hold a temporary dataset of preprocessed data
#         :param data: binary data in a supported format (numpy, pandas, networkx):
#                     size must be num_datasets x num_attributes. If data is a function,
#                     it is expected that the function is called later.
#         :param keep_atts: a boolean parameter set by the user to precise whether the
#                     attributes will be changed after preprocessing or not.
#
#         :return: None
#         """
#
#         if data is None:
#             self._binary_format = None
#             self._binary = AttributeOnlyContainer(self.attributes)
#         elif hasattr(data, '__call__'):
#             self._binary_loader_fn = data
#             return
#         elif isinstance(data, pd.DataFrame):
#             # Remove non numerical attributes
#             # TODO: Bring in support for nominal and ordinal attributes too
#             if keep_atts:
#                 # if attributes are not changed (In most cases) we append the targets to data
#                 # since transformers only return the features
#                 if data.columns.values.shape[0] != len(self.attributes):
#                     target_att = []
#                     for att in self.attributes:
#                         if att.defaultTargetAttribute:
#                             target_att.append(att.name)
#                     data[target_att] = self.targets()
#                 self._binary = PandasContainer(data, self.attributes)
#             else:
#                 attributes = []
#                 for i, feature in enumerate(data.columns.values):
#                     attributes.append(Attribute(name=feature, measurementLevel=None, unit=None, description=None,
#                                                 defaultTargetAttribute=(i == data.columns.values.shape[0] - 1),
#                                                 context=None))
#                 self._binary = PandasContainer(data, attributes)
#             self._binary_format = formats.pandas
#         elif isinstance(data, np.ndarray):
#             # Append the target data to the original data if targets are not modified
#             # If targets are modified, pass the incoming data as both features and targets
#             if keep_atts:
#                 # if attributes are not changed (In most cases) we append the targets to data
#                 # since transformers only return the features
#                 if data.shape[1] != self.data.shape[1:]:
#                     data = np.append(data, self.targets(), axis=1)
#                 self._binary = NumpyContainer(data, self.attributes)
#             else:
#                 attributes = []
#                 for i in range(data.shape[1]):
#                     attributes.append(Attribute(name=i, measurementLevel=None, unit=None, description=None,
#                                                 defaultTargetAttribute=(i == data.shape[1] - 1),
#                                                 context=None))
#                 self._binary = NumpyContainer(data, attributes)
#
#             self._binary_format = formats.numpy
#         elif isinstance(data, nx.Graph):
#             self._binary = GraphContainer(data, self.attributes)
#             self._binary_format = formats.graph
#         else:
#             raise ValueError("Unknown data format. Type %s not known." % (type(data)))
#
#     def __str__(self):
#         return str(self.id) + "_" + str(self.name) + ": " + str(self.type) + ", " + str(self.size) + ", " + str(self.binary_format())
#
#     def to_detail_string(self):
#         sb = StringBuilder()
#         sb.append_line(f"Metadata for dataset {self.id}")
#         for k, v in self.metadata.items():
#             sb.append_line("\t%s=%s" % (k, str(v)))
#         sb.append_line("Binary description:")
#         for k, v in self.describe().items():
#             # todo printing the statistics is not ideal. needs to be improved
#             if k == "stats" and isinstance(v, DescribeResult):
#                 table = get_default_table()
#                 h = ["statistic"]
#                 for a in self.attributes:
#                     h.append(a.name)
#                 table.column_headers = h
#                 for m in [("min", v.minmax[0]), ("max", v.minmax[1]), ("mean", v.mean),
#                           ("kurtosis", v.kurtosis), ("skewness", v.skewness)]:
#                     r = [m[0]]
#                     for val in m[1]:
#                         r.append(val)
#                     table.append_row(r)
#                 sb.append(table)
#             else:
#                 sb.append_line("\t%s=%s" % (k, str(v)))
#         return sb
#
#
# def _check_profiling_datatype(content):
#     if isinstance(content,dict):
#         for key in content.keys():
#             if key == "histogram" or key == "mini_histogram" or content[key] is np.nan:
#                 content[key]=None
#             elif isinstance(content[key], np.int32)or isinstance(content[key], np.int64):
#                 content[key] = int(content[key])
#             elif isinstance(content[key], np.bool_):
#                 content[key] = bool(content[key])
#             elif isinstance(content[key],list) or isinstance(content[key],dict):
#                 _check_profiling_datatype(content[key])
