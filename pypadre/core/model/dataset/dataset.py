"""
Module containing python classes for managing data sets
- TODO we should integrate statistics of datasets somehow here. see
- TODO allow group based management of binary files similar to hdF5

"""
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd
from padre.PaDREOntology import PaDREOntology
from scipy.stats.stats import DescribeResult

from pypadre.core.base import MetadataMixin
from pypadre.core.model.dataset.container.base_container import IBaseContainer
from pypadre.core.model.dataset.container.graph_container import GraphContainer
from pypadre.core.model.dataset.container.numpy_container import NumpyContainer
from pypadre.core.model.dataset.container.pandas_container import PandasContainer
from pypadre.core.model.generic.i_storable_mixin import StoreableMixin
from pypadre.core.printing.util.print_util import StringBuilder, get_default_table
from pypadre.core.util.utils import _Const
from pypadre.core.validation.json_validation import make_model


class _Formats(_Const):
    numpy = "numpy"
    numpyMulti = "numpyMultiDim"
    pandas = "pandas"
    graph = "graph"


formats = _Formats()

dataset_model = make_model(schema_resource_name='dataset.json')


class Dataset(StoreableMixin, MetadataMixin):

    def __init__(self, **kwargs):
        """
        :param id :
        :param attributes: Attributes of the data
        :param metadata:
        """

        # Add defaults
        defaults = {"name": "default_name", "version": "1.0", "description": "", "originalSource": "",
                    "type": PaDREOntology.SubClassesDataset.Multivariat.value, "published": False, "attributes": [], "targets": []}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {})}

        super().__init__(model_clz=dataset_model, metadata=metadata, **kwargs)

        self._binaries = dict()
        self._proxy_loaders = {}

    def add_proxy_loader(self, fn: Callable):
        self._proxy_loaders[fn.__hash__()] = lambda: self.set_data(data=fn())

    @property
    def name(self):
        return self.metadata["name"]

    @property
    def type(self):
        """
        returns the type of the dataset.
        :return: multivariate, matrix, graph, media
        """
        return self.metadata.get("type")

    @property
    def attributes(self):
        return self.metadata.get("attributes")

    def _execute_proxy_loaders(self):
        for key in list(self._proxy_loaders.keys()):
            self._proxy_loaders.pop(key)()

    def container(self, bin_format=None):
        """
        Gets the container holding the data with given format. If no format is given we just get the only container
        we have or throw an error. This implicitly tries to convert between containers. :param bin_format: :return:
        """
        # Try to get data of unknown binary format
        if bin_format is None:
            if len(self._binaries) == 0:
                if self._proxy_loaders.__len__() > 0:
                    self.send_info(message="Trying to load proxied object.")
                    self._execute_proxy_loaders()
                    return self.container(bin_format)
                raise ValueError("No binary exists.")
            if len(self._binaries) > 1:
                raise ValueError("More than one binary exists. Pass a format.")
            else:
                # return next(iter(self._binaries))
                return self._binaries.get(next(iter(self._binaries)))

        if bin_format in self._binaries:
            return self._binaries.get(bin_format)
        elif len(self._binaries) > 1:
            for key, value in self._binaries.items():
                container: IBaseContainer = value
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

    def set_attributes(self, attributes=None):
        if self.attributes is None or len(self.attributes) == 0:
            self._metadata['attributes'] = attributes
        else:
            pass

    def set_data(self, data):
        """
        Set new data. Container type can be derived automatically.
        :param data: Data to set
        :return:
        """

        if self.attributes is None or len(self.attributes) == 0:
            self.send_warn(message='Dataset has no attributes yet! Attempting to derive them from the binary using '
                                   'targets metadata if exits')
            if isinstance(data, pd.DataFrame):
                attributes = PandasContainer.derive_attributes(data, targets=self.metadata.get("targets", None))
                container = PandasContainer(data, attributes)
            elif isinstance(data, np.ndarray):
                attributes = NumpyContainer.derive_attributes(data, targets=self.metadata.get("targets", None))
                container = NumpyContainer(data, attributes)
            elif isinstance(data, nx.Graph):
                attributes = GraphContainer.derive_attributes(data, targets=self.metadata.get("targets", None))
                container = GraphContainer(data, attributes)
            else:
                raise ValueError("Unknown data format. Type %s not known." % (type(data)))
            self.set_attributes(attributes)
        else:
            if isinstance(data, pd.DataFrame):
                container = PandasContainer(data, self.attributes)
            elif isinstance(data, np.ndarray):
                container = NumpyContainer(data, self.attributes)
            elif isinstance(data, nx.Graph):
                container = GraphContainer(data, self.attributes)
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
        return container.shape

    def targets(self, bin_format=None):
        container: IBaseContainer = self.container(bin_format)
        return container.targets

    def features(self, bin_format=None):
        container: IBaseContainer = self.container(bin_format)
        return container.features

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

    '''
    def get(self, key):
        if key == 'id':
            return self.name

        else:
            return self.__dict__.get(key, None)
    '''

    def id_hash(self):
        # TODO create a hash of the dataset
        return self.name

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
                if self.attributes is None:
                    sb.append("No attribute metadata found on " + str(self))
                    # TODO inform user about problems
                else:
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
        return str(sb)


class Transformation(Dataset):

    def __init__(self, dataset: Dataset, **kwargs):
        """

        :param dataset: The original dataset to transform/preprocess
        :param kwargs: metadata like name, attributes, ...
        """
        # Add defaults
        defaults = {"name": dataset.name, "version": "1.0", "description": dataset.metadata["description"],
                    "originalSource": dataset.id,
                    "type": dataset.type, "published": False, "attributes": dataset.attributes}

        metadata = {**defaults,**kwargs}
        super().__init__(**metadata)
        self._dataset = dataset
        self._binaries = dict()

    def update_attributes(self, attributes=None):
        """
        Update the attributes in case the transformation changes the attributes
        :param attributes:
        :return:
        """
        self.metadata["attributes"] = attributes

    def set_data(self, data, attributes=None):
        """
        Set the transformed data and add its corresponding container
        :param attributes: The new attributes in case they were changed
        :param data: The preprocessed data binary
        :return:
        """

        if attributes is not None:
            self.update_attributes(attributes)
            self.send_warn(message='Old attributes will be overwritten in the transformed dataset')

        if isinstance(data, pd.DataFrame):
            container = PandasContainer(data, self.attributes)
        elif isinstance(data, np.ndarray):
            container = NumpyContainer(data, self.attributes)
        elif isinstance(data, nx.Graph):
            container = GraphContainer(data, self.attributes)
        else:
            raise ValueError("Unknown data format. Type %s not known." % (type(data)))

        self.add_container(container)

