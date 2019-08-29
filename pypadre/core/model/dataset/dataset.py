"""
Module containing python classes for managing data sets
- TODO we should integrate statistics of datasets somehow here. see
- TODO allow group based management of binary files similar to hdF5

"""
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from jsonschema import ValidationError
#from padre.PaDREOntology import PaDREOntology
from scipy.stats.stats import DescribeResult

from pypadre.base import MetadataEntity
from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.container.graph_container import GraphContainer
from pypadre.core.model.dataset.container.base_container import IBaseContainer
from pypadre.core.model.dataset.container.numpy_container import NumpyContainer
from pypadre.core.model.dataset.container.pandas_container import PandasContainer
from pypadre.eventhandler import assert_condition
from pypadre.importing.base_validator import IValidator
from pypadre.printing.tablefyable import Tablefyable
from pypadre.printing.util.print_util import StringBuilder, get_default_table
from pypadre.util.dict_util import get_dict_attr
from pypadre.util.utils import _Const


class _Formats(_Const):
    numpy = "numpy"
    numpyMulti = "numpyMultiDim"
    pandas = "pandas"
    graph = "graph"


formats = _Formats()


class DataSetValidator(IValidator):

    @staticmethod
    def validate(obj):
        # TODO validate if metadata are fine. This should prompt the user to input something if the validation fails.
        # TODO Maybe validate with json schema or ontology

        if not obj.metadata.get("name"):
            raise ValidationError("name has to be set for a dataset")

        if not obj.metadata.get("version"):
            raise ValidationError("version has to be set for a dataset")

        if not obj.metadata.get("originalSource"):
            raise ValidationError("originalSource has to be set for a dataset")

        if not obj.metadata.get("type"):
            raise ValidationError("type has to be set for a dataset")


class Dataset(MetadataEntity, Tablefyable):

    def __init__(self, id_=None, attributes: List[Attribute]=None, **metadata):
        """
        :param id_:
        :param attributes: Attributes of the data
        :param metadata:
        """
        MetadataEntity.__init__(self, id_, **{**{"name": "", "version": "1.0", "description": "", "originalSource": "",
                                                 "type": "", "published": False}, **metadata})
        Tablefyable.__init__(self)
        self._binaries = dict()
        self._attributes = attributes

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
    def attributes(self):
        return self._attributes

    def validate(self, options):
        assert_condition(condition=options.get("name") is not None, source=self,
                         message="name attribute has to be set for a dataset")
        assert_condition(condition=options.get("version") is not None, source=self,
                         message="version attribute has to be set for a dataset")
        assert_condition(condition=options.get("originalSource") is not None, source=self,
                         message="originalSource attribute has to be set for a dataset")
        assert_condition(condition=options.get("type") is not None, source=self,
                         message="type attribute has to be set for a dataset")

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
                #return next(iter(self._binaries))
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

    def set_data(self, data):
        """
        Set new data. Container type can be derived automatically.
        :param data: Data to set
        :return:
        """

        if self._attributes is None:
            pass
            # TODO print warning and try to derive attributes. This should be done by the container while the attributes themselves should be held on the dataset.

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

    def get(self, key):
        if key == 'id':
            return self.name

        else:
            return self.__dict__.get(key, None)

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