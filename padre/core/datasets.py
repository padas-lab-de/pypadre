"""
Module containing python classes for managing data sets
- TODO we should integrate statistics of datasets somehow here. see
- TODO allow group based management of binary files similar to hdF5

"""
import altair as alt
import numpy as np
import pandas as pd
from scipy import stats

from padre.base import MetadataEntity
from padre.utils import _const
import pandas_profiling as pd_pf
import networkx as nx


class _Formats(_const):

    numpy = "numpy"
    pandas = "pandas"
    graph = "graph"

formats = _Formats()


class NumpyContainer:

    def __init__(self, data, attributes=None):
        # todo rework binary data into delegate pattern.
        self._shape = data.shape
        if attributes is None:
            self._attributes = [Attribute(i, "RATIO") for i in range(data.shape[1])]
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
            self._targets_idx = np.array([idx for idx, a in enumerate(attributes) if a.defaultTargetAttribute])
            self._features_idx = np.array([idx for idx, a in enumerate(attributes) if not a.defaultTargetAttribute])
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

    def pandas_repr(self):
        return pd.DataFrame(self.data)



    def profile(self,bins=10,check_correlation=True,correlation_threshold=0.9,
                correlation_overrides=None,check_recoded=False):
        return pd_pf.ProfileReport(pd.DataFrame(self.data), bins=bins, check_correlation=check_correlation,
                                   correlation_threshold=correlation_threshold,
                                   correlation_overrides=correlation_overrides,check_recoded=check_recoded)


    def describe(self):
        ret = {"n_att": len(self._attributes),
               "n_target": len([a for a in self._attributes if a.defaultTargetAttribute])}
        if self._data is not None:
            ret["stats"] = stats.describe(self._data, axis=0)
        return ret


class GraphContainer:

    def __init__(self, data, attributes=None):
        # todo rework binary data into delegate pattern.
        self._shape = (data.number_of_edges(), data.number_of_nodes())
        self._data = data
        if attributes is None:
            self._attributes = {}
            self._targets_idx = None
            self.features_idx = None

        else:
            self._attributes = attributes
            self._targets_idx = np.array([idx for idx, a in enumerate(attributes) if a.defaultTargetAttribute])
            self._features_idx = np.array([idx for idx, a in enumerate(attributes) if not a.defaultTargetAttribute])



    @property
    def attributes(self):
        return self._attributes

    @property
    def features(self):
        return self._data
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
        return self._data
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

    def pandas_repr(self):
        edgelist = self.data.edges(data=True)

        source_nodes = [s for s, t, d in edgelist]
        target_nodes = [t for s, t, d in edgelist]
        all_keys = set().union(*(d.keys() for s, t, d in edgelist))
        edge_attr = {k: [d.get(k, float("nan")) for s, t, d in edgelist] for k in all_keys}
        edgelistdict = {"source": source_nodes, "target": target_nodes}
        edgelistdict.update(edge_attr)
        edge_df= pd.DataFrame(edgelistdict)

        nodelist = self.data.nodes(data=True)

        nodes = [node for node, data in nodelist]
        target_nodes = [t for s, t, d in edgelist]
        all_keys = set().union(*(data.keys() for node, data in nodelist))
        node_attr = {key: [data.get(key, float("nan")) for node, data in nodelist] for key in all_keys}
        nodelistdict = {"source": nodes}
        nodelistdict.update(node_attr)
        #edge_df["target"] = edge_df["target"].astype(str)
        unsorted_df = pd.concat([pd.DataFrame(nodelistdict), edge_df], sort=True,ignore_index=True)
        if unsorted_df["source"].dtype==np.int64:
            unsorted_df["source"]=unsorted_df["source"].astype(unsorted_df["target"].dtype)

        if self.attributes is None:
            return unsorted_df
        else:
            col_order = []
            for att in self.attributes:

                if att.context["graph_role"] == "source":
                    col_order.append("source")
                elif att.context["graph_role"] == "target":
                    col_order.append("target")
                else:
                    col_order.append(att.name)
            sorted_df = unsorted_df.reindex(columns=col_order)
            sorted_df.columns = [att.name for att in self.attributes]
            return sorted_df


    @property
    def num_attributes(self):
        return len(self.attributes)

    def getNodes(self):
        return self.data.nodes(data=True)

    def getEdges(self, node):
        return self.data.edges(data=True)

    def addNode(self,node,attr_dict):
        self.data.add_node(node, **attr_dict)
        self.shape[1] = +1

    def addEdge(self, source, target, attr_dict):
        self.data.add_edge(source, target, **attr_dict)
        self.shape[0] = +1
  #      ret = {"n_att" : len(self._attributes),
  #             "n_target" : len([a for a in self._attributes if a.is_target])}
  #      if self._data is not None:
  #          ret["stats"] = stats.describe(self._data, axis=0)
  #      return ret

    def profile(self,bins=10,check_correlation=True,correlation_threshold=0.9,
                correlation_overrides=None,check_recoded=False):
        return pd_pf.ProfileReport(self.pandas_repr(),bins=bins,check_correlation=check_correlation,correlation_threshold=correlation_threshold,
                correlation_overrides=correlation_overrides,check_recoded=check_recoded)


    def describe(self):
        ret = ""
        ret=ret+"Number of Nodes: " + str(self.data.number_of_nodes())+"\n"
        ret = ret + "Number of Edges: " + str(self.data.number_of_edges())+"\n"
        ret = ret + "Number of Selfloops: " + str(self.data.number_of_selfloops())+"\n"


        for att in self.attributes:
            ret=ret+"name: "+ str(att.name)+", graph_role: "+str(att.context["graph_role"])+"\n"
        return ret



class PandasContainer:

    def __init__(self, data, attributes=None):
        # todo rework binary data into delegate pattern.
        self._shape = data.shape
        #pd.DataFrame
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
            self._targets_idx = np.array([idx for idx, a in enumerate(attributes) if a.defaultTargetAttribute])
            self._features_idx = np.array([idx for idx, a in enumerate(attributes) if not a.defaultTargetAttribute])
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
                if(att.defaultTargetAttribute):
                    removekeys.append(att.name)
            return self._data.drop(removekeys,axis=1).values

    @property
    def targets(self):
        if self._targets_idx is None or len(self._targets_idx) == 0:
            return None
        else:
            targets=[]
            for col,att in enumerate(self._attributes):
                if (att.defaultTargetAttribute):
                    targets.append(att.name)

            # if no targets are present, return None
            if len(targets) == 0:
                return None

            return self._data[targets].values

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape


    @property
    def num_attributes(self):
        return self._data.shape[1]

    def pandas_repr(self):
        return self.data


  #      ret = {"n_att" : len(self._attributes),
  #             "n_target" : len([a for a in self._attributes if a.is_target])}
  #      if self._data is not None:
  #          ret["stats"] = stats.describe(self._data, axis=0)
  #      return ret

    def profile(self,bins=10,check_correlation=True,correlation_threshold=0.9,
                correlation_overrides=None,check_recoded=False):
        return pd_pf.ProfileReport(self.data,bins=bins,check_correlation=check_correlation,correlation_threshold=correlation_threshold,
                correlation_overrides=correlation_overrides,check_recoded=check_recoded)


    def describe(self):
        ret = {"n_att" : len(self._attributes),
               "n_target" : len([a for a in self._attributes if a.defaultTargetAttribute])}
        shallow_cp=self._data
        for col in shallow_cp:
            if isinstance(shallow_cp[col][0], str):
                shallow_cp[col]=pd.factorize(shallow_cp[col])[0]
                print(col)
        ret["status"] = stats.describe(shallow_cp.values)
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

    def __init__(self, name, measurementLevel=None, unit=None,
                 description=None, defaultTargetAttribute=False, context=None, index=None, type=None, nullable=True):

        if context is None:
            context={}
        dict.__init__(self, name=name, measurementLevel=measurementLevel,
                      unit=unit, description=description, defaultTargetAttribute=defaultTargetAttribute,
                      context=context, index=index, type=type, nullable=nullable)

    @property
    def name(self):
        if "name" in self:
            return self["name"]
        else:
            self["name"] = None
            return None

    @property
    def index(self):
        if "index" in self:
            return self["index"]
        else:
            self["index"] = None
            return None

    @property
    def measurementLevel(self):
        if "measurementLevel" in self:
            return self["measurementLevel"]
        else:
            self["measurementLevel"] = ""
            return self["measurementLevel"]

    @property
    def unit(self):
        if "unit" in self:
            return self["unit"]
        else:
            self["unit"] = ""
            return self["unit"]

    @property
    def description(self):
        if "description" in self:
            return self["description"]
        else:
            self["description"] = ""
            return self["description"]

    @property
    def defaultTargetAttribute(self):
        if "defaultTargetAttribute" in self:
            return self["defaultTargetAttribute"]
        else:
            self["defaultTaretAttribute"] = False
            return False

    @property
    def context(self):
        if "context" in self:
            return self["context"]
        else:
            self["context"] = dict()
            return self["context"]

    def __str__(self):
        return self.name + "(" + str(self.measurementLevel) + ")"

    def __repr__(self):
        if "graph_role" in self.context:
            return self.name + "(" + self.context["graph_role"] + ")"
        else:
            return str(self["name"]) + "(" + str(self["measurementLevel"]) + "/" + str(self["unit"]) + ")"


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
        self._binary_loader_fn = None
        self._binary_format = None
        self._fill_metedata()

    def _get_binary(self):
        """
        returns the binary. In case that a binary_loader_fn is given, the binary is lazy loaded.
        :return:
        """
        if self._binary is not None:
            return self._binary.data
        elif self._binary_loader_fn is not None:
            self.set_data(*self._binary_loader_fn())  # derefered or lazy loading
        return self._binary

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
        """
        should not be used to access data. @look ap .pandas_repr
        :return: the stored _binary. Can be numpy-array, pandas-DataFrame or networkx.Graph object
        """
        if self.has_data():
            return self._binary.data
        else:
            return None

    def has_data(self):
        return self._get_binary() is not None

    @property
    def size(self):
        """
        :return: (n_examples, n_attributes)
        """
        if not self.has_data():
            return None
        else:
            return self.data.shape

    @property
    def isgraph(self):
        if "type" in self.metadata:
            return self.metadata["type"] == "graph" or self.metadata["type"] == "graphDirected"
        else:
            return False

    def _fill_metedata(self):
        keys = ["name", "version", "description", "originalSource", "type"]
        for key in keys:
            if key not in self.metadata:
                self.metadata[key] = ""
        if "published" not in self.metadata:
            self.metadata["published"] = False

    def pandas_repr(self):
        """
        :return: The pandas representation of the dataset. converts Numpy-array, pandas-df, and nx.Graph objects to pandas DF
        """
        if self.has_data():
            return None
        else:
            return self.data.pandas_repr()

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

    def get_target_attribute(self):
        """
        Return default target attribute
        :return:
        """
        for attr in self.attributes:
            if attr["defaultTargetAttribute"]:
                return attr["name"]
        return None

    @property
    def num_attributes(self):
        if self.has_data():
            return self._binary.num_attributes
        else:
            return 0

    def profile(self, bins=50, check_correlation=True, correlation_threshold=0.8,
                correlation_overrides=None, check_recoded=False):
        if "profile" in self.metadata:
            return self.metadata["profile"]
        elif self.data is not None:
            profile = self.data.profile(bins, check_correlation, correlation_threshold,
                                       correlation_overrides, check_recoded).get_description()
            profile["variables"] = profile["variables"].to_dict(orient="index")
            self.metadata["profile"] = profile

            _check_profiling_datatype(profile["variables"])
            _check_profiling_datatype(profile["table"])

            for key in profile["freq"].keys():
                profile["freq"][key]=profile["freq"][key].to_dict()

            for key in profile["correlations"].keys():
                profile["correlations"][key]=profile["correlations"][key].to_dict()

            return self.metadata["profile"]
        else:
            return "No records available"

    def describe(self):
        if self.data is not None:
            return self._binary.describe()
        else:
            return "No records available"

    def set_data(self, data, attributes=None):
        """
        sets the binary data and descriptive attributes. If data is a function, it is expected that the function
        returns a tuple (data, attributes) which can be used to call set_data at a later point in time and thus support
        lazy loading.
        :param data: binary data in a supported format (numpy, pandas, networkx):
                    size must be num_datasets x num_attributes. If data is a function,
                    it is expected that the function is called later.
        :param attributes: Description for attributes or None.
                           If none, the attributes will be estimated as good as possible

        """
        self._binary = None
        self._binary_format = None
        if data is None:
            self._binary_format = None
            self._binary = AttributeOnlyContainer(attributes)
        elif hasattr(data, '__call__'):
            self._binary_loader_fn = data
            return
        elif isinstance(data, pd.DataFrame):
            # Remove non numerical attributes
            # TODO: Bring in support for nominal and ordinal attributes too
            self._binary = PandasContainer(data, attributes)
            self._binary_format = formats.pandas
        elif isinstance(data, np.ndarray):
            self._binary = NumpyContainer(data, attributes)
            self._binary_format = formats.numpy
        elif isinstance(data, nx.Graph):
            self._binary = GraphContainer(data, attributes)
            self._binary_format = formats.graph
        else:
            raise ValueError("Unknown data format. Type %s not known." % (type(data)))
        self._fill_metedata()

    def replace_data(self, data):
        """
        Function is used to hold a temporary dataset of preprocessed data
        :param data: binary data in a supported format (numpy, pandas, networkx):
                    size must be num_datasets x num_attributes. If data is a function,
                    it is expected that the function is called later.
        :return: None
        """

        if data is None:
            self._binary_format = None
            self._binary = AttributeOnlyContainer(self.attributes)
        elif hasattr(data, '__call__'):
            self._binary_loader_fn = data
            return
        elif isinstance(data, pd.DataFrame):
            # Remove non numerical attributes
            # TODO: Bring in support for nominal and ordinal attributes too
            self._binary = PandasContainer(data, self.attributes)
            self._binary_format = formats.pandas
        elif isinstance(data, np.ndarray):
            # Append the target data to the original data
            data = np.append(data, self.targets(), axis=1)

            # Create a new numpy container with the old attributes
            self._binary = NumpyContainer(data, self.attributes)
            self._binary_format = formats.numpy
        elif isinstance(data, nx.Graph):
            self._binary = GraphContainer(data, self.attributes)
            self._binary_format = formats.graph
        else:
            raise ValueError("Unknown data format. Type %s not known." % (type(data)))

    def get_scatter_plot(self, x_attr, y_attr, x_title=None, y_title=None):
        """
        Get scatter plot of vega lite specification

        :param x_attr: Data set attribute
        :param y_attr: Data set attribute
        :param x_title: Title for x axis
        :param y_title: Title for y axis
        :return: Vega lite json specification
        """
        data = self.data
        target = self.get_target_attribute()
        if x_title is None:
            x_title = x_attr[0].upper() + x_attr[1:]
        if y_title is None:
            y_title = y_attr[0].upper() + y_attr[1:]
        chart = alt.Chart(data).mark_point().encode(
            x=alt.X(x_attr, title=x_title),
            y=alt.Y(y_attr, title=y_title),
            color=target + ":N"
        ).properties(title=self.name).interactive()
        return chart.to_json()

    def get_chart_from_json(self, visualisation):
        """
        Get altair Chart from json to render in notebook
        :param visualisation: Json specification of vega lite
        :return: Altair Chart
        :rtype: <class 'altair.vegalite.v2.api.Chart'>
        """
        return alt.Chart.from_json(visualisation)

    def __str__(self):
        return str(self.id) +"_"+ str(self.name) + ": " + str(self.type) + ", " + str(self.size) + ", " + str(self.binary_format())


def _check_profiling_datatype(content):
    if(isinstance(content,dict)):
        for key in content.keys():
            if key == "histogram" or key == "mini_histogram" or content[key] is np.nan:
                content[key]=None
            elif isinstance(content[key], np.int32)or isinstance(content[key], np.int64):
                content[key] = int(content[key])
            elif isinstance(content[key], np.bool_):
                content[key] = bool(content[key])
            elif isinstance(content[key],list) or isinstance(content[key],dict):
                _check_profiling_datatype(content[key])

