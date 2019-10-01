import numpy as np
import pandas as pd
import pandas_profiling as pd_pf
from padre.PaDREOntology import PaDREOntology

from pypadre.core.model.dataset import dataset
from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.container.base_container import IBaseContainer
from pypadre.core.model.dataset.container.pandas_container import PandasContainer
from pypadre.core.model.generic.i_model_mixins import ILoggable


class GraphContainer(IBaseContainer,ILoggable):

    def __init__(self, data, attributes=None):
        # todo rework binary data into delegate pattern.
        super().__init__(dataset.formats.graph, data, attributes)

        self._shape = (data.number_of_edges(), data.number_of_nodes())
        self._data = data
        self._attributes = self.validate_attributes(attributes)
        self._targets_idx = np.array([idx for idx, a in enumerate(attributes) if a.defaultTargetAttribute])
        self._features_idx = np.array([idx for idx, a in enumerate(attributes) if not a.defaultTargetAttribute])
        #TODO rework how graphs are handled in features and targets

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
                if (att.is_target):
                    removekeys.append(att.name)
            return self._data.drop(removekeys, axis=1)

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

    def convert(self, bin_format):
        if bin_format is dataset.formats.pandas:
            # TODO attributes?
            return PandasContainer(self._pandas_repr())
        return None

    def validate_attributes(self, attributes=None):
        # TODO look for validating the attributes properties with regards to the ontology
        if attributes is None or len(attributes) == 0:
            self.send_warn(message='Attributes are missing! Attempting to derive them from the binary...',
                           condition=True)
            attributes = self.derive_attributes(self.data)

        self.send_error(
            message="Incorrect number of attributes. Data must have at least two columns (source, target), provided attributes %d."
                    % (len(attributes)), condition=len(attributes) < 2)
        return attributes

    @staticmethod
    def derive_attributes(data, targets=None):
        if targets is None:
            targets = []
        graph_attrs = ["source", "target"]
        edges_attrs = [k for k in set().union(*(d.keys() for s, t, d in data.edges(data=True)))]
        nodes_attrs = [k for k in set().union(*(d.keys() for node, d in data.nodes(data=True)))]
        _attributes = []
        index = 0
        for a in graph_attrs:
            _attributes.append(Attribute(name=a, measurementLevel=PaDREOntology.SubClassesMeasurement.Nominal.value,
                                         unit=PaDREOntology.SubClassesUnit.Count.value, index=index,
                                         defaultTargetAttribute=a in targets,
                                         type=PaDREOntology.SubClassesDatum.Character.value,
                                         context={'graph_role': a}))
            index += 1
        for a in edges_attrs:
            _attributes.append(Attribute(name=a, measurementLevel=PaDREOntology.SubClassesMeasurement.Nominal.value,
                                         unit=PaDREOntology.SubClassesUnit.Count.value, index=index,
                                         defaultTargetAttribute=a in targets,
                                         type=PaDREOntology.SubClassesDatum.Character.value,
                                         context={'graph_role': 'edgeattribute'}))
            index += 1
        for a in nodes_attrs:
            _attributes.append(Attribute(name=a, measurementLevel=PaDREOntology.SubClassesMeasurement.Nominal.value,
                                         unit=PaDREOntology.SubClassesUnit.Count.value, index=index,
                                         defaultTargetAttribute=a in targets,
                                         type=PaDREOntology.SubClassesDatum.Character.value,
                                         context={'graph_role': 'nodeattribute'}))
            index += 1

        return _attributes

    def _pandas_repr(self):
        edgelist = self.data.edges(data=True)

        source_nodes = [s for s, t, d in edgelist]
        target_nodes = [t for s, t, d in edgelist]
        all_keys = set().union(*(d.keys() for s, t, d in edgelist))
        edge_attr = {k: [d.get(k, float("nan")) for s, t, d in edgelist] for k in all_keys}
        edgelistdict = {"source": source_nodes, "target": target_nodes}
        edgelistdict.update(edge_attr)
        edge_df = pd.DataFrame(edgelistdict)

        nodelist = self.data.nodes(data=True)

        nodes = [node for node, data in nodelist]
        target_nodes = [t for s, t, d in edgelist]
        all_keys = set().union(*(data.keys() for node, data in nodelist))
        node_attr = {key: [data.get(key, float("nan")) for node, data in nodelist] for key in all_keys}
        nodelistdict = {"source": nodes}
        nodelistdict.update(node_attr)
        # edge_df["target"] = edge_df["target"].astype(str)
        unsorted_df = pd.concat([pd.DataFrame(nodelistdict), edge_df], sort=True, ignore_index=True)
        if unsorted_df["source"].dtype == np.int64:
            unsorted_df["source"] = unsorted_df["source"].astype(unsorted_df["target"].dtype)

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

    def addNode(self, node, attr_dict):
        self.data.add_node(node, **attr_dict)
        self.shape[1] += 1

    def addEdge(self, source, target, attr_dict):
        self.data.add_edge(source, target, **attr_dict)
        self.shape[0] += 1

    #      ret = {"n_att" : len(self._attributes),
    #             "n_target" : len([a for a in self._attributes if a.is_target])}
    #      if self._data is not None:
    #          ret["stats"] = stats.describe(self._data, axis=0)
    #      return ret

    def profile(self, **kwargs):
        return pd_pf.ProfileReport(self.convert(_Formats.pandas).data, **kwargs)

    def describe(self):
        ret = ""
        ret = ret + "Number of Nodes: " + str(self.data.number_of_nodes()) + "\n"
        ret = ret + "Number of Edges: " + str(self.data.number_of_edges()) + "\n"
        ret = ret + "Number of Selfloops: " + str(self.data.number_of_selfloops()) + "\n"

        for att in self.attributes:
            ret = ret + "name: " + str(att.name) + ", graph_role: " + str(att.context["graph_role"]) + "\n"
        return ret
