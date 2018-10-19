"""
Module containing python classes for managing graphs
"""

import networkx as nx
import numpy as np
from .datasets import Dataset
import pylab as plt

import pandas as pd


class GraphDataset:
    #check node attributes g.network.node(data=True)
    #get dataset/attributes of edges nx.to_pandas_edgelist(g.network)


    #def _node_generator(self,padreDataset,source,target,edge):


    #https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.convert_matrix.from_pandas_dataframe.html
    def __init__(self, padre_dataset,source,target, node_attributes,edge_attr=None,directed=False):
        """
        if directed:
            graph=nx.from_pandas_edgelist(padre_dataset.data,source,target,edge_attr,create_using = nx.DiGraph())
        else:
            graph = nx.from_pandas_edgelist(padre_dataset.data, source, target, edge_attr)
        #mgl1




        for col in node_attributes:
            nx.set_node_attributes(graph,pd.Series(padre_dataset.data[col],index=padre_dataset.data[source]).to_dict(),col)

        """


        df=padre_dataset.data

        if directed:
            g=nx.DiGraph()
        else:
            g=nx.Graph()

        # Index of source and target
        src_i = df.columns.get_loc(source)
        tar_i = df.columns.get_loc(target)
        note_attr_i=[df.columns.get_loc(col_name) for col_name in node_attributes]
        if edge_attr:
            # If all additional columns requested, build up a list of tuples
            # [(name, index),...]
            if edge_attr is True:
                # Create a list of all columns indices, ignore nodes
                edge_i = []
                for i, col in enumerate(df.columns):
                    if col is not source and col is not target:
                        edge_i.append((col, i))
            # If a list or tuple of name is requested
            elif isinstance(edge_attr, (list, tuple)):
                edge_i = [(i, df.columns.get_loc(i)) for i in edge_attr]
            # If a string or int is passed
            else:
                edge_i = [(edge_attr, df.columns.get_loc(edge_attr)), ]

            # Iteration on values returns the rows as Numpy arrays
            for row in df.values:
                #Dumps rows that only describe Nodes
                if row[tar_i] != row[tar_i]:
                    g.add_node(row[src_i],**dict(zip(node_attributes, row[note_attr_i])))

                else:
                    s, t = row[src_i], row[tar_i]
                    if g.is_multigraph():
                        g.add_edge(s, t)
                        key = max(g[s][t])  # default keys just count, so max is most recent
                        g[s][t][key].update((i, row[j]) for i, j in edge_i)
                    else:
                        g.add_edge(s, t)
                        g[s][t].update((i, row[j]) for i, j in edge_i)

        # If no column names are given, then just return the edges.
        else:
            for row in df.values:
                g.add_edge(row[src_i], row[tar_i])

        #mgl2:
        """

        node_col=dict()
        for col in node_attributes:
            node_col[col]=ds[col]
        nx.set_node_attributes(graph,pd.Series(node_col,index=ds[node_source_col]),col)

        """
        self.network = g
        padre_dataset._binary._data=None
        self.padre_dataset=padre_dataset

    def to_pandas_df(self):

        edgelist = self.network.edges(data=True)

        source_nodes = [s for s, t, d in edgelist]
        target_nodes = [t for s, t, d in edgelist]
        all_keys = set().union(*(d.keys() for s, t, d in edgelist))
        edge_attr = {k: [d.get(k, float("nan")) for s, t, d in edgelist] for k in all_keys}
        edgelistdict = {"source": source_nodes, "target": target_nodes}
        edgelistdict.update(edge_attr)
        edge_df= pd.DataFrame(edgelistdict)

        nodelist = self.network.nodes(data=True)

        nodes = [node for node, data in nodelist]
        target_nodes = [t for s, t, d in edgelist]
        all_keys = set().union(*(data.keys() for node, data in nodelist))
        node_attr = {key: [data.get(key, None) for node, data in nodelist] for key in all_keys}
        nodelistdict = {"source": nodes}
        nodelistdict.update(node_attr)
        return pd.concat([pd.DataFrame(nodelistdict),edge_df])

        pd.DataFrame.from_dict()

        pd.DataFrame.from_dict(dict(g.network.nodes(data=True)), orient='columns')




        return nx.to_pandas_edgelist(self.network,source="source",target="target")



    def getNodeAttribute(self,node):
        return nx.get_node_attributes(self.network,node)


    def addNode(self,node,attr_dict):
        self.network.add_node(node,attr_dict)

    def addEdge(self,source,target,attr_dict):
        self.network.add_edge(source,target,attr_dict)

