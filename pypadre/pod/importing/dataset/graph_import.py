"""
Module containing python classes for managing graphs
"""
import copy
import gzip
import os
import ssl
import tarfile
import tempfile
from urllib.request import urlopen
from urllib.request import urlopen as uReq

import networkx as nx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as soup

from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.core.ontology.padre_ontology import PaDREOntology

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE

def pandas_to_networkx(df, source, target, create_using, node_attr=[], edge_attr=[]):
    """Takes a pandas dataframe and convertes it to a networkx object. The column-name of the source and target column
    will be the source and target Nodes. In addition the columns of the of the node attributes and the columns of the
    edge attributes and be specified. Columns that neither are in node_attr nor in edge_attr are dropped.

    Args:
        df (pandas.Dataframe): The dataframe that is converted into a networkx object.
        source (str): Column name of source Nodes. Must be the same as in the df (pandas dataframe).
        target (str): Column name of target Nodes. Must be the same as in the df (pandas dataframe).
        create_using (networkx.Graph): The networkx.Graph object, that should be used as a networkx object.
                                        Depending on the input, it can also be a directed Graph object.
        node_attr (list): A list of all columns that contain the names of all attributes of nodes.
        edge_attr (list): A list of all columns that contain the names of all attributes of edges.
    """


    if df[source].dtype == np.int64:
        df[source] = df[source].astype(np.float)
        df[target] = df[target].astype(np.float)
    if node_attr is not None:
        node_attr=list(node_attr)
    else:
        node_attr=[]
    if edge_attr is not None:
        edge_attr=list(edge_attr)
    else:
        edge_attr=[]

    src_i = df.columns.get_loc(source)#column of source
    tar_i = df.columns.get_loc(target)#column of target
    note_attr_i = [df.columns.get_loc(col_name) for col_name in node_attr]#list of all edge-attr col-names

    edge_i = [(i, df.columns.get_loc(i)) for i in edge_attr]
    # If a string or int is passed
    # Iteration on values returns the rows as Numpy arrays
    for row in df.values:
       #check if the target is NaN
        if row[tar_i] !=row[tar_i]:
            create_using.add_node(row[src_i], **dict(zip(node_attr, row[note_attr_i])))

        else:
            #g.add_node(row[src_i], **dict(zip(node_attr, row[note_attr_i])))
            s, t = row[src_i], row[tar_i]
            if create_using.is_multigraph():
                create_using.add_edge(s, t)
                key = max(create_using[s][t])  # default keys just count, so max is most recent
                create_using[s][t][key].update((i, row[j]) for i, j in edge_i)
            else:

                create_using.add_edge(s, t)
                create_using[s][t].update( (i, row[j]) for i, j in edge_i)


def create_from_padre_dataset( padre_dataset,source,target, node_attr=[],edge_attr=[],directed=False):
    """Takes a padre dataset and convertes it to a padre dataset that is suited for networkx objects.
    The column-name of the source and target column
    will be the source and target Nodes. In addition the columns of the of the node attributes and the columns of the
    edge attributes and be specified. Columns that neither are in node_attr nor in edge_attr are dropped.
    To distinct edges form nodes, node_attributes are only assigned if the target node is none.

    Args:
        padre_dataset (pypadre.dataset): The dataset that is converted.
        source (str): Column name of source Node. Must be the same as in the padre_dataset.
        target (str): Column name of target Node. Must be the same as in the padre_dataset.
        node_attr (list): A list of all columns that contain the names of all attributes of nodes.
        edge_attr (list): A list of all columns that contain the names of all attributes of edges.
        directed (bool): Whether or not the created graph is a directed graph.
    Returns:
        pypadre.Dataset() A padre_dataset containing the data of the padre_dataset, but converted to support graphs.
    """
    padre_dataset=copy.deepcopy(padre_dataset)
    df = padre_dataset.pandas_repr()
    atts = []

    if directed:
        g = nx.DiGraph()
        padre_dataset.metadata["type"]="graphDirected"
    else:
        g = nx.Graph()
        padre_dataset.metadata["type"] = "graph"

    # Index of source and target
    pandas_to_networkx(df, source, target, g, node_attr, edge_attr)

    for attribute in padre_dataset._binary._attributes:

        if attribute["name"] == source:
            attribute.context["graph_role"]="source"
            atts.append(attribute)

        elif attribute.name == target:
            attribute.context["graph_role"] = "target"
            atts.append(attribute)
        elif attribute.name in edge_attr:
            attribute.context["graph_role"] = "edgeattribute"
            atts.append(attribute)
        elif attribute.name in node_attr:
            attribute.context["graph_role"] = "noteattribute"
            atts.append(attribute)


    ds=Dataset(None,**padre_dataset.metadata)
    ds.set_data(g, atts)

    return ds


def _get_row_in_parent(name,tables):
    """Is used for web-scrapping from the snap website to gather meta-data about the graph.

    Args:
        name (str): The name of the Graph of interest.
        tables: The table that should be checked for information.
    Returns:
        The row object, that contains meta data of the dataset.
    """
    for table in tables:
        rows = table.find_all("tr")
        if rows[0].find_all("th", {"colspan": "2"}):
            break
        for row in rows[1:]:
            row_content = row.find_all("td")
            if row_content[0].find("a").text == name:
                return row_content


def _gather_parent_data(name,url=None,isBiosnap=False):
    """Web-scrapes the Snap website to gather additional meta-data about the graph.

    Args:
        name (str): The name of the graph.
        isBiosnap (bool): Snap conatins Biosnap datasets, that have to be treated differently
    Returns:
        a Quadruple conatining the extracted metadata.
    """
    if(isBiosnap):
        parent_url="https://snap.stanford.edu/biodata/index.html"
    else:
        parent_url = "https://snap.stanford.edu/data/index.html"

    uClient = uReq(parent_url,context=_ctx)
    page_html = uClient.read()
    uClient.close()
    page_soup = soup(page_html, "html.parser")
    tables = page_soup.find_all("div", {"id": "right-column"})[0].find_all("table", {"id": "datatab2"})
    columns=[]
    row_content=_get_row_in_parent(name,tables)

    type = row_content[1].text
    short_description = row_content[-1].text
    snap_id = None
    if(isBiosnap):
        columns=row_content[2].text.split(", ")
        snap_id = url.split("/")[5]

    is_directed = not "Undirected" in type

    return {"graph_type":type,"short_description":short_description,"is_directed":is_directed,"columns":columns,
            "snap_id":snap_id}

def _pair_different(columns):
    #check pairwise distinct
    for i,coli in enumerate(columns):
        for j,colj in enumerate(columns):
            if i!=j and coli==colj:
                return False
    return True

def _get_html_page(url):
    """

    :param url:
    :return:
    """

    response = urlopen(url, context= _ctx)
    page_html = response.read()
    response.close()
    page_soup = soup(page_html,"html.parser")

    return page_soup


def _get_link(url,page_soup=None,link_num=0,logger=None):

    all_href = page_soup.find_all("a")
    if all_href is None:
        pass
    else:
        is_biodata = url.split("/")[3] == "biodata"

        if is_biodata:
            links = [i for i in all_href if ".tsv.gz" in str(i) or ".csv.gz" in str(i)]
            if len(links) < link_num:
                #
                if logger:
                    logger.send_error(message="Invalid Link or invalid Link Number")
                else:
                    print("Invalid Link or invalid Link Number")
                    return None

            snap_id = url.split("/")[5]
            name = url.split("/")[6].replace(snap_id + "-", "").replace(".html", "")
            link = "https://snap.stanford.edu/biodata/datasets/" + snap_id + "/" + links[link_num]["href"]
        else:
            links = [i for i in all_href if ".txt.gz" in str(i) or ".csv.gz" in str(i)]
            if len(links) < link_num:
                if logger:
                    logger.send_error(message="Invalid Link or invalid Link Number")
                else:
                    print("Invalid Link or invalid Link Number")
                    return None

            link = links[link_num]["href"]
            link = link.replace("../data", "")
            link = link.replace("https://snap.stanford.edu/data/", "")
            link = link.replace("https://snap.stanford.edu/", "")
            link = "https://snap.stanford.edu/data/" + link
            name = url.split("/")[4].replace(".html", "")

        return name, link, is_biodata

def create_from_snap(url, link_num=0, logger=None):
    """Takes the graph of the Snap website and puts it into a pypadre.dataset.

    Args:
        url (str): The url of the specific graph. From graph of this website: https://snap.stanford.edu/
        link_num (int): Some Graphs have several datasets and thus several download-links.
    Returns:
        A pypadre.dataset object that conatins the graph of the url.
    """
    meta_dict={"description": "","version":0, "published": False, "originalSource": url}
    atts=[]

    page_soup = _get_html_page(url)
    name, link, is_biodata = _get_link(url,page_soup=page_soup,link_num=link_num,logger=logger)

    graph_meta = _gather_parent_data(name,url,is_biodata)

    description = "".join(str(page_soup.find("div", {"id": "right-column"})).split("<table")[0].split("<p>")[1:]) \
        .replace("</p>", "").replace("\n", "").replace("\r", "").replace("<br/>", "").replace("<a", "").replace("</a>","")

    with tempfile.TemporaryFile(mode='w+b') as ftemp:
        response = urlopen(link,context=_ctx)
        buflen = 1 << 20
        while True:
            buf = response.read(buflen)
            ftemp.write(buf)
            if len(buf) < buflen:
                break
        ftemp.seek(0)
        comments=[]
        columns=[]
        if not is_biodata:
            if ".txt" in link:
                with gzip.open(ftemp) as gzipread:
                    while True:

                        line = gzipread.readline().decode("utf-8")
                        if (line[0] is "#"):
                            comments.append(line)
                        else:
                            break

                ftemp.seek(0)
                df_edges = pd.read_csv(ftemp, delim_whitespace=True, compression='gzip', comment="#", header=None)
                if len(comments) == 4:
                    meta_dict["description"] += " short-description:" + comments[
                        0] + " | " + comments[1]

                    columns = comments[-1][2:].replace("\n", "").replace("\r", "").split("\t")
                    meta_dict["description"] += " | column names: " + str(columns)
                else:
                    meta_dict["description"] += " | file_comments: " + str(comments) + " | short-description: " + \
                                           str(graph_meta["short_description"])
            else:
                df_edges = pd.read_csv(ftemp, delimiter=",", compression='gzip', header=None)

                meta_dict["description"] += " | short_description: " + str(
                    graph_meta["short_description"])

                columns = page_soup.find_all("div", {"class": "code"})[0].text.split(", ")

                meta_dict["description"] += " | column names: " + str(columns)

                for i in page_soup.find_all("ul")[-1].find_all("li"):
                    description = description + " | " + i.text
        else:
            col_names=None
            header = None
            with gzip.open(ftemp) as gzipread:
                line = gzipread.readline().decode("utf-8")
                if (line[0] is "#"):
                    header = 'infer'
                else:
                    col_names = graph_meta["columns"]
            ftemp.seek(0)

            delimiter = ","
            if ".tsv" in link:
                delimiter = "\t"

            df_edges = pd.read_csv(ftemp, delimiter=delimiter, compression='gzip', header=header, names=col_names)
            edge_col = [str(col) for col in df_edges.columns.values]

            edge_col[0] = edge_col[0][2:]
            columns = edge_col
            meta_dict["description"] += " | short_description: " + str(graph_meta["short_description"])
            meta_dict["description"] += " | columns: " + str(edge_col)
            meta_dict["description"] += " | alternative_column names: " + str(graph_meta["columns"])
            meta_dict["description"] += " | snap_ip: " + str(graph_meta["snap_id"])


        if graph_meta["is_directed"]:
            graph = nx.DiGraph()
            meta_dict["type"] = PaDREOntology.SubClassesDataset.Graph.value + "Directed"
        else:
            graph = nx.Graph()
            meta_dict["type"] = PaDREOntology.SubClassesDataset.Graph.value

        meta_dict["name"]=name
        meta_dict["description"] += " | description: " + str(description) + " | graph_type: " + graph_meta["graph_type"]

        if len(columns)==0:
            columns=["source","target"]
            for i in range(df_edges.shape[1]-2):
                columns.append("edgeattribute"+str(i))

        columns=[str(col) for col in columns]
        if _pair_different(columns):
            df_edges.columns=columns
        else:
            columns=df_edges.columns.values

        for col in enumerate(columns):
            if col[0] == 0:
                atts.append(Attribute(name=col[1], measurementLevel=PaDREOntology.SubClassesMeasurement.Nominal.value,
                                      type=PaDREOntology.SubClassesDatum.Character.value,
                                      context={"graph_role": "source"}))
            elif col[0] == 1:
                atts.append(Attribute(name=col[1], measurementLevel=PaDREOntology.SubClassesMeasurement.Nominal.value,
                                      type=PaDREOntology.SubClassesDatum.Character.value,
                                      context={"graph_role": "target"}))
            else:
                atts.append(Attribute(name=col[1], measurementLevel=PaDREOntology.SubClassesMeasurement.Nominal.value,
                                      type=PaDREOntology.SubClassesDatum.Character.value,
                                      context={"graph_role": "edgeattribute"}))

        pandas_to_networkx(df_edges, columns[0],columns[1], graph, [],columns[2:])

        meta_dict["attributes"] =atts

        return graph,meta_dict


def create_from_konect(url, zero_based=False):
    """Takes the graph of the Konect website and puts it into a pypadre.dataset.

    Args:
        url (str): The url of the specific graph.
        zero_based (bool): Some Graphs are biparite. When dealing with such a dataset, nodes of the source and target
        column are never the same. When dealing with normal Graphs keep value to False
    Returns:
        A pypadre.dataset object that contains the graph of the url.
    """
    name = url.split("/")[4]
    meta_dict = {"name":name,"description":"","version":0,"originalSource":url,"published":False}
    atts = []


    with tempfile.TemporaryFile(mode='w+b') as ftemp:
        response = urlopen('http://konect.cc/files/download.tsv.%s.tar.bz2' % name, context=_ctx)
        buflen = 1 << 20
        while True:
            buf = response.read(buflen)
            ftemp.write(buf)
            if len(buf) < buflen:
                break
        ftemp.seek(0)
        with tempfile.TemporaryDirectory(suffix=name) as tempdir:
            with tarfile.open(fileobj=ftemp, mode='r:bz2') as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, path=tempdir)

                for root, dirs, files in os.walk(tempdir):
                    for file in files:
                        if file.startswith("README"):
                            readme_file = open(os.path.join(root, file))
                            meta_dict["description"] += " | README: " + readme_file.read()
                            readme_file.close()

                        if file.startswith("meta."):
                            meta_file = open(os.path.join(root, file),encoding="utf-8")
                            meta = meta_file.read().split("\n")
                            meta_file.close()
                            meta_dict["description"]+=" | metafile-content: "+ str({entry.split(":")[0]: entry.split(":")[1][1:] for entry in meta[:-1] if entry is not ""})

                        if file.startswith("out."):
                            edges = np.loadtxt(os.path.join(root, file), comments="%")
                            line_ = open(os.path.join(root, file))
                            line = line_.read()
                            line_.close()

                            if "asym" not in line:
                                gr = nx.Graph()
                                meta_dict["type"] = PaDREOntology.SubClassesDataset.Graph.value
                            else:
                                gr = nx.DiGraph()
                                meta_dict["type"] = PaDREOntology.SubClassesDataset.Graph.value + "Directed"

                            if(zero_based):
                                edges[:, :2] -= 1  # we need zero-based indexing

                            if "bip" in line:  # bipartite graphs have non-unique indexing
                                edges[:, 1] += edges[:, 0].max() + 1

                            if edges.shape[1] == 2:
                                pandas_to_networkx(pd.DataFrame(data=edges, columns=["source", "target"]), "source",
                                                   "target", gr)
                            elif edges.shape[1] == 3:
                                pandas_to_networkx(pd.DataFrame(data=edges, columns=["source", "target","weight"]), "source",
                                                   "target", gr,[],["weight"])

                            elif edges.shape[1] == 4:
                                pandas_to_networkx(pd.DataFrame(data=edges, columns=["source", "target", "weight","time"]),
                                                   "source",
                                                   "target", gr, [], ["weight","time"])


    meta_dict["description"] += " | Columns: source, target"
    for col in range(edges.shape[1]):
        if col == 0:
            atts.append(Attribute(name="source", measurementLevel=PaDREOntology.SubClassesMeasurement.Nominal.value,
                                  type=PaDREOntology.SubClassesDatum.Character.value,
                                  context={"graph_role": "source"}))
        elif col == 1:
            atts.append(Attribute(name="target", measurementLevel=PaDREOntology.SubClassesMeasurement.Nominal.value,
                                  type=PaDREOntology.SubClassesDatum.Character.value,
                                  context={"graph_role": "target"}))
        elif col == 2:
            atts.append(Attribute(name="weight", measurementLevel=PaDREOntology.SubClassesMeasurement.Nominal.value,
                                  type=PaDREOntology.SubClassesDatum.Character.value,
                                  context={"graph_role": "edgeattribute"}))
            meta_dict["description"] += ", weight"
        else:
            atts.append(Attribute(name="time", measurementLevel=PaDREOntology.SubClassesMeasurement.Nominal.value,
                                  type=PaDREOntology.SubClassesDatum.Character.value,
                                  context={"graph_role": "edgeattribute"}))
            meta_dict["description"] += ", time"

    meta_dict["attributes"] = atts

    return gr, meta_dict