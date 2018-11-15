"""
Module containing python classes for managing graphs
"""
import os
import tarfile
import tempfile
import gzip
import networkx as nx
import numpy as np
from urllib.request import urlopen
import pandas as pd
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup


def pandas_to_networkx(df, source, target, create_using, node_attributes=[], edge_attr=None):
    g = create_using
    src_i = df.columns.get_loc(source)
    tar_i = df.columns.get_loc(target)
    note_attr_i = [df.columns.get_loc(col_name) for col_name in node_attributes]
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
            # Dumps rows that only describe Nodes
            if row[tar_i] != row[tar_i]:
                g.add_node(row[src_i], **dict(zip(node_attributes, row[note_attr_i])))

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




class GraphContainer:




    #check node attributes g.network.node(data=True)
    #get dataset/attributes of edges nx.to_pandas_edgelist(g.network)


    #def _node_generator(self,padreDataset,source,target,edge):



    def __init__(self):
        self.network = None
        self.padre_dataset = None
        self.meta_dict={}


    #https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.convert_matrix.from_pandas_dataframe.html
    def fill_from_padre_dataset(self, padre_dataset,source,target, node_attributes=[],edge_attr=None,directed=False):

        df=padre_dataset.data

        if directed:
            g=nx.DiGraph()
        else:
            g=nx.Graph()

        # Index of source and target
        pandas_to_networkx(df, source, target, g, node_attributes, edge_attr)

        self.network=g


        padre_dataset._binary._data=None
        self.padre_dataset=padre_dataset


    def _get_row_in_parent(self,name,tables):
        for table in tables:
            rows = table.find_all("tr")
            if rows[0].find_all("th", {"colspan": "2"}):
                break
            for row in rows[1:]:
                row_content = row.find_all("td")
                if row_content[0].find("a").text == name:
                    return row_content


    def _gather_parent_data(self,name,isBiosnap):
        if(isBiosnap):
            parent_url="https://snap.stanford.edu/biodata/index.html"
        else:
            parent_url = "https://snap.stanford.edu/data/index.html"

        uClient = uReq(parent_url)
        page_html = uClient.read()
        uClient.close()
        page_soup = soup(page_html, "html.parser")
        tables = page_soup.find_all("div", {"id": "right-column"})[0].find_all("table", {"id": "datatab2"})
        columns=[]
        row_content=self._get_row_in_parent(name,tables)

        type = row_content[1].text
        short_description = row_content[-1].text
        if(isBiosnap):
            columns=row_content[2].text.split(", ")

        is_directed = not "Undirected" in type

        return (type,short_description,is_directed,columns)


    #https://snap.stanford.edu/data/ca-HepPh.html
    def fill_from_snap(self,url,link_num=0):

        is_biodata = url.split("/")[3] == "biodata"



        name=""
        description=""
        type=""
        is_directed=True
        short_description = ""
        edges=None

        #gather data from url
        uClient = uReq(url)
        page_html = uClient.read()
        uClient.close()
        page_soup = soup(page_html, "html.parser")
        all_a = page_soup.find_all('a')


        if is_biodata:
            "https://snap.stanford.edu/biodata/datasets/10017/files/ChChSe-Decagon_polypharmacy.csv.gz"
            links = [str(i).split("\"")[1] for i in all_a if ".tsv.gz" in str(i) or ".csv.gz" in str(i)]
            if len(links) < link_num:
                print("Invalid Link or invalid Link Number")
                return None
                #"<a href="files/SS-Butterfly_labels.tsv.gz">SS-Butterfly_labels.tsv.gz</a>"
            link="https://snap.stanford.edu/biodata/datasets/"+url.split("/")[6]+links[link_num]
            snap_id=url.split("/")[5]
            name=url.split("/")[6].replace(snap_id+"-","").replace(".html", "")
            link="https://snap.stanford.edu/biodata/datasets/"+snap_id+"/"+links[link_num]
        else:
            links = [str(i).split("\"")[1] for i in all_a if ".tsv.gz" in str(i) or ".csv.gz" in str(i)]
            if len(links) < link_num:
                print("Invalid Link or invalid Link Number")
                return None
            link = "https://snap.stanford.edu/data/" + links[link_num].split("/")[-1]
            name = url.split("/")[4].replace(".html", "")
        description = "".join(
            str(page_soup.find("div", {"id": "right-column"})).split("<table")[0].split("<p>")[1:]).replace("</p>", "") \
            .replace("\n", "").replace("\r", "").replace("<br/>", "")




        type, short_description, is_directed, columns =self._gather_parent_data(name,is_biodata)


        with tempfile.TemporaryFile(mode='w+b') as ftemp:

            response = urlopen(link)
            buflen = 1 << 20
            while True:
                buf = response.read(buflen)
                ftemp.write(buf)
                if len(buf) < buflen:
                    break
            ftemp.seek(0)
            comments=[]
            if not is_biodata:


                if ".txt" in link:
                    with gzip.open(ftemp) as gzipread:
                        while(True):

                            line=gzipread.readline().decode("utf-8")
                            if (line[0] is "#"):
                                comments.append(line)
                            else:
                                break
                #df = pd.read_csv(ftemp,delim_whitespace=True, compression='gzip',nrows=3)

                    ftemp.seek(0)
                    edges = pd.read_csv(ftemp, delim_whitespace=True, compression='gzip', comment="#", header=None)

                    if len(comments) is 4:
                        self.meta_dict["short_description"] = comments[0] + " | " + comments[1]
                        self.meta_dict["columns"] = comments[-1][2:].replace("\n", "").replace("\r", "").split("\t")
                    else:
                        self.meta_dict["file_comments"] = comments
                else:
                    edges = pd.read_csv(ftemp, delimiter=",", compression='gzip', header=None)

                    self.meta_dict["short_description"] = short_description

                    columns=page_soup.find_all("div", {"class": "code"})[0].text.split(", ")
                    if edges.shape[1] == len(columns):
                        self.meta_dict["columns"] = columns
                    else:
                        self.meta_dict["alternative_col_name"] = columns
                    description=description+"\n"
                    for i in page_soup.find_all("ul")[-1].find_all("li"):
                        description =description+" | " + i.text

            else:
                if ".tsv" in link:
                    edges = pd.read_csv(ftemp, delimiter="\t",compression='gzip')
                else:
                    edges = pd.read_csv(ftemp, delimiter=",", compression='gzip')
                edge_col=edges.columns.values
                edges.columns=list(range(len(edge_col)))

                edge_col[0]=edge_col[0][2:]
                self.meta_dict["short_description"]=short_description
                self.meta_dict["columns"]=edge_col
                self.meta_dict["snap_id"]=snap_id
                self.meta_dict["alternative_col_name"]=columns
            #print(df)
                    #tar.extractall(path=tempdir)
                    #contains= tar.read()
#                    k=pd.read_table(tar)
                    #edges = pd.read_csv(tar, delim_whitespace=True, compression='gzip', comment="#", header=None)
                    #edges = np.loadtxt(tar, comments="#")
                    #print(edges)
            if is_directed:
                gr = nx.DiGraph()
            else:
                gr = nx.Graph()

            #for i in range(len(comments)):

            self.meta_dict["name"]=name
            self.meta_dict["description"]=description
            self.meta_dict["type"]=type

            pandas_to_networkx(edges, 0,1, gr, [], list(range(2,len(edges.columns.values))))
            self.network=gr



    def fill_from_konect(self,name,zero_based=False):
        #name from http://konect.cc/networks/
        with tempfile.TemporaryFile(mode='w+b') as ftemp:

            response = urlopen('http://konect.cc/files/download.tsv.%s.tar.bz2' % name)
            buflen = 1 << 20
            while True:
                buf = response.read(buflen)
                ftemp.write(buf)
                if len(buf) < buflen:
                    break
            ftemp.seek(0)
            with tempfile.TemporaryDirectory(suffix=name) as tempdir:
                with tarfile.open(fileobj=ftemp, mode='r:bz2') as tar:
                    tar.extractall(path=tempdir)



                    for root, dirs, files in os.walk(tempdir):
                        for file in files:
                            if file.startswith("README"):
                                self.meta_dict["README"] = open(os.path.join(root, file)).read()
                            if file.startswith("meta."):
                                meta = open(os.path.join(root, file),encoding="utf-8").read().split("\n")
                                self.meta_dict.update({entry.split(":")[0]: entry.split(":")[1][1:] for entry in meta[:-1] if entry is not ""})
                            if file.startswith("out."):
                                edges = np.loadtxt(os.path.join(root, file), comments="%")
                                print(os.path.join(root, file))
                                line = open(os.path.join(root, file)).read()
                                if "asym" not in line:
                                    gr = nx.Graph()
                                else:
                                    gr = nx.DiGraph()
                                if(zero_based):
                                    edges[:, :2] -= 1  # we need zero-based indexing
                                if "bip" in line:  # bipartite graphs have non-unique indexing
                                    edges[:, 1] += edges[:, 0].max() + 1
                                #g.add_edge_list(edges[:, :2])

                                if edges.shape[1] == 2:
                                    pandas_to_networkx(pd.DataFrame(data=edges, columns=["source", "target"]), "source",
                                                       "target", gr)
                                elif edges.shape[1] == 3:
                                    print(edges)
                                    pandas_to_networkx(pd.DataFrame(data=edges, columns=["source", "target","weight"]), "source",
                                                       "target", gr,[],["weight"])
                                    #g.ep.weight = g.new_edge_property("double")
                                    #g.ep.weight.a = edges[:, 2]
                                elif edges.shape[1] == 4:
                                    print(edges)
                                    pandas_to_networkx(pd.DataFrame(data=edges, columns=["source", "target", "weight","time"]),
                                                       "source",
                                                       "target", gr, [], ["weight","time"])
                                    #g.ep.time = g.new_edge_property("int")
                                    #g.ep.time.a = edges[:, 3]
        self.network=gr


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
        node_attr = {key: [data.get(key, float("nan")) for node, data in nodelist] for key in all_keys}
        nodelistdict = {"source": nodes}
        nodelistdict.update(node_attr)
        return pd.concat([pd.DataFrame(nodelistdict),edge_df],sort=True)


    def getNodes(self):
        return self.network.nodes(data=True)

    def getEdges(self,node):
        return self.network.edges(data=True)

    def addNode(self,node,attr_dict):
        self.network.add_node(node,**attr_dict)

    def addEdge(self,source,target,attr_dict):
        self.network.add_edge(source,target,**attr_dict)




