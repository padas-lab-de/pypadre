import os
import pypadre.graph_import
import pypadre.ds_import

import pandas as pd
import networkx as nx

if __name__ == '__main__':
    path=os.path.expanduser('~/.pypadre')
    auth_token="Bearer 590e1442-7e0f-447d-8d46-d94d539fc631"
    server_url="http://localhost:8080"

    print("download snap_csv graph from snap")
    snap_csv="https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html"
    ds = pypadre.graph_import.create_from_snap(snap_csv, 0)
    print("upload snap_csv graph to server")
    did = pypadre.ds_import.createServerDataset(ds, auth_token,server_url)
    print("download snap_csv graph to server")
    down = pypadre.ds_import.requestServerDataset(did, auth_token,server_url)
    down=None
    ds = None
    did=None

    print("download snap_txt graph from snap")
    snap_txt="https://snap.stanford.edu/data/ca-GrQc.html"
    ds = pypadre.graph_import.create_from_snap(snap_txt, 0)
    print("upload snap_txt graph to server")
    did = pypadre.ds_import.createServerDataset(ds, auth_token,server_url)
    print("download snap_txt graph from server")
    down = pypadre.ds_import.requestServerDataset(did, auth_token,server_url)
    down=None
    ds = None
    did=None

    print("download snap_biodata graph from snap")
    snap_biodata_tsv="https://snap.stanford.edu/biodata/datasets/10006/10006-DD-Miner.html"
    ds = pypadre.graph_import.create_from_snap(snap_biodata_tsv, 0)
    print("upload snap_biodata graph to server")
    did = pypadre.ds_import.createServerDataset(ds, auth_token,server_url)
    print("download snap_biodata graph from server")
    down = pypadre.ds_import.requestServerDataset(did, auth_token,server_url)
    down=None
    ds = None
    did=None

    print("download konect graph from konect")
    konect="http://konect.cc/networks/edit-slwikibooks/"
    ds = pypadre.graph_import.create_from_konect(konect)
    print("upload konect graph to server")
    did = pypadre.ds_import.createServerDataset(ds, auth_token,server_url)
    print("download konect graph to server")
    down = pypadre.ds_import.requestServerDataset(did, auth_token,server_url)
    down=None
    ds = None
    did=None
    """
    print("download openml_dataset dataset big")
    openml_big="https://www.openml.org/d/1457"
    ds = pypadre.ds_import.load_openML_dataset(openml_big, None, apikey="1f8766e1615225a727bdea12ad4c72fa")
    print("upload openml dataset to server")
    did = pypadre.ds_import.createServerDataset(ds, auth_token,server_url)
    print("download openml dataset from server")
    down = pypadre.ds_import.requestServerDataset(did, auth_token,server_url)
    down=None
    ds=None
    did=None
    """
    print("download openml_dataset dataset small")
    openml_small="https://www.openml.org/d/61"
    ds = pypadre.ds_import.load_openML_dataset(openml_small, None, apikey="1f8766e1615225a727bdea12ad4c72fa")
    print("upload openml dataset to server")
    did = pypadre.ds_import.createServerDataset(ds, auth_token,server_url)
    print("download openm dataset from server")
    down = pypadre.ds_import.requestServerDataset(did, auth_token,server_url)

    #test convert from dataset to a graph
    openml_graph="https://www.openml.org/d/31"
    ds = pypadre.ds_import.load_openML_dataset(openml_graph, None, apikey="1f8766e1615225a727bdea12ad4c72fa")
    print("upload openml dataset to server")

    print("convert to Graph 1")
    ds_g1=pypadre.graph_import.create_from_padre_dataset(ds, ds.attributes[0].name, ds.attributes[1].name)
    print("convert to Graph 2")
    ds_g2=pypadre.graph_import.create_from_padre_dataset(ds, ds.attributes[0].name, ds.attributes[1].name,
                                                       node_attr=[],edge_attr=[ds.attributes[3].name,ds.attributes[4].name],directed=False)
    print("upload Graph 1 to server")
    did1 = pypadre.ds_import.createServerDataset(ds_g1, auth_token, server_url)
    print("upload Graph 2 to server")
    did2 = pypadre.ds_import.createServerDataset(ds_g2, auth_token, server_url)

    print("download Graph 1 from server")
    down1 = pypadre.ds_import.requestServerDataset(did1, auth_token,server_url)
    print("download Graph 2 from server")
    down2 = pypadre.ds_import.requestServerDataset(did2, auth_token, server_url)

    ds_from_df=pypadre.ds_import.load_pandas_df(ds.data,target_features=[ds.targets()])
    pypadre.ds_import.createServerDataset(ds_from_df, auth_token, url=server_url)
    #search on server for dataset did from name
    t_id=pypadre.ds_import.get_did_from_meta({"name":ds.name},auth_token=auth_token,max_hits=2,url=server_url)

    print("upload standard sklearn_toys")
    for toy_ds in pypadre.ds_import.load_sklearn_toys():
        pypadre.ds_import.createServerDataset(toy_ds,auth_token,url=server_url)

    #takes some time! initial setup:
    #pypadre.ds_import.send_top_graphs(auth_token,server_url)
    #pypadre.ds_import.sendTop100DatasetsToServer(auth_token,server_url,worker=4)
