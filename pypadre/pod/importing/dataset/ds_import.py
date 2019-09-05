"""
Modul containing import methods from different packages / repositories.
"""

import copy
import json
import os.path
import tempfile
from multiprocessing import Process

import arff
import networkx as nx
import numpy as np
import openml as oml
import pandas as pd
import pypadre.pod.backend.http.protobuffer.proto_organizer as proto
import requests
from requests.exceptions import ConnectionError

import pypadre
from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.dataset import Dataset

# def _split_DESCR(s):
#     s = s.strip()
#     k = s.find("\n")
#     return s[0:k], s[k + 1:]
#
#
# def _create_dataset_data(bunch):
#     n_feat = bunch.data.shape[1]
#     if len(bunch.target.shape) == 1:
#         data = np.concatenate([bunch.data[:, :], bunch.target[:, None]], axis=1)
#     else:
#         data = np.concatenate([bunch.data[:, :], bunch.target[:, :]], axis=1)
#     fn = bunch.get("feature_names")
#     atts = []
#     for ix in range(data.shape[1]):
#         if fn is not None and len(fn) > ix:
#             atts.append(Attribute(fn[ix], "Ratio", None, None, n_feat <= ix))
#         else:
#             atts.append(Attribute(str(ix), "Ratio", None, None, n_feat <= ix))
#
#     return data, atts
#
# def _create_dataset(bunch, type,source):
#     meta = dict()
#     meta["id"] = str(uuid.uuid4())
#     meta["name"], meta["description"] = _split_DESCR(bunch["DESCR"])
#     meta["type"] = type
#     meta["originalSource"]=source
#     meta["creator"] = ""
#     meta["version"] = ""
#     meta["context"] = {}
#
#     dataset = Dataset(meta["id"], **meta)
#     dataset.set_data(lambda: _create_dataset_data(bunch))
#     return dataset
#
#
# @deprecated(reason ="use updated load_csv function")
# def load_csv_file(path_dataset,path_target=None,target_features=[],originalSource="imported by csv",
#              description="imported form csv",type="multivariate"):
#     """Takes the path of a csv file and a list of the target columns and creates a padre-Dataset.
#
#     Args:
#         path_dataset (str): The path of the csv-file
#         path_target (list): The column names of the target features of the csv-file.
#
#     Returns:
#         pypadre.Dataset() A dataset containing the data of the .csv file
#
#     """
#     assert_condition(condition=os.path.exists(os.path.abspath(path_dataset)), source='ds_import.load_csv',
#                      message='Dataset path does not exist')
#
#     trigger_event('EVENT_WARN', condition=len(target_features)>0, source='ds_import.load_csv',
#                   message='No targets defined. Program will crash when used for supervised learning')
#
#     dataset_path_list = path_dataset.split('/')
#     nameOfDataset = dataset_path_list[-1].split('.csv')[0]
#     data =pd.read_csv(path_dataset)
#
#     meta =dict()
#     meta["name"]=nameOfDataset
#
#     meta["description"]=description
#     meta["originalSource"]=originalSource
#     meta["creator"]=""
#     meta["version"]=""
#     meta["type"]=type
#     meta["context"]={}
#
#     dataset=Dataset(None, **meta)
#     trigger_event('EVENT_WARN', condition=data.applymap(np.isreal).all(1).all() == True, source='ds_import.load_csv',
#                   message='Non-numeric data values found. Program may crash if not handled by estimators')
#
#     targets=None
#     if path_target != None:
#         target = pd.read_csv(path_dataset)
#         data=data.join(target,lsuffix="data",rsuffix="target")
#         targets=list(target.columns.values)
#
#     else:
#         targets=target_features
#
#     atts = []
#
#     for feature in data.columns.values:
#         atts.append(Attribute(feature,None, None, None,feature in targets,None,None))
#
#     dataset.set_data(data,atts)
#     return dataset
#
#
# def load_csv(csv_path, targets=None, name=None, description="imported form csv", source="csvloaded",
#              type="Multivariat"):
#     """Takes the path of a csv file and a list of the target columns and creates a padre-Dataset.
#
#     Args:
#         csv_path (str): The path of the csv-file
#         targets (list): The column names of the target features of the csv-file.
#         name(str): Optional name of dataset
#         description(str): Optional description of the dataset
#         source(str): original source - should be url
#         type(str): type of dataset
#
#     Returns:
#         <class 'pypadre.datasets.Dataset'> A dataset containing the data of the .csv file
#
#     """
#     assert_condition(condition=os.path.exists(os.path.abspath(csv_path)), source='ds_import.load_csv',
#                      message='Dataset path does not exist')
#
#     if targets is None:
#         targets = []
#     trigger_event('EVENT_WARN', condition=len(targets)>0, source='ds_import.load_csv',
#                   message='No targets defined. Program will crash when used for supervised learning')
#
#     dataset_path_list = csv_path.split('/')
#     if name is None:
#         name = dataset_path_list[-1].split('.csv')[0]
#
#     data = pd.read_csv(csv_path)
#     meta = dict()
#     meta["id"] = str(uuid.uuid4())
#     meta["name"] = name
#     meta["description"] = description
#     meta["originalSource"]="http://" + source
#     meta["version"] = 1
#     meta["type"] = type
#     meta["published"] = True
#
#     dataset = Dataset(None, **meta)
#     trigger_event('EVENT_WARN', condition=data.applymap(np.isreal).all(1).all() == True,
#                   source='ds_import.load_csv',
#                   message='Non-numeric data values found. Program may crash if not handled by estimators')
#
#     for col_name in targets:
#         data[col_name] = data[col_name].astype('category')
#         data[col_name] = data[col_name].cat.codes
#     atts = []
#     for feature in data.columns.values:
#         atts.append(Attribute(name=feature,
#                               measurementLevel="Ratio" if feature in targets else None,
#                               defaultTargetAttribute=feature in targets))
#     dataset.set_data(data,atts)
#     return dataset
#
#
# def load_pandas_df(pandas_df,target_features=[]):
#     """
#     Takes a pandas dataframe and a list of the names of target columns and creates a padre-Dataset.
#
#     Args:
#         pandas_df (str): The pandas dataset.
#         path_target (list): The column names of the target features of the csv-file.
#
#     Returns:
#         pypadre.Dataset() A dataset containing the data of the .csv file
#
#     """
#     meta = dict()
#
#     meta["name"] = "pandas_imported_df"
#     meta["description"]="imported by pandas_df"
#     meta["originalSource"]="https://imported/from/pandas/Dataframe.html"
#     meta["creator"]=""
#     meta["version"]=""
#     meta["context"]={}
#     meta["type"]="multivariate"
#     dataset = Dataset(None, **meta)
#
#     atts = []
#
#     if len(target_features) == 0:
#         targets = [0] * len(pandas_df)
#
#     for feature in pandas_df.columns.values:
#         atts.append(Attribute(name=feature, measurementLevel=None, unit=None, description=None,
#                               defaultTargetAttribute=feature in target_features, context=None))
#     dataset.set_data(pandas_df, atts)
#     return dataset
#
#
# def load_numpy_array_multidimensional(features, targets, columns=None, target_features=None):
#     """
#     Takes a multidimensional numpy array and creates a dataset out of it
#     :param features: The input n dimensional numpy array
#     :param targets: The targets corresponding to every feature
#     :param columns: Array of data column names
#     :param target_features: Target features column names
#     :return: A dataset object
#     """
#     meta = dict()
#
#     meta["name"] = "numpy_imported"
#     meta["description"] = "imported by numpy multidimensional"
#     meta["originalSource"] = ""
#     meta["creator"] = ""
#     meta["version"] = ""
#     meta["context"] = {}
#     meta["type"] = "multivariate"
#     dataset = Dataset(None, **meta)
#     atts = []
#
#     if len(target_features) == 0:
#         targets = [0] * len(features)
#
#     for feature in columns:
#         atts.append(Attribute(name=feature, measurementLevel=None, unit=None, description=None,
#                               defaultTargetAttribute=feature in target_features, context=None))
#     dataset.set_data_multidimensional(features, targets, atts)
#     return dataset


# def load_sklearn_toys():
#     #returns an iterator loading different sklearn datasets
#     loaders = [(ds.load_boston, ("regression", "Multivariat"),"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston"),
#                (ds.load_breast_cancer, ("classification", "Multivariat"),"https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)"),
#                (ds.load_diabetes, ("regression", "Multivariat"),"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes"),
#                (ds.load_digits, ("classification", "Multivariat"),"http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits"),
#                (ds.load_iris, ("classification", "Multivariat"),"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris"),
#                (ds.load_linnerud, ("mregression", "Multivariat"),"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html#sklearn.datasets.load_linnerud")]
#
#     for loader in loaders:
#         yield _create_dataset(loader[0](), loader[1][1],loader[2])

#possible Datatypes of imported open-ml dataset columns
LEGAL_DATA_TYPES = ['NOMINAL','INTEGER', 'NUMERIC', 'REAL', 'STRING']

DATATYPE_MAP={'INTEGER': np.int64,'NUMERIC':np.float64, 'REAL':np.float64, 'STRING': str}



def storeDataset(dataset,file_backend):
    if dataset.id not in file_backend.datasets.list_datasets():
        file_backend._dataset_repository.put_dataset(dataset)

def __check_ds_meta(dataset_meta,check_key_value):
    for key in check_key_value.keys():
        try:
            if dataset_meta[key] != check_key_value[key]:
                return False
        except KeyError:
            return False
    return True


def get_did_from_meta(check_dict,auth_token,max_hits=10,url="http://localhost:8080"):
    """Takes a dictionary of dataset attributes to search for. Searches the Server for a Dataset, that matches the
    requirements. Returns the dids of the matches.

    Args:
        check_dict (dict): A dictionary, that specifies the required metadata of a dataset.
        auth_token (str): The auth_token for authentication at the server.
        max_hits (int): Amount of maximum results.
        url (str): The url of the server.

    Returns:
        pypadre.Dataset() A list of several did, that fulfill the required check_dict.

    """
    hed = {'Authorization': auth_token}

    morePages=True
    hit_list = []
    page=0

    while len(hit_list)<max_hits and morePages:
        url = url+"/api/datasets?page=" + str(page) + "&size=9"
        response = requests.get(url, headers=hed)
        try:
            content = json.loads(response.content,encoding="utf-8")

            for dataset_meta in content["_embedded"]["datasets"]:
                if(__check_ds_meta(dataset_meta,check_dict)):

                    hit_list.append(dataset_meta["uid"])
        except KeyError as err:
            print("invalid name!"+str(err))

        page+=1
        if content["page"]["totalPages"]<=page:
            morePages=False
    if len(hit_list)>max_hits:
        hit_list=hit_list[:max_hits]
    return hit_list


def getDataset_load_or_cached(did,file_backend,force_download=False,auth_token=None):
    """Fetches the requested Dataset from the Server or (if available) from the local cache. A downloaded dataset gets
    cached. Returns the Dataset as pypadre.Dataset().

    Args:
        did (str): id of the requested dataset
        path (str): path of the pypadre directory
        force_download (bool): If set to True, downloads the dataset from the server unconditionally.
        auth_token (str): The Token for identification.

    Returns:
        pypadre.Dataset() A dataset containing with the requested data.

    """


    dataset=None

    if(force_download or did not in file_backend.datasets.list_datasets(search_metadata="")):
        dataset=requestServerDataset(did,auth_token)
        file_backend.datasets.put_dataset(dataset)
        return dataset
    return file_backend.datasets.get_dataset(did)


def sendTop100Datasets_single(auth_token,server_url="http://localhost:8080"):
    """Takes a list of the Top-100 Datasets and downloads all of them from open-ml and uploads them to the Server.
        Those Datasets are not cached locally. The list of Datasets is available under
        /datasets/config/top100datasetIDs.txt and can be customized.

    Args:
        auth_token (str): The Token for identification.

    Returns:
        pypadre.Dataset() A dataset containing with the given source

    """
    for i in pypadre.ds_import.load_sklearn_toys():
        pypadre.ds_import.createServerDataset(i, auth_token, server_url)

    data = "18,12,22,23,28,60,46,32,36,14,1112,1114,1120,1489,1494,1497,1501,1067,1068,300,1049,1050,1053,182," \
           "4135,4134,1487,1466,1471,1475,6,4534,4538,38,3,1504,23512,24,1493,44,554,11,1038,29,151,15,40981," \
           "40499,42,1590,307,16,37,6332,1476,1479,458,1480,334,335,333,1515,188,1461,1046,1063,1467,1459,1464," \
           "50,1478,377,54,375,451,40496,1462,1485,1510,40668,1468,40536,1486,23380,23381,470,469,20,312,1492,1491,31"

    id_list = data.split(",")

    i=0
    amount=str(len(id_list))
    for id in id_list:
        print("Progress: ("+str(i)+"/"+amount+") id of next dataset:" + str(id))
        ds = load_openML_dataset("/" + id, destpath=None)
        createServerDataset(ds, auth_token, server_url)
        i = i+1


def _scratchSnap(url_list,auth_token,server_url="http://localhost:8080"):
    for url in url_list:
        ds = pypadre.graph_import.create_from_snap(url,0)
        createServerDataset(ds,auth_token,server_url)


def _scratchKonect(url_list,auth_token,server_url="http://localhost:8080"):
    for url in url_list:
        ds = pypadre.graph_import.create_from_konect(url)
        createServerDataset(ds,auth_token,server_url)


def send_top_graphs(auth_token,server_url="http://localhost:8080",multithread=False):
    """Downloads a set of Datasets from snap and konect and uploads them to the server.

    Args:
        path (str): path of the pypadre directory
        auth_token (str): The Token for identification.
        multithread (bool): Requires at least 12GB of ram
    Returns:
        pypadre.Dataset() A dataset containing with the given source

    """
    #"https://snap.stanford.edu/data/gemsec-Deezer.html","https://snap.stanford.edu/data/gemsec-Facebook.html",
    snap_normal=[
        "https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html",
        "https://snap.stanford.edu/data/com-Youtube.html",#failed
        "https://snap.stanford.edu/data/ca-CondMat.html",
        "https://snap.stanford.edu/data/Oregon-2.html",
        "https://snap.stanford.edu/data/web-Google.html",#failed
        "https://snap.stanford.edu/data/web-Stanford.html",#failed
        "https://snap.stanford.edu/data/roadNet-TX.html",#failed
        "https://snap.stanford.edu/data/CollegeMsg.html",
        "https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html"]#failed

    snap_biosnap=["https://snap.stanford.edu/biodata/datasets/10001/10001-ChCh-Miner.html",
        "https://snap.stanford.edu/biodata/datasets/10017/10017-ChChSe-Decagon.html",
        "https://snap.stanford.edu/biodata/datasets/10024/10024-GF-Miner.html",
        "https://snap.stanford.edu/biodata/datasets/10029/10029-SS-Butterfly.html",
        "https://snap.stanford.edu/biodata/datasets/10021/10021-D-DoMiner.html",
        "https://snap.stanford.edu/biodata/datasets/10025/10025-D-OmimMiner.html",
        "https://snap.stanford.edu/biodata/datasets/10011/10011-G-MtfPathways.html"]

    konect=["http://konect.cc/networks/edit-rnwiktionary/",
        "http://konect.cc/networks/league-be1-2016/",
        "http://konect.cc/networks/moreno_train/",
        "http://konect.cc/networks/moreno_highschool/",
        "http://konect.cc/networks/maayan-faa/",
        "http://konect.cc/networks/petster-hamster-household/",
        "http://konect.cc/networks/flickrEdges/",#failed
        "http://konect.cc/networks/stackexchange-stackoverflow/",#failed
        "http://konect.cc/networks/lasagne-yahoo/",#failed
        "http://konect.cc/networks/youtube-u-growth/",
        "http://konect.cc/networks/dimacs9-E/"]

    if multithread:
        snap_normal_p = Process(target=_scratchSnap, args=(snap_normal,auth_token,  server_url))
        snap_biosnap_p = Process(target=_scratchSnap, args=(snap_biosnap,auth_token,  server_url))
        konect_p = Process(target=_scratchKonect, args=(konect,auth_token,  server_url))
        snap_normal_p.start()
        snap_biosnap_p.start()
        konect_p.start()
        snap_normal_p.join()
        snap_biosnap_p.join()
        konect_p.join()
    else:
        _scratchSnap(snap_normal,auth_token, server_url)
        _scratchSnap(snap_biosnap, auth_token, server_url)
        _scratchKonect(konect, auth_token, server_url)

def sendTop100Datasets_multi(auth_token,server_url="http://localhost:8080",worker=1):
    """Takes a list of the Top-100 Datasets and downloads all of them from open-ml and uploads them to the Server.
        Those Datasets are not cached locally. The list of Datasets is available under
        /datasets/config/top100datasetIDs.txt and can be customized.

    Args:
        path (str): path of the pypadre directory. If None, a predefined list of datasets is taken.
        auth_token (str): The Token for identification.
        server_url (str): The url of the server
        worker (int): The amount of Threads that should be used. Ensure at least 2GB of free Ram for each Thread!

    """

    print("Upload of toy scikit- toydatasets")
    for i in pypadre.ds_import.load_sklearn_toys():
        pypadre.ds_import.createServerDataset(i, auth_token, server_url)

    data = "11,12,14,15,16,18,20,3,6,32,36,38,22,23,24,28,29,42,44,46,54,60,182,188,151,300,312,307,375,377,333,334,335," \
         "451,458,469,470,554,1046,1049,1050,1038,1114,1120,1063,1067,1068,1053,1459,1471,1479,1480,1466,1467,1486," \
         "1489,1504,4135,23380,1461,1476,1475,1492,1491,1485,1468,1501,1462,1487,1494,1493,1478,1590,1112,1515,1510," \
         "1497,23381,4538,23512,4134,6332,4534,1464,37,31,50,40536,40496,40668,40499,40981"




    id_list = data.split(",")

    #datasets = []
    i=0
    import math
    amount=len(id_list)
    workerDatasets=[[] for x in range(worker)]
    for i in range(worker):
        for j in range(0,int(math.floor(amount/worker))):
            workerDatasets[i].append(id_list.pop())
        if(i<amount%worker):
            workerDatasets[i].append(id_list.pop())

    plist=[]
    for i in range(worker):
        p=Process(target=_sendDatasetWorker,args=(auth_token,workerDatasets[i],i,server_url))
        p.start()
        plist.append(p)
        print("thread started: "+str(i))
    for i in plist:
        i.join()


def _sendDatasetWorker(auth_token,id_list,worker,server_url):
    """This function gets called by each Thread of sendTop100DatasetsToServer. For each entry in id_list, it
    downloads the corresponding dataset from openml and uploades it to server_url.

    Args:
        path (str): path of the pypadre directory. If None, the local data is stored in a temporary directory.
        auth_token (str): The Token for identification.
        id_list (list): A list containing all the id's of the ompenml-datasets.
        worker (int): The amount of Threads that should be used. Ensure at least 2GB of free Ram for each Thread!
        server_url (str): The url of the server
    """
    i=0
    for id in id_list:
        amount = str(len(id_list))
        print("Worker: "+str(worker)+" Progress: (" + str(i) + "/" + amount + ") id of next dataset:" + str(id))
        ds = load_openML_dataset("/" + id, destpath=None)
        did=createServerDataset(ds,auth_token,server_url)
        i = i + 1


def requestServerDataset(did,auth_token,url="http://localhost:8080"):
    """Downloads a dataset and the matching meta-data from the server and converts it to a padre.Dataset

    Args:
        did (str): The id of the Dataset of interest.
        auth_token (str): The Token for identification.
        url (str): The url of the server.

    Returns:
        (pypadre.Dataset) A Dataset that contains the requested Data.

    """
    hed = {'Authorization': auth_token}
    response = requests.get(url+"/api/datasets/"+str(did), headers=hed)

    k = response.content.decode("utf-8")
    response_meta=json.loads(k)
    response.close()
    requests.session().close()

    attribute_name_list = []
    atts = []
    for attr in response_meta["attributes"]:
        if attr["name"] != "INVALID_COLUMN":
            atts.append(Attribute(**attr))
        attribute_name_list.append(attr["name"])

    del response_meta["attributes"]
    df_data = proto.get_Server_Dataframe(did, auth_token, url=url)
    dataset = Dataset(None, **response_meta)
    df_data.columns = attribute_name_list

    if "INVALID_COLUMN" in list(df_data.columns.values):
        df_data=df_data.drop(["INVALID_COLUMN"], axis=1)

    if dataset.isgraph:
        node_attr = []
        edge_attr = []

        for attr in atts:
            graph_role = attr.context["graph_role"]
            if graph_role == "source":
                source = attr.name
            elif graph_role == "target":
                target = attr.name
            elif graph_role == "nodeattribute":
                node_attr.append(attr.name)
            elif graph_role == "edgeattribute":
                edge_attr.append(attr.name)
        network=nx.Graph()if response_meta["type"]=="graph" else nx.DiGraph()
        pypadre.graph_import.pandas_to_networkx(df_data, source, target, network, node_attr, edge_attr)
        dataset.set_data(network, atts)
    else:
        dataset.set_data(df_data, atts)
        print("Load dataset "+did+" from server:")

    return dataset



def createServerDataset(dataset,auth_token,url="http://localhost:8080"):
    """Creates a dataset on the server and transferees its' content. It returns a
    String, that stands for the id the Dataset on the server side.

    Args:
        dataset (pypadre.Dataset): Dataset that metadata should be used to create a new dataset at the server
        auth_token (str): Token for identification for communication with the server.
        url (str): The url of the server.

    Returns:
        str: did of the created dataset at the server. The datset can be downloaded using this did

    """

    binary = tempfile.TemporaryFile(mode='w+b')

    proto_enlarged = pypadre.pod.backend.http.http.protobuffer.proto_organizer.createProtobuffer(dataset, binary)

    hed = {'Authorization': auth_token}

    attributes=copy.deepcopy(dataset.attributes)
    if proto_enlarged:
        attributes.append({"name":"INVALID_COLUMN","context":"{}"})

    data=dataset.metadata
    data["attributes"]=attributes
    response=requests.post(url+"/api/datasets",json=data,headers=hed)

    for i in range(3):
        if "Location" in response.headers:
            did = str((response.headers["Location"]).split("/")[-1])
            break;
        else:
            print("retry sending for dataset" + str(data["name"]))
            print("statuscode: "+ str(response.status_code))
            print("header: " +str(response.headers))
            print("content: "+ str(response.content))
            response = requests.post(url + "/api/datasets", json=data, headers=hed)

    if "Location" not in response.headers:
        raise ValueError('Server does not accept metadata of dataset. Dataset could not be uploaded!, response header. '
                         + str(response.headers)+" response content: "+str(response.content))

    binary.seek(0)
    print("Send dataset " + str(dataset.name) + " to server")
    proto.send_Dataset(dataset,did,auth_token,binary,url=url)
    binary.close()


    response.close()
    requests.session().close()
    del data["attributes"]
    return did

def search_oml_datasets(name, root_dir, key):
    path = root_dir + '/temp/openml'

    oml.config.apikey = key
    oml.config.cache_directory = path
    meta = {"data_name": name}
    return oml.datasets.list_datasets(**meta)


def load_openML_dataset(url,destpath=os.path.expanduser('~/.pypadre'),apikey="1f8766e1615225a727bdea12ad4c72fa"):
    """Downloads a dataset from the given open-ml url and Converts it to a padre.Dataset. The metadata is also added to
    the pypadre.Dataset

    Args:
        url (str): url of the open-ml dataset
        destpath (str): path of padre directory. If None, the directory of openml will be in a temporary directory. As
        a result openml will not cache the dataset arff files.
        apikey (str): apikey of openml. The default value is linked to the openml account of markush.473@gmail.com

    Returns:
        pypadre.Dataset() A dataset filled with the values of the given source.

    """
    # apikey is from useraccount markush.473@gmail.com

    if destpath is None:
        path = tempfile.mkdtemp()

    else:
        path = destpath+'/datasets/temp/openml'
    dataset_id = url.split("/")[-1]
    dataset_id = dataset_id.strip(" ")
    oml.config.apikey = apikey
    oml.config.cache_directory = path
    try:
        load = oml.datasets.get_dataset(dataset_id)
    except oml.exceptions.OpenMLServerException as err:
        print("Dataset not found! \nErrormessage: " + str(err))
        return None
    except ConnectionError as err:
        print("openML unreachable! \nErrormessage: " + str(err))
        return None
    except OSError as err:
        print("Invalid datapath! \nErrormessage: " + str(err))
        return None

    meta = dict()
    meta["name"] = load.name
    meta["version"] = load.version
    meta["description"] = load.description
    meta["originalSource"] = load.url
    meta["type"] = "multivariate"
    meta["published"] = False

    dataset = Dataset(None, **meta)
    raw_data = arff.load(open(path+'/org/openml/www/datasets/'+dataset_id+'/dataset.arff', encoding='utf-8'))
    df_attributes = raw_data['attributes']
    attribute_list = []

    for att in df_attributes:
        attribute_list.append(att[0])

    df_data = pd.DataFrame(data=raw_data['data'])
    raw_data = None
    atts = []

    for col in df_data.keys():
        data_class=None
        current_attribute=df_attributes[col]
        if load.features[col].name!=current_attribute[0]:
            print("Name failure. Inconsistency encountered!")

        if isinstance(current_attribute[1],list):
            data_class='NOMINAL'
            df_data[col] = df_data[col].astype('category')
        elif current_attribute[1] in DATATYPE_MAP.keys() and isinstance(current_attribute[1],str):
            data_class = current_attribute[1]
        else:
            print("failed to recognize format")
            raise ValueError('Invalid data format from openml!')

        atts.append(Attribute(name=current_attribute[0], measurementLevel="nominal" if isinstance(current_attribute[1],list) else None,
                              unit=None, description=None, defaultTargetAttribute=(current_attribute[0] == load.default_target_attribute)))

    df_data.columns = attribute_list
    dataset.set_data(df_data, atts)

    return dataset
