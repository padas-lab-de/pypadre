"""
Modul containing import methods from different packages / repositories.
"""
import tempfile

import sklearn.datasets as ds
import numpy as np
from .datasets import Dataset, Attribute
import openml as oml
import os.path
from requests.exceptions import ConnectionError
import arff
import pandas as pd
import json
import requests
import padre.protobuffer.proto_organizer as proto
import padre.backend.file
import padre.graph
import networkx as nx
import os.path


def _split_DESCR(s):
    s = s.strip()
    k = s.find("\n")
    return s[0:k], s[k + 1:]


def _create_dataset_data(bunch):
    n_feat = bunch.data.shape[1]
    if len(bunch.target.shape) == 1:
        data = np.concatenate([bunch.data[:, :], bunch.target[:, None]], axis=1)
    else:
        data = np.concatenate([bunch.data[:, :], bunch.target[:, :]], axis=1)
    fn = bunch.get("feature_names")
    atts = []
    for ix in range(data.shape[1]):
        if fn is not None and len(fn) > ix:
            atts.append(Attribute(fn[ix], "ratio", None, None, n_feat <= ix))
        else:
            atts.append(Attribute(str(ix), "ratio", None, None, n_feat <= ix))

    return data, atts

def _create_dataset(bunch, type):
    meta = dict()
    meta["name"], meta["description"] = _split_DESCR(bunch["DESCR"])
    meta["type"] = type
    dataset =  Dataset(None, **meta)
    dataset.set_data(*_create_dataset_data(bunch))
    return dataset

def load_csv(path_dataset,path_target=None,target_features=[]):
    dataset_path_list = path_dataset.split('/')
    nameOfDataset = dataset_path_list[-1].split('.csv')[0]
    data =pd.read_csv(path_dataset)
    meta =dict()
    meta["name"]=nameOfDataset

    meta["description"]="imported by csv"
    meta["originalSource"]=""
    meta["creator"]=""
    meta["openml_id"]=""
    meta["version"]=""

    dataset=Dataset(None, **meta)

    #column_names = list(data.columns.values)
    #n_feat = data.shape[1]
    targets=None
    if path_target != None:
        target = pd.read_csv(path_dataset)
        data=data.join(target,lsuffix="data",rsuffix="target")
        targets=list(target.columns.values)
    else:
        targets=target_features


        """
        if list(data.columns.values).__contains__(target_feature):
            targets=target_feature
        else:
            targets=[]"""

    atts = []

    for feature in data.columns.values:
        atts.append(Attribute(feature,None, None, None,feature in targets,None,None,None))

    dataset.set_data(data,atts)
    return dataset

def load_pandas_df(pandas_df,target_features=[]):
    meta = dict()
    meta["name"] = "pandas_imported_df"
    meta["description"]="imported by pandas_df"
    meta["originalSource"]=""
    meta["creator"]=""
    meta["openml_id"]=""
    meta["version"]=""
    dataset = Dataset(None, **meta)

    atts = []

    for feature in pandas_df.columns.values:
        atts.append(Attribute(feature, None, None, None, feature in target_features, None, None, None))

    dataset.set_data(pandas_df, atts)
    return dataset


def load_sklearn_toys():
    #returns an iterator loading different sklearn datasets
    loaders = [(ds.load_boston, ("regression", "multivariate")),
               (ds.load_breast_cancer, ("classification", "multivariate")),
               (ds.load_diabetes, ("regression", "multivariate")),
               (ds.load_digits, ("classification", "multivariate")),
               (ds.load_iris, ("classification", "multivariate")),
#              ds.load_lfw_pairs,
#              ds.load_lfw_people,
               (ds.load_linnerud, ("mregression", "multivariate"))]

    for loader in loaders:
        yield _create_dataset(loader[0](), loader[1][1])

#possible Datatypes of imported open-ml dataset columns
LEGAL_DATA_TYPES = ['NOMINAL','INTEGER', 'NUMERIC', 'REAL', 'STRING']

DATATYPE_MAP={'INTEGER': np.int64,'NUMERIC':np.float64, 'REAL':np.float64, 'STRING': str}



def storeDataset(dataset,path):
    file_backend = padre.backend.PadreFileBackend(path)
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

def get_did_from_ds_name(check_key_value,auth_token,max_hits=10):
    hed = {'Authorization': 'Bearer ' + auth_token}

    morePages=True
    hit_list = []
    page=0

    while len(hit_list)<max_hits and morePages:
        url = "http://localhost:8080/api/datasets?page=" + str(page) + "&size=9"
        response = requests.get(url, headers=hed)
        try:
            content = json.loads(response.content,encoding="utf-8")

            for dataset_meta in content["_embedded"]["datasets"]:
                if(__check_ds_meta(dataset_meta,check_key_value)):

                    hit_list.append(dataset_meta["uid"])
        except KeyError as err:
            print("invalid name!"+str(err))

        page+=1
        if content["page"]["totalPages"]<=page:
            morePages=False

    return hit_list




def getDataset_load_or_cached(did,path,force_download=False,auth_token=None):
    """Fetches the requested Dataset from the Server or (if available) from the local cache. A downloaded dataset gets
    cached. Returns the Dataset as padre.Dataset().

    Args:
        did (str): id of the requested dataset
        path (str): path of the pypadre directory
        force_download (bool): If set to True, downloads the dataset from the server unconditionally.
        auth_token (str): The Token for identification.

    Returns:
        padre.Dataset() A dataset containing with the requested data.

    """
    t=proto.start_measure_time()

    dataset=None
    file_backend = padre.backend.file.PadreFileBackend(path)

    if(force_download or did not in file_backend.datasets.list_datasets(search_metadata="")):
        dataset=requestServerDataset(did,auth_token)
        file_backend.datasets.put_dataset(dataset)
        print("Time to load dataset from server:", end=" ")
        proto.end_measure_time(t)
        return dataset
    print("Time to load dataset from cache:",end=" ")
    proto.end_measure_time(t)
    return file_backend.datasets.get_dataset(did)


def sendTop100DatasetsToServer_old(path,auth_token):
    """Takes a list of the Top-100 Datasets and downloads all of them from open-ml and uploads them to the Server.
        Those Datasets are not cached locally. The list of Datasets is available under
        /datasets/config/top100datasetIDs.txt and can be customized.

    Args:
        path (str): path of the pypadre directory
        auth_token (str): The Token for identification.

    Returns:
        padre.Dataset() A dataset containing with the given source

    """
    path100=path+"/datasets/config/top100datasetIDs.txt"

    data="18,12,22,23,28,60,46,32,36,14,1112,1114,1120,1489,1494,1497,1501,1067,1068,300,1049,1050,1053,182,4135,4134,1487,1466,1471,1475,6,4534,4538,38,3,1504,23512,24,1493,44,554,11,1038,29,151,15,40981,40499,42,1590,307,16,37,6332,1476,1479,458,1480,334,335,333,1515,188,1461,1046,1063,1467,1459,1464,50,1478,377,54,375,451,40496,1462,1485,1510,40668,1468,40536,1486,23380,23381,470,469,20,312,1492,1491,31"
    if(os.path.exists(path100)):
        file = open(path100, "r")
        with open(path100, "r") as file:
            data=file.read()

    else:
        os.makedirs(os.path.dirname(path100), exist_ok=True)
        file=open(path100,"w")
        file.write(data)
        file.flush()
        file.close

    id_list = data.split(",")

    #datasets = []
    i=0
    amount=str(len(id_list))
    for id in id_list:
        print("Progress: ("+str(i)+"/"+amount+") id of next dataset:" + str(id))
        ds=load_openML_dataset("/" + id,path)
        did=createServerDataset(ds,path,auth_token)
        #proto.send_Dataset(ds, did, auth_token, path)
        #datasets.append(load_openML_dataset_new("/" + id))
        i=i+1


def sendTop100DatasetsToServer(path,auth_token,server_url="http://localhost:8080",worker=1):
    from multiprocessing import Process
    """Takes a list of the Top-100 Datasets and downloads all of them from open-ml and uploads them to the Server.
        Those Datasets are not cached locally. The list of Datasets is available under
        /datasets/config/top100datasetIDs.txt and can be customized.

    Args:
        path (str): path of the pypadre directory
        auth_token (str): The Token for identification.

    Returns:
        padre.Dataset() A dataset containing with the given source

    """
    #t=proto.start_measure_time()

    path100=path+"/datasets/config/top100datasetIDs.txt"

    data="11,12,14,15,16,18,20,3,6,32,36,38,22,23,24,28,29,42,44,46,54,60,182,188,151,300,312,307,375,377,333,334,335," \
         "451,458,469,470,554,1046,1049,1050,1038,1114,1120,1063,1067,1068,1053,1459,1471,1479,1480,1466,1467,1486," \
         "1489,1504,4135,23380,1461,1476,1475,1492,1491,1485,1468,1501,1462,1487,1494,1493,1478,1590,1112,1515,1510," \
         "1497,23381,4538,23512,4134,6332,4534,1464,37,31,50,40536,40496,40668,40499,40981"
    if(path is None):
        data=data

    elif(os.path.exists(path100)):
        file = open(path100, "r")
        with open(path100, "r") as file:
            data=file.read()

    else:
        os.makedirs(os.path.dirname(path100), exist_ok=True)
        file=open(path100,"w")
        file.write(data)
        file.flush()
        file.close

    id_list = data.split(",")

    #datasets = []
    i=0
    import _thread
    import math
    amount=len(id_list)
    workerDatasets=[[] for x in range(worker)]
    for i in range(worker):
        for j in range(0,int(math.floor(amount/worker))):
            workerDatasets[i].append(id_list.pop())
        if(i<amount%worker):
            workerDatasets[i].append(id_list.pop())
    print("leftovers:")
    print(id_list)
    plist=[]
    for i in range(worker):
        p=Process(target=sendDatasetWorker,args=(path,auth_token,workerDatasets[i],i,server_url))
        p.start()
        plist.append(p)
        #p.join()
        #_thread.start_new_thread(sendDatasetWorker,(path,auth_token,workerDatasets[i],i))
        print("thread started: "+str(i))
    for i in plist:
        i.join()
    #
    #proto.end_measure_time(t)

def sendDatasetWorker(path,auth_token,id_list,worker,server_url):
    i=0
    for id in id_list:
        t2 = proto.start_measure_time()
        amount = str(len(id_list))
        print("Progress: (" + str(i) + "/" + amount + ") id of next dataset:" + str(id))
        ds = load_openML_dataset("/" + id, path)
        did=createServerDataset(ds,auth_token,server_url)
        #did = createServerDataset(ds, path, auth_token)
        #proto.send_Dataset(ds, did, auth_token, path)
        # datasets.append(load_openML_dataset_new("/" + id))
        i = i + 1
        print("Time for dataset " + str(i) + " to fetch and send to server:", end=" ")
        proto.end_measure_time(t2)
        #print("Time to fetch and send all datasets to server:", end=" ")

def requestServerDataset(did,auth_token,url="http://localhost:8080"):
    """Downloads a dataset and the matching meta-data from the server and converts it to a padre.Dataset()

    Args:
        did (str): The id of the Dataset of interest.
        auth_token (str): The Token for identification.

    Returns:
        padre.Dataset() A Dataset that contains the requested Data.

    """
    t=proto.start_measure_time()

    hed = {'Authorization': 'Bearer ' + auth_token}
    response = requests.get(url+"/api/datasets/"+str(did), headers=hed)

    k=response.content.decode("utf-8")
    response_meta=json.loads(k)
    response.close()
    requests.session().close()

    meta = dict()
    meta["openml_id"] = response_meta["uid"]
    #meta["links"]=response_meta["links"]
    meta["name"] = response_meta["name"]
    meta["version"] = response_meta["version"]
    meta["description"] = response_meta["description"]
    meta["creator"] = response_meta["owner"]
    meta["published"]=response_meta["published"]
    meta["standardSplitting"]=response_meta["standardSplitting"]
    meta["type"]=response_meta["type"]
    meta["contributor"] = None
    meta["collection_date"] = None
    meta["upload_to_openml_date"] = None
    meta["language"] = None
    meta["licence"] = None
    meta["originalSource"] = response_meta["originalSource"]
    meta["version_label"] = response_meta["version"]
    meta["citation"] = None
    meta["paper_url"] = None
    meta["update_comment"] = None
    meta["qualities"] = None



    df_data = proto.get_Server_Dataframe(did,auth_token,url=url)





    #df_data = pd.DataFrame()

    attribute_name_list = []
    attribute_list = []
    #print(response_meta["attributes"])
    for att in response_meta["attributes"]:
        if att["name"] != "INVALID_COLUMN":
            attribute_list.append(Attribute(att["name"],att["measurementLevel"], att["unit"],att["description"],att["defaultTargetAttribute"]))
            if att["defaultTargetAttribute"]:
                meta["default_target_attribute"]=att["name"]
            attribute_name_list.append(att["name"])
            print(att["measurementLevel"])
        else:
            attribute_name_list.append(att["name"])

    dataset = Dataset(None, **meta)
#    atts = []
#    #Begründung für +1 hinzufügen
#    for ix in range(0, df_data.shape[1] + 1):
#        atts.append(
#            Attribute(response_meta["attributes"][ix]["name"], response_meta["attributes"][ix]["measurementLevel"],response_meta["attributes"][ix]["unit"], response_meta["attributes"][ix]["description"], None, response_meta["attributes"][ix]["name"] == response_meta["defaultTargetAttribute"],
#                      None,
#                      None))

    df_data.columns = attribute_name_list

    if("INVALID_COLUMN" in list(df_data.columns.values)):
        df_data=df_data.drop(["INVALID_COLUMN"],axis=1)
        print(df_data)

    dataset.set_data(df_data, attribute_list)
    print("Time to load dataset "+did+" from server:",end=" ")
    proto.end_measure_time(t)
    return dataset



def createServerDataset(dataset,auth_token,url="http://localhost:8080"):
    """Creates a dataset on the server and transferees its' content. It returns a
    String, that stands for the id the Dataset on the server side.

    Args:
        dataset (padre.Dataset): Dataset that metadata should be used to create a new dataset at the server
        auth_token (str): Token for identification for communication with the server.

    Returns:
        str: did of the created dataset, that can be used to upload the dataset to

    """
    t=proto.start_measure_time()

    binary= tempfile.TemporaryFile(mode='w+b')

    proto_enlarged=padre.protobuffer.proto_organizer.createProtobuffer(dataset,binary)

    hed = {'Authorization': 'Bearer ' + auth_token}

    attributes=[]
    attributeNum=0
    for attribute in dataset.attributes:
        col={}
        col["defaultTargetAttribute"]=attribute.is_target
        col["description"]=attribute.description
        col["index"]=attributeNum
        col["links"] = " "
        col["measurementLevel"] =attribute.measurement_level
        col["name"]=attribute.name
        col["unit"]=attribute.unit
        attributes.append(col)
        attributeNum+=1
    if proto_enlarged:
        attributes.append({"name":"INVALID_COLUMN"})

    data={}
    data["attributes"]=attributes
    data["description"]=dataset.metadata["description"]
    data["links"]=" "
    data["name"] = dataset.metadata["name"]
    data["originalSource"] = dataset.metadata["originalSource"]
    if(dataset.metadata["creator"]is not None):
        data["owner"] = ''.join(dataset.metadata["creator"])
    else:
        data["owner"] =None
    data["published"] = "false"
    data["standardSplitting"] = "TODO"
    data["type"] = "multivariate"
    data["uid"] = dataset.metadata["openml_id"]
    data["version"] = dataset.metadata["version"]


    response=requests.post(url+"/api/datasets",json=data,headers=hed)
    print(response.headers)


    did=str((response.headers["Location"]).split("/")[-1])
    binary.seek(0)
    proto.send_Dataset(dataset,did,auth_token,binary,url=url)
    binary.close()
    print("Time to send dataset " + str(dataset.id) + " to server:", end=" ")
    proto.end_measure_time(t)
    response.close()
    requests.session().close()
    return did

def add_snap_csv(source_col_number,target_col_number,edge_attribute_dict,node_attribute_dict,filepath):
    return nx.read_edgelist(filepath,create_using=nx.Graph())



def load_openML_dataset(url,destpath=os.path.expanduser('~/.pypadre'),apikey="1f8766e1615225a727bdea12ad4c72fa"):
    """Downloads a dataset from the given open-ml url or takes it from the cache. Transforms it to a padre.Dataset

    Args:
        param1 (str): url of the open-ml dataset
        param2 (str): path of padre directory. If None, the directory of openml will be in a temporary directory.
        param3 (str): apikey of open-ml for login

    Returns:
        padre.Dataset() A dataset filled with the given source

    """
    # apikey is from useraccount markush.473@gmail.com
    import shutil
    if destpath is None:
        path=tempfile.mkdtemp()

    else:
        path=destpath+'/datasets/temp/openml'
    #apikey is from useraccount markush.473@gmail.com
    #raw_data = arff.load(open(os.path.expanduser('~/.openml/cache')+'/org/openml/www/datasets/14/dataset.arff',encoding='utf-8'))
    dataset_id=url.split("/")[-1]
    dataset_id=dataset_id.strip(" ")
    oml.config.apikey=apikey
    oml.config.cache_directory=path
    try:
        load = oml.datasets.get_dataset(dataset_id)
    except oml.exceptions.OpenMLServerException as err:
        print("Dataset not found! \nErrormessage: "+ str(err))
        return None
    except ConnectionError as err:
        print("openML unreachable! \nErrormessage: "+str(err))
        return None
    except OSError as err:
        print("Invalid datapath! \nErrormessage: " + str(err))
        return None

    meta = dict()
    meta["openml_id"]=dataset_id
    meta["name"] = load.name
    meta["version"]=load.version
    meta["description"] = load.description
    meta["creator"] = load.creator
    meta["contributor"] = load.contributor
    meta["collection_date"] = load.collection_date
    meta["upload_to_openml_date"] = load.upload_date
    meta["language"] = load.language
    meta["licence"] = load.licence
    meta["originalSource"] = load.url
    meta["default_target_attribute"]=load.default_target_attribute
    meta["version_label"] = load.version_label
    meta["citation"] = load.citation
    meta["paper_url"] = load.paper_url
    meta["update_comment"] = load.update_comment
    meta["qualities"] = load.qualities


    dataset = Dataset(None, **meta)
    raw_data = arff.load(open(path+'/org/openml/www/datasets/'+dataset_id+'/dataset.arff',encoding='utf-8'))
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
        if(load.features[col].name!=current_attribute[0]):
            print("failure")
            #TODO Throw exception
        if isinstance(current_attribute[1],list):
            #data_type=str
            data_class='NOMINAL'
            df_data[col] = df_data[col].astype('category')
            #print(df_data[col]._data.categrical)
           # pd.Categorical
           # pd.Series
        elif current_attribute[1] in DATATYPE_MAP.keys() and isinstance(current_attribute[1],str):
            #data_type=DATATYPE_MAP[current_attribute[1]]
            data_class = current_attribute[1]
        else:
            print("failure")
            #TODO Throw exception
            df_data.keys()[0]
        atts.append(Attribute(current_attribute[0], None, None, None, current_attribute[0] == load.default_target_attribute,
                              data_class,current_attribute[1] if data_class is "NOMINAL" else None,
                              load.features[col].number_missing_values))
    #+atts.append(Attribute(current_attribute[0], None, None, None, current_attribute[0] == load.default_target_attribute,
    #                      data_type,
    #                      data_class, current_attribute[1] if data_class is "NOMINAL" else None,
    #                      load.features[col].number_missing_values))

    #for i in range(32):
    #    df_data=pd.concat([df_data, df_data2])
    #    print(i)
    #df_data2=None
    df_data.columns = attribute_list



    dataset.set_data(df_data, atts)



    if os.path.isfile(path):
        import shutil
        shutil.rmtree(path)

    if destpath is None:
        shutil.rmtree(path)

    return dataset



