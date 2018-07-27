"""
Modul containing import methods from different packages / repositories.
"""
import sklearn.datasets as ds
import numpy as np
from .datasets import Dataset, Attribute
import openml as oml
import os.path
from requests.exceptions import ConnectionError
import arff
import pandas as pd


# TODO: import standard sklearn data sets
# TODO: import data sets from mldata and openml
# TODO: Import different embeddings
# TODO: check consistency of datasets from different repositories
# TODO: replace dataset instantiation with factory pattern

def _split_DESCR(s):
    s = s.strip()
    k = s.find("\n")
    return s[0:k], s[k + 1:]

"""avoid code redundany by adding a check, if target in bunch dictionary"""
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


def load_sklearn_toys():
    """
    returns an iterator loading different sklearn datasets
    """

        
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


LEGAL_DATA_TYPES = ['NOMINAL','INTEGER', 'NUMERIC', 'REAL', 'STRING']

DATATYPE_MAP={'INTEGER': np.int64,'NUMERIC':np.float64, 'REAL':np.float64, 'STRING': str}

def load_openML_dataset_old(url,datapath='~/.openml/cache',apikey="1f8766e1615225a727bdea12ad4c72fa"):
    #apikey is from useraccount markush.473@gmail.com

    dataset_id=url.split("/")[-1]
    oml.config.apikey=apikey
    oml.config.cache_directory=os.path.expanduser(datapath)
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
    meta["openml_url"] = load.url
    meta["default_target_attribute"]=load.default_target_attribute
    meta["version_label"] = load.version_label
    meta["citation"] = load.citation
    meta["paper_url"] = load.paper_url
    meta["update_comment"] = load.update_comment
    meta["qualities"] = load.qualities


    dataset = Dataset(None, **meta)


    data = load.get_data()
    if(type(data) is not np.ndarray):
        try:
            data=data.toarray()
        except AttributeError as err:
            print("Invalid datatype of dataset! \nErrormessage: " + str(err))
            return None

    atts = []

    for ix in range(0, data.shape[1]+1):
        if (load.features[ix].name == load.row_id_attribute):
            continue

        atts.append(Attribute(load.features[ix].name, None, None, None, load.features[ix].name == load.default_target_attribute,
                              load.features[ix].data_type if (load.features[ix].data_type in LEGAL_DATA_TYPES) else None,
                              load.features[ix].number_missing_values))
    dataset.set_data(data, atts)
    return dataset

def getTop100():
    file = open("../padre/top100datasetIDs.txt", "r")
    id_list = file.read().split(",")

    datasets = []
    i=0
    amount=len(id_list)
    for id in id_list:
        print("Progress: ("+str(i)+"/"+str(amount)+") id of next dataset:" + str(id))
        datasets.append(load_openML_dataset_new("/" + id))
        i=i+1
    return datasets

def load_openML_dataset_new(url,datapath='~/.openml/cache',apikey="1f8766e1615225a727bdea12ad4c72fa"):
    print("start")
    #apikey is from useraccount markush.473@gmail.com
    #raw_data = arff.load(open(os.path.expanduser('~/.openml/cache')+'/org/openml/www/datasets/14/dataset.arff',encoding='utf-8'))
    dataset_id=url.split("/")[-1]
    dataset_id=dataset_id.strip(" ")
    oml.config.apikey=apikey
    oml.config.cache_directory=os.path.expanduser(datapath)
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
    meta["openml_url"] = load.url
    meta["default_target_attribute"]=load.default_target_attribute
    meta["version_label"] = load.version_label
    meta["citation"] = load.citation
    meta["paper_url"] = load.paper_url
    meta["update_comment"] = load.update_comment
    meta["qualities"] = load.qualities

    print("meta done")
    dataset = Dataset(None, **meta)

    raw_data = arff.load(open(os.path.expanduser(datapath)+'/org/openml/www/datasets/'+dataset_id+'/dataset.arff',encoding='utf-8'))
    print("arff loaded")
    df_attributes = raw_data['attributes']
    attribute_list = []

    for att in df_attributes:
        attribute_list.append(att[0])
    print("dataframe")
    df_data = pd.DataFrame(data=raw_data['data'])
    print("gotDataframe")
    #list(df_data)
    #enumerate(df_data.itertuples())
    raw_data = None


    atts = []
    for col in df_data.keys():


        print(df_attributes[col][0]+" "+str((df_data[col]).dtype))
        #data_type=None
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
    return dataset



