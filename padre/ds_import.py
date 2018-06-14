"""
Modul containing import methods from different packages / repositories.
"""
import sklearn.datasets as ds
import numpy as np
from .datasets import Dataset, Attribute
import openml as oml
import os.path
from requests.exceptions import ConnectionError

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


LEGAL_DATA_TYPES = ['nominal', 'numeric', 'string', 'date']

def load_openML_dataset(url,datapath='~/.openml/cache',apikey="1f8766e1615225a727bdea12ad4c72fa"):
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




