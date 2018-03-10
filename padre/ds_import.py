"""
Modul containing import methods from different packages / repositories.
"""
import sklearn.datasets as ds
import numpy as np
from .datasets import Dataset, Attribute


# TODO: import standard sklearn data sets
# TODO: import data sets from mldata and openml
# TODO: Import different embeddings
# TODO: check consistency of datasets from different repositories
# TODO: replace dataset instantiation with factory pattern

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






    
