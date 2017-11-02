"""
Modul containing import methods from different packages / repositories.
"""
import sklearn.datasets as ds
from .datasets import NumpyBaseDataset

# TODO: import standard sklearn data sets
# TODO: import data sets from mldata and openml
# TODO: Import different embeddings
# TODO: check consistency of datasets from different repositories
# TODO: replace dataset instantiation with factory pattern

def load_sklearn_toys():
    """
    returns an iterator loading different sklearn datasets
    """
    def _title(s):
        s = s.strip()
        return s[0:s.find("\n")]
        
    loaders = [(ds.load_boston, ("regression", "multivariate")),
               (ds.load_breast_cancer, ("classification", "multivariate")),
               (ds.load_diabetes, ("regression", "multivariate")),
               (ds.load_digits, ("classification", "multivariate")),
               (ds.load_iris, ("classification", "multivariate")),
#              ds.load_lfw_pairs,
#              ds.load_lfw_people,
               (ds.load_linnerud, ("mregression", "multivariate"))]

    for loader in loaders:
        bunch = loader[0]()
        if "DESCR" not in bunch:
            # TODO: extract only the function name (not also its address)
            bunch["DESCR"] = str(loader[0]) + "\n" + str("SK Learn Import. No Description available")
        fn = bunch.get("feature_names")
        if fn is not None:
            fn = [i for i in fn]
        yield NumpyBaseDataset(_title(bunch["DESCR"]), bunch.data, bunch.target,
                               attributes=fn,
                               description=bunch.get("DESCR"),
                               task=loader[1][0],
                               type=loader[1][1])






    
