from sklearn import pipeline
from sklearn import base


# TODO wrapper for sklearn pipelines and estimators should be linked with owlready2
class Pipeline(pipeline.Pipeline):

    def __init__(self, steps):
        super().__init__(steps)


class BaseEstimator(base.BaseEstimator):

    def __init__(self):
        super().__init__()
