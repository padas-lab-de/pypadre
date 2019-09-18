from abc import ABCMeta, abstractmethod

from networkx import DiGraph
from sklearn import pipeline
from sklearn import base


class PipelineComponent:
    __metaclass__ = ABCMeta

    def __init__(self, sources):
        self._sources = sources

    @property
    def sources(self):
        return self._sources

    @abstractmethod
    def compute(self, data, **kwargs):
        pass


# TODO wrapper for sklearn pipelines and estimators should be linked with owlready2, add own workflow / pipeline definition?
class Pipeline(DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)

    def compute(self):
        # TODO networkx for pipelines? / How to match metrics calculation (maybe also with networkx?)
        entries = [node for node, in_degree in self.in_degree() if in_degree == 0]



