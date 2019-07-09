from abc import ABC, abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable
from pypadre.backend.interfaces.backend.generic.i_sub_backend import ISubBackend


class IExecutionBackend(ISearchable, IStoreable, ISubBackend):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def run(self):
        pass
