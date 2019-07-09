from abc import abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable
from pypadre.backend.interfaces.backend.generic.i_sub_backend import ISubBackend


class IProjectBackend(ISearchable, IStoreable, ISubBackend):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def experiment(self):
        pass
