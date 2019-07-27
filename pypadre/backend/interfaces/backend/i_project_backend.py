from abc import abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_git_backend import GitBackend
from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable
from pypadre.base import ChildEntity


class IProjectBackend(GitBackend):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def experiment(self):
        pass
