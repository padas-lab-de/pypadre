from abc import abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_git_backend import IBaseGitBackend
from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable
from pypadre.backend.interfaces.backend.generic.i_sub_backend import ISubBackend


# noinspection PyAbstractClass
class IDatasetBackend(IBaseGitBackend, ISearchable, IStoreable, ISubBackend):
    __metaclass__ = ABCMeta
