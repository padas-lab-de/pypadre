from abc import abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable
from pypadre.backend.interfaces.backend.generic.i_sub_backend import ISubBackend


class IDatasetBackend(ISearchable, IStoreable, ISubBackend):
    __metaclass__ = ABCMeta
