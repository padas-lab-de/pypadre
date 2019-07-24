from abc import ABC, abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import IBaseFileBackend
from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable


# noinspection PyAbstractClass
from pypadre.backend.interfaces.backend.generic.i_sub_backend import ISubBackend


class IResultBackend(IBaseFileBackend, ISearchable, IStoreable, ISubBackend):
    __metaclass__ = ABCMeta

