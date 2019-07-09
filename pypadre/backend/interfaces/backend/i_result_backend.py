from abc import ABC, abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_meta_file_backend import IBaseMetaFileBackend
from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable


# noinspection PyAbstractClass
from pypadre.backend.interfaces.backend.generic.i_sub_backend import ISubBackend


class IResultBackend(IBaseMetaFileBackend, ISearchable, IStoreable, ISubBackend):
    __metaclass__ = ABCMeta

