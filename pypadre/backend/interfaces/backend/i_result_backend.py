from abc import ABC, abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import FileBackend
from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable


# noinspection PyAbstractClass
from pypadre.base import ChildEntity


class IResultBackend(FileBackend, ISearchable, IStoreable, ChildEntity):
    __metaclass__ = ABCMeta

