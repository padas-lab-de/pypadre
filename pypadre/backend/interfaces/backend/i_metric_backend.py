from abc import ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import FileBackend
from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable
from pypadre.base import ChildEntity


# noinspection PyAbstractClass
class IMetricBackend(FileBackend):

    """ This is a backend for metrics """

    __metaclass__ = ABCMeta
