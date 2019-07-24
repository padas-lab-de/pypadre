from abc import ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import IBaseFileBackend
from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable
from pypadre.backend.interfaces.backend.generic.i_sub_backend import ISubBackend


class IMetricBackend(IBaseFileBackend, ISearchable, IStoreable, ISubBackend):

    """ This is a backend for metrics """

    __metaclass__ = ABCMeta
