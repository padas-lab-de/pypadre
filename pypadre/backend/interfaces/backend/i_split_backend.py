from abc import abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_progressable import IProgressable
from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable


# noinspection PyAbstractClass
from pypadre.backend.interfaces.backend.generic.i_sub_backend import ISubBackend


class ISplitBackend(IProgressable, ISearchable, IStoreable, ISubBackend):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def result(self):
        pass

    @property
    @abstractmethod
    def metric(self):
        pass
