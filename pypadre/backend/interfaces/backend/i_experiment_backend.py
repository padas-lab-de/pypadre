from abc import ABC, abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_loggable import ILoggable
from pypadre.backend.interfaces.backend.generic.i_progressable import IProgressable
from pypadre.backend.interfaces.backend.generic import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable
from pypadre.backend.interfaces.backend.generic.i_sub_backend import ISubBackend


class IExperimentBackend(IProgressable, ISearchable, IStoreable, ILoggable, ISubBackend):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def execution(self):
        pass

    @abstractmethod
    def put_config(self, obj):
        pass
