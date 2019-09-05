from abc import abstractmethod, ABCMeta

from pypadre.pod.backend.interfaces.backend.generic.i_base_file_backend import FileBackend
from pypadre.pod.backend.interfaces.backend.generic.i_log_backend import ILogBackend
from pypadre.pod.backend.interfaces.backend.generic.i_progressable import IProgressable


# noinspection PyAbstractClass


class ISplitBackend(FileBackend, ILogBackend, IProgressable):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def result(self):
        pass

    @property
    @abstractmethod
    def metric(self):
        pass
