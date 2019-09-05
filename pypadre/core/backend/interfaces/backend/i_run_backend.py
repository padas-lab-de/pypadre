from abc import abstractmethod, ABCMeta

from pypadre.core.backend.interfaces.backend.generic.i_base_file_backend import FileBackend
from pypadre.core.backend.interfaces.backend.generic.i_log_backend import ILogBackend
from pypadre.core.backend.interfaces.backend.generic.i_progressable import IProgressable


# noinspection PyAbstractClass


class IRunBackend(FileBackend, ILogBackend, IProgressable):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def split(self):
        pass
