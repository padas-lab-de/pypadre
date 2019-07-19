from abc import abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_git_backend import IBaseGitBackend
from pypadre.backend.interfaces.backend.generic.i_base_loggable_file_backend import IBaseLoggableFileBackend
from pypadre.backend.interfaces.backend.generic.i_progressable import IProgressable


class IExperimentBackend(IBaseGitBackend, IBaseLoggableFileBackend, IProgressable):
    """ This is the interface of the experiment backend. """
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def execution(self):
        pass

    @abstractmethod
    def put_config(self, obj):
        pass
