from abc import abstractmethod, ABCMeta

from pypadre.core.backend.interfaces.backend.generic.i_base_git_backend import GitBackend
from pypadre.core.backend.interfaces.backend.generic.i_log_backend import ILogBackend
from pypadre.core.backend.interfaces.backend.generic.i_progressable import IProgressable


class IExperimentBackend(GitBackend, ILogBackend, IProgressable):
    """ This is the interface of the experiment backend. """
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def execution(self):
        pass

    @abstractmethod
    def put_config(self, obj):
        pass

    @abstractmethod
    def put(self, object):
        pass
