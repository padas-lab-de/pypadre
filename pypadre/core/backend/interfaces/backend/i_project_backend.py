from abc import abstractmethod, ABCMeta

from pypadre.core.backend.interfaces.backend.generic.i_base_git_backend import GitBackend


class IProjectBackend(GitBackend):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def experiment(self):
        pass
