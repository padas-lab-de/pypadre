from abc import abstractmethod, ABCMeta

from pypadre.pod.backend.interfaces.backend.generic.i_base_git_backend import GitBackend


class ISourceBackend(GitBackend):
    """ This is the interface of the execution backend. All data should be stored in a local file system. Currently
    we only store metadata. We store executions here. Executions are to be differentiated on code version (as well as
    dataset version???) and their call command (cluster, local???). These information are to be extracted from parent"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def put(self, source_path, **kwargs):
        pass
