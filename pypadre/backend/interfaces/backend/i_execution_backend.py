from abc import ABC, abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_meta_file_backend import IBaseMetaFileBackend
from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable
from pypadre.backend.interfaces.backend.generic.i_sub_backend import ISubBackend


class IExecutionBackend(IBaseMetaFileBackend):
    """ This is the interface of the execution backend. All data should be stored in a local file system. Currently
    we only store metadata. We store executions here. Executions are to be differentiated on code version (as well as
    dataset version???) and their call command (cluster, local???). These information are to be extracted from parent"""
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def run(self):
        pass
