import os
from abc import abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_meta_file_backend import IBaseMetaFileBackend
from pypadre.backend.interfaces.backend.generic.i_sub_backend import ISubBackend
from pypadre.backend.serialiser import PickleSerializer, JSonSerializer
from pypadre.util.file_util import get_path


class IBaseGitBackend(IBaseMetaFileBackend):
    __metaclass__ = ABCMeta

    @abstractmethod
    def _init(self):
        pass

    @abstractmethod
    def _commit(self):
        pass

