import os
from abc import ABC, abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import IBaseFileBackend


class ILogBackend:
    """ This is the interface for all backends which are able to log interactions into some kind of log store """
    __metaclass__ = ABCMeta

    @abstractmethod
    def log(self, msg):
        pass
