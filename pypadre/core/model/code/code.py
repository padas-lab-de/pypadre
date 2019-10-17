from _py_abc import ABCMeta
from abc import abstractmethod

from pypadre.core.base import MetadataEntity


class Code(MetadataEntity):
    """ Custom code to execute. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError()

    def hash(self):
        return self.__hash__()


class SourceCode(Code):
    """ This code is provided in padre and doesn't have to be stored """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(metadata={"scope": "provided", "ref": str(self.__class__)}, **kwargs)
