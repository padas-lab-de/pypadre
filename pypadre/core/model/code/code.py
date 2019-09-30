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
