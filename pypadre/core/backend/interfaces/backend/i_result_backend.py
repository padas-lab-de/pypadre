from abc import ABCMeta

from pypadre.core.backend.interfaces.backend.generic.i_base_file_backend import FileBackend


# noinspection PyAbstractClass
class IResultBackend(FileBackend):
    __metaclass__ = ABCMeta
