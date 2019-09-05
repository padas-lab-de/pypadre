from abc import ABCMeta

from pypadre.pod.backend.interfaces.backend.generic.i_base_file_backend import FileBackend


# noinspection PyAbstractClass
class IResultBackend(FileBackend):
    __metaclass__ = ABCMeta
