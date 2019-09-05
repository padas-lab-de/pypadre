from abc import ABCMeta

from pypadre.core.backend.interfaces.backend.generic.i_base_file_backend import FileBackend


# noinspection PyAbstractClass
class IMetricBackend(FileBackend):

    """ This is a backend for metrics """

    __metaclass__ = ABCMeta
