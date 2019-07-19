from abc import ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_git_backend import IBaseGitBackend


# noinspection PyAbstractClass
class IDatasetBackend(IBaseGitBackend):
    """ This is the interface of a data set backend. Data sets meta information should be stored in git. The data
    set itself can only be stored in something like git lfs"""
    __metaclass__ = ABCMeta
