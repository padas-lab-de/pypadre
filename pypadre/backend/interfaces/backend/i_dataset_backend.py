from abc import ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_git_backend import IBaseGitBackend


# noinspection PyAbstractClass
class IDatasetBackend(IBaseGitBackend):
    __metaclass__ = ABCMeta
