from _py_abc import ABCMeta
from abc import abstractmethod

from pypadre.core.base import _CodeTypes
from pypadre.core.model.code.code import Code


class CodeFile(Code):
    """ Interface for a code file (script etc.) which can be executed from python."""
    __metaclass__ = ABCMeta

    def __init__(self, *, path, **kwargs):
        # TODO Add defaults
        defaults = {}

        # TODO Constants into ontology stuff
        # Merge defaults TODO some file metadata extracted from the path
        metadata = {**defaults, **{Code.CODE_TYPE: _CodeTypes.file}, **kwargs.pop("metadata", {})}
        super().__init__(metadata=metadata, **kwargs)
        self._path = path

    @property
    def path(self):
        return self._path

    @abstractmethod
    def call(self):
        raise NotImplementedError()
