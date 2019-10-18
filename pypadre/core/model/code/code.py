from _py_abc import ABCMeta
from abc import abstractmethod

from pypadre.core.base import MetadataEntity, _CodeTypes
from pypadre.core.model.generic.i_model_mixins import IStoreable


class Code(IStoreable, MetadataEntity):
    """ Custom code to execute. """
    __metaclass__ = ABCMeta

    CODE_TYPE = "code_type"
    CODE_CLASS = "code_class"

    def __init__(self, *, metadata: dict, **kwargs):
        # TODO Add defaults
        defaults = {}

        # TODO Constants into ontology stuff
        # Merge defaults TODO some file metadata extracted from the path
        metadata = {**defaults, **{Code.CODE_TYPE: _CodeTypes.env, Code.CODE_CLASS: str(self.__class__)}, **kwargs.pop("metadata", {})}
        super().__init__(metadata=metadata, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError()

    def get_bin(self):
        raise NotImplementedError()

    def hash(self):
        return self.__hash__()

    @property
    def code_type(self):
        return self.metadata.get(self.CODE_TYPE)

# Code should be given by one of the following ways: A file (local, remote), a function to be persisted, a function
# on the environment


class EnvCode(Code):
    """ This code is provided in environment and doesn't have to be stored """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        # TODO Add defaults
        defaults = {}

        # TODO Constants into ontology stuff
        # Merge defaults TODO some file metadata extracted from the path
        metadata = {**defaults, **{Code.CODE_TYPE: _CodeTypes.env}, **kwargs.pop("metadata", {})}
        super().__init__(metadata=metadata, **kwargs)
