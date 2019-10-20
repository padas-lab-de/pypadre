from _py_abc import ABCMeta
from abc import abstractmethod

from pypadre.core.base import MetadataEntity, _CodeTypes
from pypadre.core.model.generic.i_model_mixins import IStoreable


class Code(IStoreable, MetadataEntity):
    """ Custom code to execute. """
    __metaclass__ = ABCMeta

    CODE_TYPE = "code_type"
    CODE_CLASS = "code_class"

    def __init__(self, **kwargs):
        # TODO Add defaults
        defaults = {}

        # TODO Constants into ontology stuff
        # Merge defaults TODO some file metadata extracted from the path
        metadata = {**defaults, **{Code.CODE_TYPE: _CodeTypes.env, Code.CODE_CLASS: str(self.__class__)}, **kwargs.pop("metadata", {})}
        super().__init__(metadata=metadata, **kwargs)
        self.send_put()

    @abstractmethod
    def _call(self, ctx, **kwargs):
        raise NotImplementedError()

    def call(self, **kwargs):
        parameters = kwargs.pop("parameters", {})
        # kwargs are the padre context to be used
        return self._call(kwargs, **parameters)

    def get_bin(self):
        raise NotImplementedError()

    def hash(self):
        return self.__hash__()

    @property
    def code_type(self):
        return self.metadata.get(self.CODE_TYPE)

# Code should be given by one of the following ways: A file (local, remote), a function to be persisted, a function
# on the environment


class ProvidedCode(Code):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        defaults = {}
        # TODO save data about runtime versions / libraries etc for reproducibility
        metadata = {**defaults, **{Code.CODE_TYPE: _CodeTypes.provided},
                    **kwargs.pop("metadata", {})}
        super().__init__(metadata=metadata, **kwargs)
