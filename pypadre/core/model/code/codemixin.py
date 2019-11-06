from _py_abc import ABCMeta
from abc import abstractmethod
from typing import Callable

from pypadre.core.base import MetadataMixin, _CodeTypes
from pypadre.core.model.generic.i_model_mixins import StoreableMixin
from pypadre.core.pickling.pickle_base import Pickleable


class CodeMixin(StoreableMixin, MetadataMixin):
    """ Custom code to execute. """
    __metaclass__ = ABCMeta

    CODE_TYPE = "code_type"
    CODE_CLASS = "code_class"

    def __init__(self, **kwargs):
        # TODO Add defaults
        defaults = {}

        # TODO Constants into ontology stuff
        # Merge defaults TODO some file metadata extracted from the path
        metadata = {**defaults, **{CodeMixin.CODE_TYPE: _CodeTypes.env, CodeMixin.CODE_CLASS: str(self.__class__)}, **kwargs.pop("metadata", {})}
        super().__init__(metadata=metadata, **kwargs)

    @abstractmethod
    def _call(self, ctx, **kwargs):
        raise NotImplementedError()

    def call(self, **kwargs):
        parameters = kwargs.pop("parameters", {})
        # kwargs are the padre context to be used
        return self._call(kwargs, **parameters)

    @abstractmethod
    def hash(self):
        raise NotImplementedError()

    @property
    def code_type(self):
        return self.metadata.get(self.CODE_TYPE)

# Code should be given by one of the following ways: A file (local, remote), a function to be persisted, a function
# on the environment


class Function(CodeMixin):
    """ Function to execute."""

    def hash(self):
        return self.fn.__hash__()

    def __init__(self, *, fn: Callable, **kwargs):
        # TODO Add defaults
        defaults = {"name": fn.__name__}

        # TODO Constants into ontology stuff
        # Merge defaults TODO some file metadata extracted from the path
        metadata = {**defaults, **{CodeMixin.CODE_TYPE: _CodeTypes.fn}, **kwargs.pop("metadata", {})}
        super().__init__(metadata=metadata, **kwargs)
        self._fn = fn

    # TODO don't dill the function but use a function-store which can be initialized by the backends. The store
    # itself sends a request for init and then tries to load the fn by a hash value identifier?

    @property
    def fn(self):
        # TODO we could dill this serialize https://medium.com/@emlynoregan/serialising-all-the-functions-in-python-cd880a63b591 or write the maximum of possible information and warn the user about no possibility to reload
        return self._fn

    def _call(self, ctx, **kwargs):
        return self.fn(ctx, **kwargs)


class EnvCode(Function, Pickleable):
    """ This code is provided in environment and doesn't have to be stored """
    __metaclass__ = ABCMeta

    PACKAGE = "package"
    FUNCTION_NAME = "function_name"

    @abstractmethod
    def __init__(self, *, fn_name, package, requirement, version, **kwargs):

        # TODO check if requirement is full filled and then load it (version of package)
        self._requirement = requirement
        self._version = version
        import importlib.util
        importlib.import_module(package)
        fn = getattr(importlib.import_module(package), fn_name)

        defaults = {}
        # TODO Constants into ontology stuff
        metadata = {**defaults, **{CodeMixin.CODE_TYPE: _CodeTypes.env, self.PACKAGE: package, self.FUNCTION_NAME: fn_name}, **kwargs.pop("metadata", {})}
        super().__init__(metadata=metadata, fn=fn, **kwargs)

    @property
    def requirement(self):
        return self._requirement

    @property
    def version(self):
        return self._version

    def transient_fields(self):
        return ["_fn"]
