from _py_abc import ABCMeta
from abc import abstractmethod
from typing import Callable

from pypadre.core.base import _CodeTypes
from pypadre.core.model.code.code import Code
from pypadre.core.pickling.pickle_base import Pickleable


class Function(Code):
    """ Function to execute."""

    def __init__(self, *, fn: Callable, **kwargs):
        # TODO Add defaults
        defaults = {}

        # TODO Constants into ontology stuff
        # Merge defaults TODO some file metadata extracted from the path
        metadata = {**defaults, **{Code.CODE_TYPE: _CodeTypes.fn}, **kwargs.pop("metadata", {})}
        super().__init__(metadata=metadata, **kwargs)
        self._fn = fn

    @property
    def fn(self):
        # TODO serialize https://medium.com/@emlynoregan/serialising-all-the-functions-in-python-cd880a63b591 or write the maximum of possible information and warn the user about no possibility to reload
        return self._fn

    def _call(self, ctx, **kwargs):
        return self.fn(ctx, **kwargs)


class EnvCode(Function, Pickleable):
    """ This code is provided in environment and doesn't have to be stored """
    __metaclass__ = ABCMeta

    PACKAGE = "package"
    FUNCTION_NAME = "function_name"

    @abstractmethod
    def __init__(self, *, fn_name, package, **kwargs):
        import importlib.util
        spec = importlib.util.find_spec(fn_name, package=package)
        foo = importlib.util.module_from_spec(spec)
        fn = spec.loader.exec_module(foo)

        defaults = {}
        # TODO Constants into ontology stuff
        metadata = {**defaults, **{Code.CODE_TYPE: _CodeTypes.env, self.PACKAGE: package, self.FUNCTION_NAME: fn_name}, **kwargs.pop("metadata", {})}
        super().__init__(metadata=metadata, fn=fn, **kwargs)

    def transient_fields(self):
        return ["_fn"]
