from _py_abc import ABCMeta
from typing import Callable

from pypadre.core.model.code.code import Code


class Function(Code):
    """ Function to execute."""
    __metaclass__ = ABCMeta

    def __init__(self, *, fn: Callable, **kwargs):
        # TODO Add defaults
        defaults = {}

        # Merge defaults TODO some fn metadata extracted from the fn
        metadata = {**defaults, **kwargs.pop("metadata", {})}
        super().__init__(metadata=metadata, **kwargs)
        self._fn = fn

    @property
    def fn(self):
        return self._fn

    def call(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
