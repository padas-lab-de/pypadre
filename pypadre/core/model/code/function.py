from typing import Callable

from pypadre.core.base import _CodeTypes
from pypadre.core.model.code.code import Code


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

    def call(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
