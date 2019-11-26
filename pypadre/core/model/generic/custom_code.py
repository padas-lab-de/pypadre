from abc import ABCMeta, abstractmethod
from typing import Type

from pypadre.core.events.events import Signaler
from pypadre.core.model.code.code_mixin import CodeMixin, Function
from pypadre.core.model.generic.i_executable_mixin import ExecuteableMixin


class CustomCodeHolder(ExecuteableMixin, Signaler):
    """ This is a class holding a custom function. """

    def __init__(self, *args, code: Type[CodeMixin] = None, **kwargs):
        if code is None:
            raise ValueError(
                "ICustomCodeSupport needs a code object to reference. This can be provided code but also external code.")

        self._code = code
        super().__init__(*args, **kwargs)

    def _execute_helper(self, *args, **kwargs):
        self.code.send_put(allow_overwrite=True)
        return self.code.call(**kwargs)

    @property
    def code(self):
        return self._code


class CodeManagedMixin:
    """ Class of objects which are derived from a user supplied code block. The code should be versioned and stored
    in a repository. """
    __metaclass__ = ABCMeta

    DEFINED_IN = "defined_in"

    @abstractmethod
    def __init__(self, *args, reference: Type[CodeMixin], **kwargs):
        self._reference = reference

        metadata = {**kwargs.pop("metadata", {}), **{self.DEFINED_IN: reference.id}}
        super().__init__(*args, metadata=metadata, **kwargs)
        self.reference.send_put()

    @property
    def reference(self):
        return self._reference

    @property
    def reference_hash(self):
        return self.reference.id


class ProvidedCodeHolderMixin(CustomCodeHolder):

    @abstractmethod
    def __init__(self, *, reference: CodeMixin, fn, **kwargs):
        # noinspection PyTypeChecker
        super().__init__(code=Function(fn=fn, repository_identifier=reference.repository_identifier, transient=True),
                         reference=reference, **kwargs)

    @abstractmethod
    def call(self, ctx, **kwargs):
        raise NotImplementedError()

    def _execute_component_code(self, **kwargs):
        return self.code.call(component=self, **kwargs)
