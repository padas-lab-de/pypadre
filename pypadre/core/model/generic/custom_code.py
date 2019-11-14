from abc import ABCMeta, abstractmethod
from typing import Type

from pypadre.core.events.events import Signaler
from pypadre.core.model.code.code_mixin import CodeMixin, Function
from pypadre.core.model.generic.i_executable_mixin import ExecuteableMixin


class CustomCodeHolder(ExecuteableMixin, Signaler):
    """ This is a class holding a custom function. """

    def __init__(self, *args, code: Type[CodeMixin] = None, **kwargs):

        # if code_name is not None:
        #     code_obj = ICode.send_get(self, name=code_name)
        #
        # if code is None:
        #     metadata = {"name": code_name}
        #     if isinstance(code, Callable):
        #         # TODO find a better definition
        #         code = Function(fn=code, metadata=metadata)
        #     elif isinstance(code, str):
        #         code = CodeFile(file_path=code, metadata=metadata)

        if code is None:
            raise ValueError(
                "ICustomCodeSupport needs a code object to reference. This can be provided code but also external code.")

        # if not isinstance(code, EnvCode):
        #     # TODO move put somewhere else?
        #     code.send_put(allow_overwrite=True)

        self._code = code
        super().__init__(*args, **kwargs)

    def _execute_helper(self, *args, **kwargs):
        self.code.send_put(allow_overwrite=True)
        return self.code.call(**kwargs)

    @property
    def code(self):
        return self._code

    # @property
    # def id_hash(self):
    #     return self.code.id


# def _convert_path_to_code_object(path: str, cmd=None):
#     from pypadre.core.model.code.code_file import CodeFile
#     return CodeFile(file_path=path, cmd=cmd)
#
#
# def _convert_function_to_code_object(fn):
#     from pypadre.core.model.code.codemixin import Function
#     return Function(fn=fn)


class CodeManagedMixin:
    """ Class of objects which are derived from a user supplied code block. The code should be versioned and stored
    in a repository. """
    __metaclass__ = ABCMeta

    DEFINED_IN = "defined_in"

    @abstractmethod
    def __init__(self, *args, reference: Type[CodeMixin], **kwargs):
        # if creator is None:
        #     file_path = os.path.realpath(sys.argv[0])
        #     creator = CodeFile(file_path=file_path)
        # self._creator = self.resolve_to_code_object(creator)

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

    # def resolve_to_code_object(self, creator):
    #     # If the user has not specified a creator, get the file that was initially executed
    #     if creator is None:
    #         file_ = os.path.realpath(sys.argv[0])
    #         return _convert_path_to_code_object(path=file_)
    #
    #     # If the user has specified
    #     elif callable(creator):
    #         return _convert_function_to_code_object(fn=creator)
    #
    #     elif isinstance(creator, str) and os.path.exists(creator):
    #         # cmd is set as the __main__ function as default for a file
    #         return _convert_path_to_code_object(path=creator, cmd='__main__')
    #
    #     elif isinstance(creator, Dict):
    #         # TODO: Support for dictionary and command to be introduced
    #         # The dictionary can contain the path to a package or a file
    #         raise ValueError('Dictionary is currently not supported in Custom Code Object')
    #
    #     elif isinstance(creator, CodeMixin):
    #         return creator
    #
    #     else:
    #         raise ValueError('Parameter not supported')
    #     return None


# def component_self_call(ctx, **kwargs):
#     (component,) = unpack(ctx, "component")
#     return component.call(ctx, **kwargs)


class ProvidedCodeHolderMixin(CustomCodeHolder):

    @abstractmethod
    def __init__(self, *, reference: CodeMixin, fn, **kwargs):
        # noinspection PyTypeChecker
        super().__init__(code=Function(fn=fn, identifier=reference.identifier, transient=True), reference=reference, **kwargs)

    @abstractmethod
    def call(self, ctx, **kwargs):
        raise NotImplementedError()

    def _execute_component_code(self, **kwargs):
        return self.code.call(component=self, **kwargs)
