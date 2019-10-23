from abc import ABCMeta, abstractmethod
from typing import Callable, Type, Union

from pypadre.core.events.events import Signaler
from pypadre.core.model.code.code_file import CodeFile
from pypadre.core.model.code.function import Function
from pypadre.core.model.code.icode import ICode, IProvidedCode
from pypadre.core.model.generic.i_executable_mixin import IExecuteable


class ICustomCodeSupport(IExecuteable, Signaler):
    """ This is a class being created by a managed code file. The code file has to be stored in a git repository and
    versioned. """

    def __init__(self, *args, code_name: str = None, code: Union[str, Callable, Type[ICode]]=None, **kwargs):

        if code_name is not None:
            code_obj = ICode.send_get(self, name=code_name)

        if code_obj is None:
            metadata = {"name": code_name}
            if isinstance(code, Callable):
                code = Function(fn=code, metadata=metadata)
            elif isinstance(code, str):
                code = CodeFile(file_path=code, metadata=metadata)

        if code is None:
            raise ValueError("ICustomCodeSupport needs a code object to reference. This can be provided code but also external code.")

        if not isinstance(code, IProvidedCode):
            # TODO move put somewhere else?
            code.send_put(allow_overwrite=True)

        super().__init__(*args, **kwargs)
        self._code = code

    def _execute_helper(self, *args, **kwargs):
        return self.code.call(**kwargs)

    @property
    def code(self):
        return self._code

    @property
    def hash(self):
        return self.code.hash()


class ICodeManagedObject:
    """ Class of objects which are derived from a user supplied code block. The code should be versioned and stored
    in a repository. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, creator: ICustomCodeSupport=None, creator_name: str = None, creator_code: Union[str, Callable, Type[ICode]]=None, **kwargs):
        if creator is None:
            creator = ICustomCodeSupport(code_name=creator_name, code=creator_code)
        self._creator = creator

        super().__init__(*args, **kwargs)

    @property
    def creator(self):
        return self._creator

    @property
    def creator_hash(self):
        return self.creator.hash()
