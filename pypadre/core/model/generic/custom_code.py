from abc import ABCMeta, abstractmethod
from typing import Callable, Type, Union

from pypadre.core.events.events import Signaler
from pypadre.core.model.code.code import Code, ProvidedCode
from pypadre.core.model.code.code_file import CodeFile
from pypadre.core.model.code.function import Function


class ICustomCodeSupport(Signaler):
    """ This is a class being created by a managed code file. The code file has to be stored in a git repository and
    versioned. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, code_name: str = None, code: Union[str, Callable, Type[Code]]=None, **kwargs):
        if code is None:
            if code_name is not None:
                code = Code.send_get(self, name=code_name)
            super().__init__(*args, **kwargs)
        else:
            metadata = {"name": code_name}
            if isinstance(code, Callable):
                code = Function(fn=code, metadata=metadata)
            elif isinstance(code, str):
                code = CodeFile(file_path=code, metadata=metadata)
            super().__init__(*args, **kwargs)

            if not isinstance(code, ProvidedCode):
                # TODO move put somewhere else?
                code.send_put(allow_overwrite=True)

        if code is None:
            raise ValueError("ICustomCodeSupport needs a code object to reference. This can be provided code but also "
                             "external code.")
        self._code = code

    @property
    def code(self):
        return self._code


class IGitManagedObject:
    """ Class of objects which are derived from a user supplied code block. The code should be versioned and stored
    in a repository. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, creator: ICustomCodeSupport=None, creator_name: str = None, creator_code: Union[str, Callable, Type[Code]]=None, **kwargs):
        if creator is None:
            creator = ICustomCodeSupport(code_name=creator_name, code=creator_code)
        self._creator = creator

        super().__init__(*args, **kwargs)

    @property
    def creator(self):
        return self._creator
