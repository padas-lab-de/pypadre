from abc import ABCMeta, abstractmethod
from typing import Callable, Type, Union

from pypadre.core.model.code.code import Code, ProvidedCode
from pypadre.core.model.code.code_file import CodeFile
from pypadre.core.model.code.function import Function


class ICustomCodeSupport:
    """ This is a class being created by a managed code file. The code file has to be stored in a git repository and
    versioned. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, store_code: bool=False, code_name: str = None, code: Union[str, Callable, Type[Code]]=None, **kwargs):
        if code is None:
            if code_name is not None:
                # TODO load code object from store. Fail if we didn't find it.
                pass
            pass
            super().__init__(*args, **kwargs)
        else:
            metadata = {"name": code_name}
            if isinstance(code, Callable):
                code = Function(fn=code, metadata=metadata)
            elif isinstance(code, str):
                code = CodeFile(file_path=code, metadata=metadata)
            self._code = code
            super().__init__(*args, **kwargs)

            if not isinstance(code, ProvidedCode) and store_code:
                code.send_put(allow_overwrite=True)

    @property
    def code(self):
        return self._code


class IGitManagedObject:
    """ Class of objects which are derived from a user supplied code block. The code should be versioned and stored
    in a repository. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, creator: ICustomCodeSupport=None, store_creator: bool=False, creator_name: str = None, creator_code: Union[str, Callable, Type[Code]]=None, **kwargs):
        if creator is None:
            creator = ICustomCodeSupport(store_code=store_creator, code_name=creator_name, code=creator_code)
        self._creator = creator

    @property
    def creator(self):
        return self._creator
