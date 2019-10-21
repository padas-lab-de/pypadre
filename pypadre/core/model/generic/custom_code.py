from abc import ABCMeta, abstractmethod
from typing import Callable, Type, Union

from pypadre.core.model.code.code import Code, ProvidedCode
from pypadre.core.model.code.code_file import CodeFile
from pypadre.core.model.code.function import Function


class CustomCodeSupport:
    """ This is a class being created by a managed code file. The code file has to be stored in a git repository and
    versioned. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, store_code: bool=False, code_name: str = None, code: Union[str, Callable, Type[Code]]=None, **kwargs):
        if code is None:
            if code_name is not None:
                # TODO load code from store?
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
