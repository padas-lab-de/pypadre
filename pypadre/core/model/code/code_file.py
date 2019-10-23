import os
from _py_abc import ABCMeta

from pypadre.core.base import _CodeTypes
from pypadre.core.model.code.icode import ICode


class CodeFile(ICode):
    """ Interface for a code file or folder (script etc.) which can be executed from python."""
    __metaclass__ = ABCMeta

    CODE_PATH = "path"

    def __init__(self, *, path=None, cmd=None, file_path=None, file=None, **kwargs):
        # TODO Add defaults

        if file_path:
            path = os.path.dirname(file_path)
            file = os.path.basename(file_path)

        if file and cmd is None:
            cmd = file

        if path is None or cmd is None:
            raise ValueError("path and cmd need to be set")

        defaults = {}

        metadata = {**defaults, **{ICode.CODE_TYPE: _CodeTypes.file, self.CODE_PATH: path}, **kwargs.pop("metadata", {})}
        if file is not None:
            metadata["file"] = file
        if cmd is not None:
            metadata["cmd"] = cmd

        super().__init__(metadata=metadata, **kwargs)

    @property
    def file(self):
        return self.metadata.get("file", None)

    @property
    def path(self):
        return self.metadata.get("path", None)

    @property
    def cmd(self):
        return self.metadata.get("cmd", None)

    def _call(self, ctx, **kwargs):
        return os.system(self.cmd)

#
# class CodeFolder(CodeFile):
#     """ Interface for a code file (script etc.) which can be executed from python."""
#
#     __metaclass__ = ABCMeta
#
#     def __init__(self, *, entry_point, **kwargs):
#         # TODO Add defaults
#         defaults = {}
#
#         # TODO Constants into ontology stuff
#         # Merge defaults TODO some file metadata extracted from the path
#         metadata = {**defaults, **{Code.CODE_TYPE: _CodeTypes.file}, **kwargs.pop("metadata", {})}
#         super().__init__(metadata=metadata, **kwargs)
#         self._entry_point = entry_point
#
#     @property
#     def entry_point(self):
#         return self._entry_point
