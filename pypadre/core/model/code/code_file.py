import os
from _py_abc import ABCMeta

from pypadre.core.base import _CodeTypes
from pypadre.core.events.events import CommonSignals, signals
from pypadre.core.model.code.icode import ICode


@signals(CommonSignals.HASH)
class CodeFile(ICode):
    """ Interface for a code file or folder (script etc.) which can be executed from python."""

    def hash(self):
        return super().hash()

    __metaclass__ = ABCMeta

    CODE_PATH = "path"
    _hash = None

    def __init__(self, *, path=None, cmd=None, file_path=None, file=None, hash=None, **kwargs):
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
        if hash is not None:
            metadata["hash"] = hash
        super().__init__(metadata=metadata, **kwargs)

        # Set hash. This can be provided by a git repository for example
        self._hash = hash

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

    def hash(self):
        if self._hash is None:
            # Send a signal and ask for the code hash
            dict_object = {'path': self.metadata.get(self.CODE_PATH),
                           'init_repo': False, 'hash_value': None}
            self.send_signal(CommonSignals.HASH, self, **dict_object)
            if self._hash is  None:
                self._hash = super.__hash__()

        return self._hash

    def set_hash(self, hash_value:str):
        self._hash = hash_value






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
